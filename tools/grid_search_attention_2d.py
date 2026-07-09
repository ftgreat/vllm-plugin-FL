#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.
# Grid search over tunable knobs for kernel_unified_attention_2d.
# Supports multi-GPU: configs are evenly distributed across all visible GPUs.
# Supports both full-tensor samples and lightweight (shape-only) samples.
#
# Usage:
#   # Default: sweep each knob independently (~40 configs), all GPUs
#   python tools/grid_search_attention_2d.py --data-dir /path/to/saved_samples
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/grid_search_attention_2d.py \
#       --data-dir /path/to/saved_samples
#
#   # Sweep specific knobs
#   python tools/grid_search_attention_2d.py --data-dir /path/to/saved_samples \
#       --sweep TILE_SIZE=16,32,64 --sweep num_warps=1,2,4,8
#
#   # Full grid over specified knobs
#   python tools/grid_search_attention_2d.py --data-dir /path/to/saved_samples \
#       --grid \
#       --sweep TILE_SIZE=32,64,128 \
#       --sweep num_warps=2,4 \
#       --sweep matrix_instr_nonkdim=0,16 \
#       --sweep kpack=1,2
#
#   # Filter by batch size and sequence length
#   python tools/grid_search_attention_2d.py --data-dir /path/to/saved_samples \
#       --filter-bs 64 --filter-max-seqlen-range 1000,5000

import argparse
import csv
import itertools
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════
# Default knob values (matching production defaults)
# ═══════════════════════════════════════════════════════════════════
DEFAULTS = {
    # Tiling (constexpr — triggers recompilation)
    "TILE_SIZE": 64,
    "BLOCK_M": 128,
    # Triton launch params
    "num_warps": 8,
    "num_stages": 2,
    # HCU compiler extra_kargs (defaults from HIPOptions in compiler_hcu.py)
    "matrix_instr_nonkdim": 0,
    "kpack": 1,
    "waves_per_eu": 1,
    "schedule_hint": "none",
    "sched_latency": "none",
    "mmac_layout_force": -1,
}

# ═══════════════════════════════════════════════════════════════════
# Search space for each knob (used in independent sweep mode)
# ═══════════════════════════════════════════════════════════════════
SEARCH_SPACE = {
    "TILE_SIZE": [32, 64, 128],
    "BLOCK_M": [16, 32, 64, 128, 256],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 4],
    "matrix_instr_nonkdim": [0, 16, 32],
    "kpack": [1, 2],
    "waves_per_eu": [0, 1, 2, 4],
    "schedule_hint": ["none", "attention", "memory-bound-attention"],
    "sched_latency": ["none", "mmac5-ds10", "mmac5-ds6"],
    "mmac_layout_force": [-1, 0, 1, 2, 3, 4],
}

# Knobs passed as Triton HCU compiler extra_kargs
EXTRA_KARG_KNOBS = {
    "matrix_instr_nonkdim", "kpack", "waves_per_eu",
    "schedule_hint", "sched_latency", "mmac_layout_force",
}

# Defaults that mean "no-op" / "auto" (from HIPOptions in compiler_hcu.py)
_EXTRA_KARG_DEFAULTS = {
    "matrix_instr_nonkdim": 0,
    "kpack": 1,
    "waves_per_eu": 1,
    "schedule_hint": "none",
    "sched_latency": "none",
    "mmac_layout_force": -1,
}

WARMUP_ITERS = 3
TIMED_ITERS = 10
KERNEL_TIMEOUT_US = 10_000_000  # per-kernel-invocation timeout in microseconds (10s)

# ═══════════════════════════════════════════════════════════════════
# Dtype mapping for lightweight samples
# ═══════════════════════════════════════════════════════════════════
_DTYPE_MAP = {
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
}


def load_samples(data_dir: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load all .pt sample files from directory."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("attn_sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No attn_sample_*.pt files in {data_dir}")
    if max_samples is not None:
        files = files[:max_samples]
    samples = []
    for pt_file in files:
        sample = torch.load(pt_file, map_location="cpu", weights_only=False)
        samples.append(sample)
    return samples


def filter_samples(
    samples: List[Dict[str, Any]],
    filter_bs: int = None,
    filter_bs_range: str = None,
    filter_max_seqlen_range: str = None,
) -> List[Dict[str, Any]]:
    """Filter samples by num_seqs and max_seq_len."""
    filtered = samples
    if filter_bs is not None:
        filtered = [s for s in filtered if s["num_seqs"] == filter_bs]
    if filter_bs_range is not None:
        lo, hi = map(int, filter_bs_range.split(","))
        filtered = [s for s in filtered if lo <= s["num_seqs"] <= hi]
    if filter_max_seqlen_range is not None:
        lo, hi = map(int, filter_max_seqlen_range.split(","))
        filtered = [s for s in filtered
                    if lo <= s.get("max_seq_len", int(s["seqused_k"].max().item())) <= hi]
    return filtered


def _build_sequential_block_table(seqused_k, block_size, device):
    """Build a sequential block_table mapping for synthetic data."""
    num_seqs = seqused_k.shape[0]
    num_blocks_per_seq = (seqused_k + block_size - 1) // block_size
    max_blocks = int(num_blocks_per_seq.max().item())
    total_blocks = int(num_blocks_per_seq.sum().item())

    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
    offset = 0
    for i in range(num_seqs):
        n = int(num_blocks_per_seq[i].item())
        block_table[i, :n] = torch.arange(offset, offset + n, dtype=torch.int32, device=device)
        offset += n
    return block_table, total_blocks


def prepare_kernel_args_lite(
    sample: Dict[str, Any], config: Dict[str, Any],
) -> Tuple[tuple, dict, "torch.Tensor"]:
    """Prepare kernel args from a lightweight (shape-only) sample using synthetic data."""
    # Parse dtype
    dtype_str = sample["q_dtype"]
    dtype = getattr(torch, _DTYPE_MAP.get(dtype_str, dtype_str.replace("torch.", "")))

    num_query_heads = sample["num_query_heads"]
    num_kv_heads = sample["num_kv_heads"]
    num_queries_per_kv = sample["num_queries_per_kv"]
    head_size = sample["head_size"]
    block_size = sample["block_size"]
    num_seqs = sample["num_seqs"]
    softmax_scale = sample["softmax_scale"]
    softcap = sample["softcap"]
    output_scale = sample.get("output_scale")
    softmax_threshold_val = sample["softmax_threshold_val"]
    use_qq_bias = sample["USE_QQ_BIAS"]

    seqused_k = sample["seqused_k"].cuda()
    cu_seqlens_q = sample["cu_seqlens_q"].cuda()

    # Construct synthetic tensors from recorded shapes
    q_shape = tuple(sample["q_shape"])
    q = torch.randn(q_shape, dtype=dtype, device="cuda")
    out = torch.empty_like(q)

    # Build sequential block_table and KV cache
    block_table, total_blocks = _build_sequential_block_table(seqused_k, block_size, q.device)
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device="cuda")
    v_cache = torch.randn_like(k_cache)

    # Tiling from config
    TILE_SIZE = config["TILE_SIZE"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_Q = max(BLOCK_M // num_queries_per_kv, 1)

    # Recompute grid
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    grid = (total_num_q_blocks, num_kv_heads)

    kwargs = dict(
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=None,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        scale=softmax_scale,
        k_scale=1.0,
        v_scale=1.0,
        out_scale=1.0 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=sample["USE_ALIBI_SLOPES"],
        USE_ALIBI_SQRT=sample["USE_ALIBI_SQRT"],
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=sample["USE_SOFTCAP"],
        USE_SINKS=sample["USE_SINKS"],
        USE_MM_PREFIX=sample["USE_MM_PREFIX"],
        MAX_MM_RANGES=sample["MAX_MM_RANGES"],
        mm_prefix_range_ptr=None,
        SLIDING_WINDOW=sample["sliding_window"],
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        USE_FP8=sample["USE_FP8"],
        FP8_MIN=float8_info.min,
        FP8_MAX=float8_info.max,
        softmax_threshold=softmax_threshold_val,
        USE_SPARSE=sample["USE_SPARSE"],
    )

    # Launch params
    kwargs["num_warps"] = config["num_warps"]
    kwargs["num_stages"] = config["num_stages"]

    # HCU compiler extra_kargs
    for knob in EXTRA_KARG_KNOBS:
        val = config.get(knob)
        if val is not None and val != _EXTRA_KARG_DEFAULTS.get(knob):
            kwargs[knob] = val

    return grid, kwargs, out


def prepare_kernel_args_full(
    sample: Dict[str, Any], config: Dict[str, Any],
) -> Tuple[tuple, dict, "torch.Tensor"]:
    """Prepare kernel launch arguments from a full-tensor sample."""
    q = sample["q"].cuda()
    k_compact = sample["k_compact"].cuda()
    v_compact = sample["v_compact"].cuda()
    block_table = sample["block_table"].cuda()
    seqused_k = sample["seqused_k"].cuda()
    cu_seqlens_q = sample["cu_seqlens_q"].cuda()

    out = torch.empty_like(q)

    num_query_heads = sample["num_query_heads"]
    num_kv_heads = sample["num_kv_heads"]
    num_queries_per_kv = sample["num_queries_per_kv"]
    head_size = sample["head_size"]
    block_size = sample["block_size"]
    num_seqs = sample["num_seqs"]
    softmax_scale = sample["softmax_scale"]
    softcap = sample["softcap"]
    output_scale = sample["output_scale"]
    softmax_threshold_val = sample["softmax_threshold_val"]
    use_qq_bias = sample["USE_QQ_BIAS"]

    # Tiling from config
    TILE_SIZE = config["TILE_SIZE"]
    BLOCK_M = config["BLOCK_M"]
    BLOCK_Q = max(BLOCK_M // num_queries_per_kv, 1)

    # Recompute grid
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    grid = (total_num_q_blocks, num_kv_heads)

    # Optional tensors
    sinks = sample.get("sinks")
    if sinks is not None:
        sinks = sinks.cuda()
    alibi_slopes = sample.get("alibi_slopes")
    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes.cuda()
    qq_bias = sample.get("qq_bias")
    if qq_bias is not None:
        qq_bias = qq_bias.cuda()
    mm_prefix_range = sample.get("mm_prefix_range")
    if mm_prefix_range is not None:
        mm_prefix_range = mm_prefix_range.cuda()

    k_descale = sample["k_descale"]
    if isinstance(k_descale, torch.Tensor):
        k_descale = k_descale.cuda()
    v_descale = sample["v_descale"]
    if isinstance(v_descale, torch.Tensor):
        v_descale = v_descale.cuda()

    kwargs = dict(
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_compact,
        value_cache_ptr=v_compact,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1.0 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=sample["USE_ALIBI_SLOPES"],
        USE_ALIBI_SQRT=sample["USE_ALIBI_SQRT"],
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=sample["USE_SOFTCAP"],
        USE_SINKS=sample["USE_SINKS"],
        USE_MM_PREFIX=sample["USE_MM_PREFIX"],
        MAX_MM_RANGES=sample["MAX_MM_RANGES"],
        mm_prefix_range_ptr=mm_prefix_range,
        SLIDING_WINDOW=sample["sliding_window"],
        stride_k_cache_0=k_compact.stride(0),
        stride_k_cache_1=k_compact.stride(1),
        stride_k_cache_2=k_compact.stride(2),
        stride_k_cache_3=k_compact.stride(3),
        stride_v_cache_0=v_compact.stride(0),
        stride_v_cache_1=v_compact.stride(1),
        stride_v_cache_2=v_compact.stride(2),
        stride_v_cache_3=v_compact.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        USE_FP8=sample["USE_FP8"],
        FP8_MIN=float8_info.min,
        FP8_MAX=float8_info.max,
        softmax_threshold=softmax_threshold_val,
        USE_SPARSE=sample["USE_SPARSE"],
    )

    # Launch params
    kwargs["num_warps"] = config["num_warps"]
    kwargs["num_stages"] = config["num_stages"]

    # HCU compiler extra_kargs (skip no-op defaults to avoid needless recompilation)
    for knob in EXTRA_KARG_KNOBS:
        val = config.get(knob)
        if val is not None and val != _EXTRA_KARG_DEFAULTS.get(knob):
            kwargs[knob] = val

    return grid, kwargs, out


def prepare_kernel_args(
    sample: Dict[str, Any], config: Dict[str, Any],
) -> Tuple[tuple, dict, "torch.Tensor"]:
    """Dispatch to lite or full prepare based on sample format."""
    if sample.get("_lite", False):
        return prepare_kernel_args_lite(sample, config)
    return prepare_kernel_args_full(sample, config)


def benchmark_config(
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    warmup_iters: int,
    timed_iters: int,
    kernel_timeout_us: float = KERNEL_TIMEOUT_US,
) -> Tuple[float, str]:
    """Benchmark one config across all samples. Returns (geo_mean_us, status).

    kernel_timeout_us: if any single kernel invocation exceeds this threshold,
    the config is considered timed out.
    """
    log_sum = 0.0
    n_ok = 0

    for sample in samples:
        try:
            grid, kwargs, out = prepare_kernel_args(sample, config)

            # Warmup (includes Triton JIT compilation)
            for _ in range(warmup_iters):
                kernel_unified_attention_2d[grid](**kwargs)
            torch.cuda.synchronize()

            # Timed runs
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(timed_iters):
                kernel_unified_attention_2d[grid](**kwargs)
            end.record()
            torch.cuda.synchronize()

            elapsed_ms = start.elapsed_time(end)
            avg_us = (elapsed_ms / timed_iters) * 1000.0

            # Per-invocation timeout check
            if avg_us > kernel_timeout_us:
                return float("nan"), f"timeout: avg {avg_us:.0f}us > {kernel_timeout_us:.0f}us limit"

            log_sum += math.log(avg_us)
            n_ok += 1
        except Exception as e:
            return float("nan"), f"error: {e}"

    if n_ok == 0:
        return float("nan"), "error: no samples succeeded"
    geo_mean_us = math.exp(log_sum / n_ok)
    return geo_mean_us, "ok"


def generate_configs_independent() -> List[Dict[str, Any]]:
    """Sweep each knob independently (default mode).
    Only sweeps tiling/launch params, excludes HCU compiler knobs."""
    configs = [dict(DEFAULTS)]  # baseline
    for knob, values in SEARCH_SPACE.items():
        if knob in EXTRA_KARG_KNOBS:
            continue
        for val in values:
            if val == DEFAULTS[knob]:
                continue
            cfg = dict(DEFAULTS)
            cfg[knob] = val
            configs.append(cfg)
    return configs


def generate_configs_grid(sweeps: Dict[str, List]) -> List[Dict[str, Any]]:
    """Full grid over specified knobs."""
    knob_names = list(sweeps.keys())
    value_lists = list(sweeps.values())
    configs = []
    for combo in itertools.product(*value_lists):
        cfg = dict(DEFAULTS)
        for name, val in zip(knob_names, combo):
            cfg[name] = val
        configs.append(cfg)
    return configs


def generate_configs_from_sweeps(
    sweeps: Dict[str, List], full_grid: bool,
) -> List[Dict[str, Any]]:
    if full_grid:
        return generate_configs_grid(sweeps)
    configs = [dict(DEFAULTS)]
    for knob, values in sweeps.items():
        for val in values:
            if val == DEFAULTS.get(knob):
                continue
            cfg = dict(DEFAULTS)
            cfg[knob] = val
            configs.append(cfg)
    return configs


def parse_sweep_arg(sweep_str: str) -> Tuple[str, List]:
    """Parse 'knob=v1,v2,v3' into (knob_name, [v1, v2, v3])."""
    name, vals_str = sweep_str.split("=", 1)
    name = name.strip()
    raw_vals = [v.strip() for v in vals_str.split(",")]

    if name in SEARCH_SPACE:
        example = SEARCH_SPACE[name][0]
        if isinstance(example, int):
            return name, [int(v) for v in raw_vals]
        elif isinstance(example, float):
            return name, [float(v) for v in raw_vals]
        return name, raw_vals

    try:
        return name, [int(v) for v in raw_vals]
    except ValueError:
        return name, raw_vals


def config_desc(config: Dict[str, Any]) -> str:
    diffs = [f"{k}={config[k]}" for k in DEFAULTS if config[k] != DEFAULTS[k]]
    return ", ".join(diffs) if diffs else "(baseline)"


# ═══════════════════════════════════════════════════════════════════
# Multi-GPU worker
# ═══════════════════════════════════════════════════════════════════

def _gpu_worker(
    gpu_id: int,
    config_indices: List[int],
    configs: List[Dict[str, Any]],
    data_dir: str,
    max_samples: int,
    filter_bs: int,
    filter_bs_range: str,
    filter_max_seqlen_range: str,
    sample_n: int,
    warmup_iters: int,
    timed_iters: int,
    kernel_timeout_us: float,
    result_queue: mp.Queue,
):
    """Worker process: set device, load data, benchmark assigned configs."""
    # Each worker only sees one GPU
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Heavy imports after setting device visibility
    import torch as _torch
    _torch.cuda.set_device(0)  # device 0 within this process's view

    global torch, triton, kernel_unified_attention_2d, float8_info
    torch = _torch

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.ops.triton_unified_attention import (
        kernel_unified_attention_2d as _kernel,
    )
    from vllm.platforms import current_platform
    from vllm.triton_utils import triton as _triton

    kernel_unified_attention_2d = _kernel
    triton = _triton
    float8_info = _torch.finfo(current_platform.fp8_dtype())

    samples = load_samples(data_dir, max_samples)
    samples = filter_samples(samples, filter_bs, filter_bs_range, filter_max_seqlen_range)

    # Random subsampling
    if sample_n is not None and len(samples) > sample_n:
        random.seed(42)  # deterministic across GPUs
        samples = random.sample(samples, sample_n)

    if not samples:
        print(f"[GPU {gpu_id}] No samples after filtering, skipping")
        for ci in config_indices:
            result_queue.put((ci, configs[ci], float("nan"), "no samples after filtering"))
        return

    print(f"[GPU {gpu_id}] Loaded {len(samples)} samples (after filtering), "
          f"{len(config_indices)} configs to run")

    for ci in config_indices:
        config = configs[ci]
        desc = config_desc(config)
        geo_mean_us, status = benchmark_config(samples, config, warmup_iters, timed_iters, kernel_timeout_us)
        if status == "ok":
            print(f"[GPU {gpu_id}] [{ci+1}/{len(configs)}] {desc} ... {geo_mean_us:.1f} us (geo_mean)")
        else:
            print(f"[GPU {gpu_id}] [{ci+1}/{len(configs)}] {desc} ... FAILED: {status}")
        result_queue.put((ci, config, geo_mean_us, status))


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for kernel_unified_attention_2d knobs (multi-GPU)",
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory with saved .pt samples",
    )
    parser.add_argument(
        "--sweep", action="append", default=[],
        help="Knob sweep: 'knob=v1,v2,v3'. Repeatable.",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Full grid over swept knobs (default: independent sweep)",
    )
    parser.add_argument("--output", default="grid_search_results.csv")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=TIMED_ITERS)
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs (default: all visible GPUs)",
    )
    parser.add_argument(
        "--timeout", type=float, default=KERNEL_TIMEOUT_US,
        help=f"Per-kernel-invocation timeout in microseconds (default: {KERNEL_TIMEOUT_US})",
    )
    # ── Filtering ──
    parser.add_argument(
        "--filter-bs", type=int, default=None,
        help="Only benchmark samples with num_seqs==N",
    )
    parser.add_argument(
        "--filter-bs-range", type=str, default=None,
        help="Filter num_seqs in range, e.g. '1,16'",
    )
    parser.add_argument(
        "--filter-max-seqlen-range", type=str, default=None,
        help="Filter max_seq_len in range, e.g. '1000,5000'",
    )
    parser.add_argument(
        "--sample-n", type=int, default=None,
        help="Randomly sample N samples after filtering (for large datasets)",
    )
    args = parser.parse_args()

    # Build configs
    if args.sweep:
        sweeps = {}
        for s in args.sweep:
            name, vals = parse_sweep_arg(s)
            sweeps[name] = vals
        configs = generate_configs_from_sweeps(sweeps, args.grid)
    else:
        configs = generate_configs_independent()

    # Determine GPUs
    if args.gpus is not None:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        import torch as _torch
        gpu_ids = list(range(_torch.cuda.device_count()))

    num_gpus = len(gpu_ids)
    if num_gpus == 0:
        print("ERROR: No GPUs available.")
        sys.exit(1)

    filter_desc = []
    if args.filter_bs is not None:
        filter_desc.append(f"num_seqs=={args.filter_bs}")
    if args.filter_bs_range is not None:
        filter_desc.append(f"num_seqs in [{args.filter_bs_range}]")
    if args.filter_max_seqlen_range is not None:
        filter_desc.append(f"max_seq_len in [{args.filter_max_seqlen_range}]")
    filter_str = f"  Filters: {', '.join(filter_desc)}\n" if filter_desc else ""

    print(
        f"Benchmarking {len(configs)} configs on {num_gpus} GPU(s) {gpu_ids}\n"
        f"  ({args.warmup} warmup + {args.iters} timed iters each, "
        f"kernel timeout {args.timeout:.0f}us per invocation)\n"
        f"  Metric: geometric mean across samples\n"
        f"{filter_str}"
    )

    # Distribute configs round-robin across GPUs
    gpu_config_indices: Dict[int, List[int]] = {gid: [] for gid in gpu_ids}
    for ci in range(len(configs)):
        gid = gpu_ids[ci % num_gpus]
        gpu_config_indices[gid].append(ci)

    # Launch workers
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    workers = []
    for gid in gpu_ids:
        p = mp.Process(
            target=_gpu_worker,
            args=(
                gid,
                gpu_config_indices[gid],
                configs,
                args.data_dir,
                args.max_samples,
                args.filter_bs,
                args.filter_bs_range,
                args.filter_max_seqlen_range,
                args.sample_n,
                args.warmup,
                args.iters,
                args.timeout,
                result_queue,
            ),
        )
        p.start()
        workers.append(p)

    # Collect results
    results_by_idx = {}
    total_expected = len(configs)
    while len(results_by_idx) < total_expected:
        ci, config, geo_mean_us, status = result_queue.get()
        results_by_idx[ci] = (config, geo_mean_us, status)

    for p in workers:
        p.join()

    # Sort by original config index for deterministic output
    knob_names = list(DEFAULTS.keys())
    fieldnames = knob_names + ["geo_mean_us", "status"]
    results = []
    for ci in range(len(configs)):
        config, geo_mean_us, status = results_by_idx[ci]
        row = {k: config.get(k, DEFAULTS.get(k)) for k in knob_names}
        row["geo_mean_us"] = geo_mean_us
        row["status"] = status
        results.append(row)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {args.output}")

    # Print Top-10
    print("\n=== Top 10 configs (by geometric mean time across samples) ===")
    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: r["geo_mean_us"])
    for i, row in enumerate(ok_results[:10]):
        cfg = {k: row[k] for k in knob_names}
        desc = config_desc(cfg)
        print(f"  #{i+1}: {row['geo_mean_us']:.1f} us  —  {desc}")

    if not ok_results:
        print("  (no successful configs)")


if __name__ == "__main__":
    main()
