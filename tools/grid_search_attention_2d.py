#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.
# Grid search over tunable knobs for kernel_unified_attention_2d.
# Supports multi-GPU: configs are evenly distributed across all visible GPUs.
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

import argparse
import csv
import itertools
import math
import multiprocessing as mp
import os
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
CONFIG_TIMEOUT = 900  # seconds (15 minutes)


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


def prepare_kernel_args(
    sample: Dict[str, Any], config: Dict[str, Any],
) -> Tuple[tuple, dict, dict]:
    """Prepare kernel launch arguments from a sample + config.

    Returns (grid, kernel_kwargs, out_tensor).
    kernel_kwargs includes all named kernel args, launch params,
    and extra_kargs.
    """
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


def benchmark_config(
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    warmup_iters: int,
    timed_iters: int,
    timeout: float = CONFIG_TIMEOUT,
) -> Tuple[float, str]:
    """Benchmark one config across all samples. Returns (avg_us, status)."""
    total_us = 0.0
    n_ok = 0
    t0 = time.monotonic()

    for sample in samples:
        if time.monotonic() - t0 > timeout:
            return float("nan"), f"timeout: exceeded {timeout:.0f}s"
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
            total_us += avg_us
            n_ok += 1
        except Exception as e:
            return float("nan"), f"error: {e}"

    if n_ok == 0:
        return float("nan"), "error: no samples succeeded"
    return total_us / n_ok, "ok"


def generate_configs_independent() -> List[Dict[str, Any]]:
    """Sweep each knob independently (default mode). ~40 configs."""
    configs = [dict(DEFAULTS)]  # baseline
    for knob, values in SEARCH_SPACE.items():
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
    warmup_iters: int,
    timed_iters: int,
    timeout: float,
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
    print(f"[GPU {gpu_id}] Loaded {len(samples)} samples, {len(config_indices)} configs to run")

    for ci in config_indices:
        config = configs[ci]
        desc = config_desc(config)
        avg_us, status = benchmark_config(samples, config, warmup_iters, timed_iters, timeout)
        if status == "ok":
            print(f"[GPU {gpu_id}] [{ci+1}/{len(configs)}] {desc} ... {avg_us:.1f} us")
        else:
            print(f"[GPU {gpu_id}] [{ci+1}/{len(configs)}] {desc} ... FAILED: {status}")
        result_queue.put((ci, config, avg_us, status))


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
        "--timeout", type=float, default=CONFIG_TIMEOUT,
        help=f"Per-config timeout in seconds (default: {CONFIG_TIMEOUT})",
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

    print(
        f"Benchmarking {len(configs)} configs on {num_gpus} GPU(s) {gpu_ids}\n"
        f"  ({args.warmup} warmup + {args.iters} timed iters each, "
        f"timeout {args.timeout:.0f}s per config)"
    )
    print()

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
        ci, config, avg_us, status = result_queue.get()
        results_by_idx[ci] = (config, avg_us, status)

    for p in workers:
        p.join()

    # Sort by original config index for deterministic output
    knob_names = list(DEFAULTS.keys())
    fieldnames = knob_names + ["avg_time_us", "status"]
    results = []
    for ci in range(len(configs)):
        config, avg_us, status = results_by_idx[ci]
        row = {k: config.get(k, DEFAULTS.get(k)) for k in knob_names}
        row["avg_time_us"] = avg_us
        row["status"] = status
        results.append(row)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {args.output}")

    # Print Top-10
    print("\n=== Top 10 configs (by avg time across samples) ===")
    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: r["avg_time_us"])
    for i, row in enumerate(ok_results[:10]):
        cfg = {k: row[k] for k in knob_names}
        desc = config_desc(cfg)
        print(f"  #{i+1}: {row['avg_time_us']:.1f} us  —  {desc}")

    if not ok_results:
        print("  (no successful configs)")


if __name__ == "__main__":
    main()
