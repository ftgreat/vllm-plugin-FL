#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.
# Grid search over tunable knobs for kernel_unified_attention_2d.
#
# Usage:
#   # Default: sweep each knob independently (~40 configs)
#   python tools/grid_search_attention_2d.py --data-dir /path/to/saved_samples
#
#   # Sweep specific knobs
#   python tools/grid_search_attention_2d.py --data-dir /path/to/saved_samples \
#       --sweep TILE_SIZE=16,32,64 --sweep num_warps=1,2,4,8
#
#   # Full grid over specified knobs
#   python tools/grid_search_attention_2d.py --data-dir /path/to/saved_samples \
#       --grid \
#       --sweep TILE_SIZE=32,64,128 \
#       --sweep num_warps=2,4

import argparse
import csv
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.ops.triton_unified_attention import (
    kernel_unified_attention_2d,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

float8_info = torch.finfo(current_platform.fp8_dtype())

# ═══════════════════════════════════════════════════════════════════
# Default knob values (matching production defaults)
# ═══════════════════════════════════════════════════════════════════
DEFAULTS = {
    # Tiling (constexpr — triggers recompilation)
    "TILE_SIZE": 32,
    "BLOCK_M": 16,
    # Triton launch params
    "num_warps": 2,
    "num_stages": 1,
}

# ═══════════════════════════════════════════════════════════════════
# Search space for each knob (used in independent sweep mode)
# ═══════════════════════════════════════════════════════════════════
SEARCH_SPACE = {
    "TILE_SIZE": [32, 64, 128],
    "BLOCK_M": [16, 32, 64, 128, 256],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 4],
}

WARMUP_ITERS = 3
TIMED_ITERS = 10


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
    print(f"Loaded {len(samples)} samples from {data_dir}")
    return samples


def prepare_kernel_args(
    sample: Dict[str, Any], config: Dict[str, Any],
) -> Tuple[tuple, dict, dict]:
    """Prepare kernel launch arguments from a sample + config.

    Returns (grid, kernel_kwargs, out_tensor).
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

    return grid, kwargs, out


def benchmark_config(
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    warmup_iters: int,
    timed_iters: int,
) -> Tuple[float, str]:
    """Benchmark one config across all samples. Returns (avg_us, status)."""
    total_us = 0.0
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


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for kernel_unified_attention_2d knobs",
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
    args = parser.parse_args()

    samples = load_samples(args.data_dir, args.max_samples)

    # Build configs
    if args.sweep:
        sweeps = {}
        for s in args.sweep:
            name, vals = parse_sweep_arg(s)
            sweeps[name] = vals
        configs = generate_configs_from_sweeps(sweeps, args.grid)
    else:
        configs = generate_configs_independent()

    print(
        f"Benchmarking {len(configs)} configs × {len(samples)} samples "
        f"({args.warmup} warmup + {args.iters} timed iters each)"
    )
    print()

    knob_names = list(DEFAULTS.keys())
    fieldnames = knob_names + ["avg_time_us", "status"]
    results = []

    for ci, config in enumerate(configs):
        desc = config_desc(config)
        print(
            f"[{ci+1}/{len(configs)}] {desc} ... ", end="", flush=True,
        )

        avg_us, status = benchmark_config(
            samples, config, args.warmup, args.iters,
        )

        if status == "ok":
            print(f"{avg_us:.1f} us")
        else:
            print(f"FAILED: {status}")

        row = {k: config[k] for k in knob_names}
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
