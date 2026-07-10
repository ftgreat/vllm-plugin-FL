# Copyright (c) 2025 BAAI. All rights reserved.
# Recording utility for kernel_unified_attention_2d/3d inputs/outputs.
# Controlled by env vars; zero overhead when disabled.
#
# Env vars:
#   VLLM_FL_RECORD_ATTN_DIR       - Directory to save samples (empty=disabled)
#   VLLM_FL_RECORD_ATTN_LAYER_COUNT - Record every N-th layer call (default: 16)
#   VLLM_FL_RECORD_ATTN_MAX_SAMPLES - Max samples to record (default: inf)
#   VLLM_FL_RECORD_ATTN_LITE      - 1=lightweight (shapes only), 0=full tensors (default: 1)
#   VLLM_FL_RECORD_ATTN_SKIP      - Skip first N effective calls before recording (default: 0)

import math
import os

import torch
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

# ── Module-level config (read once at import) ──
_RECORD_DIR = os.environ.get("VLLM_FL_RECORD_ATTN_DIR", "").strip()
_LAYER_COUNT = int(os.environ.get("VLLM_FL_RECORD_ATTN_LAYER_COUNT", "16"))
_MAX_SAMPLES_STR = os.environ.get("VLLM_FL_RECORD_ATTN_MAX_SAMPLES", "inf")
_MAX_SAMPLES = float(_MAX_SAMPLES_STR) if _MAX_SAMPLES_STR.lower() == "inf" else int(_MAX_SAMPLES_STR)
_LITE_MODE = os.environ.get("VLLM_FL_RECORD_ATTN_LITE", "1") == "1"
_SKIP_COUNT = int(os.environ.get("VLLM_FL_RECORD_ATTN_SKIP", "0"))

RECORDING_ENABLED: bool = bool(_RECORD_DIR)

if RECORDING_ENABLED:
    logger.info(
        "Attention recording enabled: dir=%s, layer_count=%d, max_samples=%s, "
        "lite=%s, skip=%d",
        _RECORD_DIR, _LAYER_COUNT, _MAX_SAMPLES_STR, _LITE_MODE, _SKIP_COUNT,
    )

# ── Mutable state ──
_call_counter: int = 0
_effective_call_counter: int = 0
_sample_counter: int = 0


def _gather_used_blocks(
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Return sorted unique physical block indices used across all sequences."""
    num_blocks_per_seq = (seqused_k + block_size - 1) // block_size
    all_used = []
    for i in range(block_table.shape[0]):
        n = int(num_blocks_per_seq[i].item())
        if n > 0:
            all_used.append(block_table[i, :n])
    if not all_used:
        return torch.tensor([], dtype=block_table.dtype, device=block_table.device)
    return torch.cat(all_used).unique().sort().values


def _should_record_sample(seqused_k: torch.Tensor) -> bool:
    """Check counters and decide whether to record this call.

    Returns True if this call should be recorded, False otherwise.
    Manages _call_counter, _effective_call_counter, _sample_counter internally.
    """
    global _call_counter, _effective_call_counter, _sample_counter

    if _sample_counter >= _MAX_SAMPLES:
        return False

    # Skip CUDA Graph warmup dummy data (all seqused_k == 1)
    if seqused_k.max().item() <= 1:
        return False

    should_record = (_call_counter % _LAYER_COUNT == 0)
    _call_counter += 1

    if not should_record:
        return False

    # Skip first N effective calls (warmup/CUDA Graph capture phase)
    _effective_call_counter += 1
    if _effective_call_counter <= _SKIP_COUNT:
        return False

    return True


def _save_sample(payload: dict, kernel_tag: str) -> None:
    """Save a payload dict and manage sample counter."""
    global _sample_counter

    _sample_counter += 1
    sample_idx = _sample_counter

    save_dir = Path(_RECORD_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    filepath = save_dir / f"attn_sample_{sample_idx:04d}.pt"
    torch.save(payload, filepath)
    logger.info("Recorded %s attention sample %d → %s (lite=%s)",
                kernel_tag, sample_idx, filepath, _LITE_MODE)

    if _sample_counter >= _MAX_SAMPLES:
        logger.info(
            "Reached max attention samples (%s). Recording stopped.", _MAX_SAMPLES,
        )


def maybe_record(
    q, k, v, out, block_table, seqused_k, cu_seqlens_q,
    sinks, alibi_slopes, qq_bias, mm_prefix_range,
    k_descale, v_descale,
    softmax_scale, softcap, output_scale, softmax_threshold_val,
    num_query_heads, num_kv_heads, num_queries_per_kv, head_size, block_size,
    BLOCK_M, BLOCK_Q, TILE_SIZE, use_sparse, sliding_window,
    USE_ALIBI_SLOPES, USE_ALIBI_SQRT, USE_QQ_BIAS, USE_SOFTCAP,
    USE_SINKS, USE_MM_PREFIX, MAX_MM_RANGES, USE_FP8, USE_SPARSE,
    num_seqs,
):
    """Record one 2D kernel sample if it's this layer's turn and cap not reached."""
    if not _should_record_sample(seqused_k):
        return

    if _LITE_MODE:
        payload = _build_lite_payload(
            q, k, block_table, seqused_k, cu_seqlens_q,
            softmax_scale, softcap, output_scale, softmax_threshold_val,
            num_query_heads, num_kv_heads, num_queries_per_kv, head_size, block_size,
            BLOCK_M, BLOCK_Q, TILE_SIZE, use_sparse, sliding_window,
            USE_ALIBI_SLOPES, USE_ALIBI_SQRT, USE_QQ_BIAS, USE_SOFTCAP,
            USE_SINKS, USE_MM_PREFIX, MAX_MM_RANGES, USE_FP8, USE_SPARSE,
            num_seqs,
        )
    else:
        payload = _build_full_payload(
            q, k, v, out, block_table, seqused_k, cu_seqlens_q,
            sinks, alibi_slopes, qq_bias, mm_prefix_range,
            k_descale, v_descale,
            softmax_scale, softcap, output_scale, softmax_threshold_val,
            num_query_heads, num_kv_heads, num_queries_per_kv, head_size, block_size,
            BLOCK_M, BLOCK_Q, TILE_SIZE, use_sparse, sliding_window,
            USE_ALIBI_SLOPES, USE_ALIBI_SQRT, USE_QQ_BIAS, USE_SOFTCAP,
            USE_SINKS, USE_MM_PREFIX, MAX_MM_RANGES, USE_FP8, USE_SPARSE,
            num_seqs,
        )

    payload["_kernel"] = "2d"
    _save_sample(payload, "2d")


def _build_lite_payload(
    q, k, block_table, seqused_k, cu_seqlens_q,
    softmax_scale, softcap, output_scale, softmax_threshold_val,
    num_query_heads, num_kv_heads, num_queries_per_kv, head_size, block_size,
    BLOCK_M, BLOCK_Q, TILE_SIZE, use_sparse, sliding_window,
    USE_ALIBI_SLOPES, USE_ALIBI_SQRT, USE_QQ_BIAS, USE_SOFTCAP,
    USE_SINKS, USE_MM_PREFIX, MAX_MM_RANGES, USE_FP8, USE_SPARSE,
    num_seqs,
):
    """Build lightweight payload: shapes + metadata only (~KB per sample)."""
    seqused_k_cpu = seqused_k.cpu()
    cu_seqlens_q_cpu = cu_seqlens_q.cpu()

    # Compute grouping fields
    query_lens = cu_seqlens_q_cpu[1:] - cu_seqlens_q_cpu[:-1]
    max_query_len = int(query_lens.max().item()) if query_lens.numel() > 0 else 0
    max_seq_len = int(seqused_k_cpu.max().item())
    total_q_tokens = int(q.shape[0])

    return {
        # Lite mode marker
        "_lite": True,

        # Shapes for synthetic tensor construction
        "q_shape": tuple(q.shape),
        "q_dtype": str(q.dtype),
        "q_stride": tuple(q.stride()),
        "k_shape": tuple(k.shape),
        "k_dtype": str(k.dtype),
        "k_stride": tuple(k.stride()),
        "block_table_shape": tuple(block_table.shape),
        "block_table_stride_0": block_table.stride(0),

        # Small tensors (preserved for exact access pattern replay)
        "seqused_k": seqused_k_cpu,
        "cu_seqlens_q": cu_seqlens_q_cpu,

        # Scalar parameters
        "softmax_scale": softmax_scale,
        "softcap": softcap,
        "output_scale": output_scale,
        "softmax_threshold_val": softmax_threshold_val,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "num_queries_per_kv": num_queries_per_kv,
        "head_size": head_size,
        "block_size": block_size,
        "BLOCK_M": BLOCK_M,
        "BLOCK_Q": BLOCK_Q,
        "TILE_SIZE": TILE_SIZE,
        "use_sparse": use_sparse,
        "sliding_window": sliding_window,
        "num_seqs": num_seqs,

        # Boolean flags
        "USE_ALIBI_SLOPES": USE_ALIBI_SLOPES,
        "USE_ALIBI_SQRT": USE_ALIBI_SQRT,
        "USE_QQ_BIAS": USE_QQ_BIAS,
        "USE_SOFTCAP": USE_SOFTCAP,
        "USE_SINKS": USE_SINKS,
        "USE_MM_PREFIX": USE_MM_PREFIX,
        "MAX_MM_RANGES": MAX_MM_RANGES,
        "USE_FP8": USE_FP8,
        "USE_SPARSE": USE_SPARSE,

        # Grouping fields (for filtering in grid search)
        "max_query_len": max_query_len,
        "max_seq_len": max_seq_len,
        "total_q_tokens": total_q_tokens,
    }


def _build_full_payload(
    q, k, v, out, block_table, seqused_k, cu_seqlens_q,
    sinks, alibi_slopes, qq_bias, mm_prefix_range,
    k_descale, v_descale,
    softmax_scale, softcap, output_scale, softmax_threshold_val,
    num_query_heads, num_kv_heads, num_queries_per_kv, head_size, block_size,
    BLOCK_M, BLOCK_Q, TILE_SIZE, use_sparse, sliding_window,
    USE_ALIBI_SLOPES, USE_ALIBI_SQRT, USE_QQ_BIAS, USE_SOFTCAP,
    USE_SINKS, USE_MM_PREFIX, MAX_MM_RANGES, USE_FP8, USE_SPARSE,
    num_seqs,
):
    """Build full payload with all tensors (legacy mode, ~75-300MB per sample)."""
    # Compact KV cache
    used_block_ids = _gather_used_blocks(block_table, seqused_k, block_size)
    k_compact = k[used_block_ids].cpu()
    v_compact = v[used_block_ids].cpu()

    # Remap block_table indices
    remap = torch.full(
        (k.shape[0],), -1, dtype=torch.int32, device=block_table.device,
    )
    remap[used_block_ids] = torch.arange(
        len(used_block_ids), dtype=torch.int32, device=block_table.device,
    )
    block_table_remapped = remap[block_table.long()].cpu()

    payload = {
        "q": q.cpu(),
        "k_compact": k_compact,
        "v_compact": v_compact,
        "block_table": block_table_remapped,
        "seqused_k": seqused_k.cpu(),
        "cu_seqlens_q": cu_seqlens_q.cpu(),
        "out": out.cpu(),
        # Scalars
        "softmax_scale": softmax_scale,
        "softcap": softcap,
        "output_scale": output_scale,
        "softmax_threshold_val": softmax_threshold_val,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "num_queries_per_kv": num_queries_per_kv,
        "head_size": head_size,
        "block_size": block_size,
        "BLOCK_M": BLOCK_M,
        "BLOCK_Q": BLOCK_Q,
        "TILE_SIZE": TILE_SIZE,
        "use_sparse": use_sparse,
        "sliding_window": sliding_window,
        "num_seqs": num_seqs,
        # Boolean flags
        "USE_ALIBI_SLOPES": USE_ALIBI_SLOPES,
        "USE_ALIBI_SQRT": USE_ALIBI_SQRT,
        "USE_QQ_BIAS": USE_QQ_BIAS,
        "USE_SOFTCAP": USE_SOFTCAP,
        "USE_SINKS": USE_SINKS,
        "USE_MM_PREFIX": USE_MM_PREFIX,
        "MAX_MM_RANGES": MAX_MM_RANGES,
        "USE_FP8": USE_FP8,
        "USE_SPARSE": USE_SPARSE,
        # Grouping fields
        "max_query_len": int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()),
        "max_seq_len": int(seqused_k.max().item()),
        "total_q_tokens": int(q.shape[0]),
    }

    # Optional tensors
    for name, tensor in [
        ("sinks", sinks),
        ("alibi_slopes", alibi_slopes),
        ("qq_bias", qq_bias),
        ("mm_prefix_range", mm_prefix_range),
    ]:
        if tensor is not None:
            payload[name] = tensor.cpu()

    # k_descale / v_descale may be scalar or tensor
    payload["k_descale"] = (
        k_descale.cpu() if isinstance(k_descale, torch.Tensor) else k_descale
    )
    payload["v_descale"] = (
        v_descale.cpu() if isinstance(v_descale, torch.Tensor) else v_descale
    )

    return payload


def maybe_record_3d(
    q, k, v, out, block_table, seqused_k, cu_seqlens_q,
    sinks, alibi_slopes, qq_bias, mm_prefix_range,
    k_descale, v_descale,
    softmax_scale, softcap, output_scale, softmax_threshold_val,
    num_query_heads, num_kv_heads, num_queries_per_kv, head_size, block_size,
    BLOCK_M, BLOCK_Q, TILE_SIZE, use_sparse, sliding_window,
    USE_ALIBI_SLOPES, USE_ALIBI_SQRT, USE_QQ_BIAS, USE_SOFTCAP,
    USE_SINKS, USE_MM_PREFIX, MAX_MM_RANGES, USE_SPARSE,
    num_seqs, NUM_SEGMENTS_PER_SEQ,
):
    """Record one 3D kernel sample if it's this layer's turn and cap not reached."""
    if not _should_record_sample(seqused_k):
        return

    if _LITE_MODE:
        payload = _build_lite_payload(
            q, k, block_table, seqused_k, cu_seqlens_q,
            softmax_scale, softcap, output_scale, softmax_threshold_val,
            num_query_heads, num_kv_heads, num_queries_per_kv, head_size, block_size,
            BLOCK_M, BLOCK_Q, TILE_SIZE, use_sparse, sliding_window,
            USE_ALIBI_SLOPES, USE_ALIBI_SQRT, USE_QQ_BIAS, USE_SOFTCAP,
            USE_SINKS, USE_MM_PREFIX, MAX_MM_RANGES, False, USE_SPARSE,
            num_seqs,
        )
    else:
        payload = _build_full_payload(
            q, k, v, out, block_table, seqused_k, cu_seqlens_q,
            sinks, alibi_slopes, qq_bias, mm_prefix_range,
            k_descale, v_descale,
            softmax_scale, softcap, output_scale, softmax_threshold_val,
            num_query_heads, num_kv_heads, num_queries_per_kv, head_size, block_size,
            BLOCK_M, BLOCK_Q, TILE_SIZE, use_sparse, sliding_window,
            USE_ALIBI_SLOPES, USE_ALIBI_SQRT, USE_QQ_BIAS, USE_SOFTCAP,
            USE_SINKS, USE_MM_PREFIX, MAX_MM_RANGES, False, USE_SPARSE,
            num_seqs,
        )

    payload["_kernel"] = "3d"
    payload["NUM_SEGMENTS_PER_SEQ"] = NUM_SEGMENTS_PER_SEQ
    _save_sample(payload, "3d")
