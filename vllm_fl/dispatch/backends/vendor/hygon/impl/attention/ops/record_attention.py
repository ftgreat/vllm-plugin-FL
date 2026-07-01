# Copyright (c) 2025 BAAI. All rights reserved.
# Recording utility for kernel_unified_attention_2d inputs/outputs.
# Controlled by env vars; zero overhead when disabled.

import os

import torch
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

# ── Module-level config (read once at import) ──
_RECORD_DIR = os.environ.get("VLLM_FL_RECORD_ATTN_DIR", "").strip()
_LAYER_COUNT = int(os.environ.get("VLLM_FL_RECORD_ATTN_LAYER_COUNT", "16"))
_MAX_SAMPLES = int(os.environ.get("VLLM_FL_RECORD_ATTN_MAX_SAMPLES", "50"))

RECORDING_ENABLED: bool = bool(_RECORD_DIR)

if RECORDING_ENABLED:
    logger.info(
        "Attention recording enabled: dir=%s, layer_count=%d, max_samples=%d",
        _RECORD_DIR, _LAYER_COUNT, _MAX_SAMPLES,
    )

# ── Mutable state ──
_call_counter: int = 0
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
    """Record one sample if it's this layer's turn and cap not reached."""
    global _call_counter, _sample_counter

    if _sample_counter >= _MAX_SAMPLES:
        return

    # Skip CUDA Graph warmup dummy data (all seqused_k == 1)
    if seqused_k.max().item() <= 1:
        return

    should_record = (_call_counter % _LAYER_COUNT == 0)
    _call_counter += 1

    if not should_record:
        return

    _sample_counter += 1
    sample_idx = _sample_counter

    save_dir = Path(_RECORD_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Compact KV cache ──
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

    # ── Build payload ──
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

    filepath = save_dir / f"attn_sample_{sample_idx:04d}.pt"
    torch.save(payload, filepath)
    logger.info("Recorded attention sample %d → %s", sample_idx, filepath)

    if _sample_counter >= _MAX_SAMPLES:
        logger.info(
            "Reached max attention samples (%d). Recording stopped.", _MAX_SAMPLES,
        )
