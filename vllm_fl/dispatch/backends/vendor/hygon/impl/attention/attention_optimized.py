# Copyright (c) 2025 BAAI. All rights reserved.
# Optimized attention backend: flash_attn prefill (long) + Triton (short/decode).

import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionType
from vllm.utils.torch_utils import is_quantized_kv_cache

logger = init_logger(__name__)

_3D_CONFIG_LOW = (32, 64, 4, 1)   # num_seqs <= 16
_3D_CONFIG_MID = (16, 64, 4, 1)   # num_seqs 17~56
_3D_CONFIG_HIGH = (8, 16, 4, 1)   # num_seqs 57~64

_SPARSE_THRESHOLD_ENV = os.environ.get('VLLM_SPARSE_THRESHOLD', '0')
_SPARSE_THRESHOLD = (
    float(_SPARSE_THRESHOLD_ENV)
    if float(_SPARSE_THRESHOLD_ENV) > 0
    else None
)

# Prefill sequences with max_seq_len > this threshold use flash_attn (gather KV).
# Shorter sequences use the Triton paged kernel directly (no gather overhead).
_FLASH_PREFILL_THRESHOLD = int(
    os.environ.get('VLLM_FLASH_PREFILL_THRESHOLD', '32768')
)

if _SPARSE_THRESHOLD is not None:
    logger.info("Sparse attention enabled with threshold=%s", _SPARSE_THRESHOLD)
else:
    logger.info("Sparse attention disabled")
logger.info("Flash prefill threshold: %d (use flash_attn for seq_len > %d)",
            _FLASH_PREFILL_THRESHOLD, _FLASH_PREFILL_THRESHOLD)


class AttentionOptimizedBackend(TritonAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AttentionOptimizedImpl"]:
        return AttentionOptimizedImpl

    @staticmethod
    def get_builder_cls() -> type["AttentionOptimizedMetadataBuilder"]:
        return AttentionOptimizedMetadataBuilder


def _get_3d_config(num_seqs):
    if num_seqs <= 16:
        return _3D_CONFIG_LOW
    elif num_seqs >= 57:
        return _3D_CONFIG_HIGH
    else:
        return _3D_CONFIG_MID


def _gather_kv(key_cache, value_cache, block_table, seqused_k, max_seqlen_k):
    """Gather paged KV into contiguous [total_k, nkv, hd] tensors + cu_seqlens_k.

    Only called for long prefill (never under CUDA Graph capture).
    max_seqlen_k is passed as a Python int from attn_metadata.max_seq_len,
    avoiding an extra .item() call on seqused_k.
    """
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_size = key_cache.shape[3]

    # Single GPU→CPU transfer for all seq lengths (1 sync instead of N .item() calls).
    seq_len_list = seqused_k.tolist()

    k_parts = []
    v_parts = []
    cu_seqlens_k_list = [0]
    total_k = 0
    for i, seq_len in enumerate(seq_len_list):
        seq_len = int(seq_len)
        if seq_len == 0:
            cu_seqlens_k_list.append(total_k)
            continue
        n_blk = (seq_len + block_size - 1) // block_size
        blk_ids = block_table[i, :n_blk]
        k_parts.append(key_cache[blk_ids].reshape(-1, num_kv_heads, head_size)[:seq_len])
        v_parts.append(value_cache[blk_ids].reshape(-1, num_kv_heads, head_size)[:seq_len])
        total_k += seq_len
        cu_seqlens_k_list.append(total_k)

    k_gathered = torch.cat(k_parts, dim=0) if k_parts else key_cache.new_empty(0, num_kv_heads, head_size)
    v_gathered = torch.cat(v_parts, dim=0) if v_parts else value_cache.new_empty(0, num_kv_heads, head_size)
    cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=key_cache.device)
    return k_gathered, v_gathered, cu_seqlens_k, max_seqlen_k


class AttentionOptimizedMetadataBuilder(TritonAttentionMetadataBuilder):

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        from triton import next_power_of_2
        headdim_padded = next_power_of_2(self.headdim)

        self.softmax_segm_output_32 = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, 32, headdim_padded),
            dtype=torch.float32, device=device,
        )
        self.softmax_segm_max_32 = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, 32),
            dtype=torch.float32, device=device,
        )
        self.softmax_segm_expsum_32 = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, 32),
            dtype=torch.float32, device=device,
        )
        self.softmax_segm_output_8 = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, 8, headdim_padded),
            dtype=torch.float32, device=device,
        )
        self.softmax_segm_max_8 = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, 8),
            dtype=torch.float32, device=device,
        )
        self.softmax_segm_expsum_8 = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, 8),
            dtype=torch.float32, device=device,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build,
        )
        num_seqs = common_attn_metadata.num_reqs
        seg, _, _, _ = _get_3d_config(num_seqs)
        if seg == 32:
            attn_metadata.num_par_softmax_segments = 32
            attn_metadata.softmax_segm_output = self.softmax_segm_output_32
            attn_metadata.softmax_segm_max = self.softmax_segm_max_32
            attn_metadata.softmax_segm_expsum = self.softmax_segm_expsum_32
        elif seg == 8:
            attn_metadata.num_par_softmax_segments = 8
            attn_metadata.softmax_segm_output = self.softmax_segm_output_8
            attn_metadata.softmax_segm_max = self.softmax_segm_max_8
            attn_metadata.softmax_segm_expsum = self.softmax_segm_expsum_8
        return attn_metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata


class AttentionOptimizedImpl(TritonAttentionImpl):
    """Hybrid attention: flash_attn for long prefill, Triton for short prefill and decode."""

    def _triton_attention(self, query, key_cache, value_cache, output,
                          cu_seqlens_q, max_seqlen_q, seqused_k,
                          max_seqlen_k, block_table, k_descale, v_descale,
                          attn_metadata, output_scale):
        from .ops.triton_unified_attention import (
            unified_attention as optimized_unified_attention,
        )

        num_seqs = cu_seqlens_q.shape[0] - 1
        seg, tile_3d, warps_3d, stages_3d = _get_3d_config(num_seqs)

        optimized_unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=attn_metadata.mm_prefix_range_tensor,
            softmax_threshold=_SPARSE_THRESHOLD,
            tile_size_3d=tile_3d,
            num_warps_3d=warps_3d,
            num_stages_3d=stages_3d,
        )

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata: TritonAttentionMetadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        if self._is_per_token_head_quant:
            return super().forward(
                layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        key_cache, value_cache = kv_cache.unbind(1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            if key_cache.dtype != self.fp8_dtype:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )
        descale_shape = (
            attn_metadata.query_start_loc.shape[0] - 1,
            key_cache.shape[2],
        )
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        # --- Long prefill: flash_attn (gather KV, non-paged varlen) ---
        if max_seqlen_q > 1 and max_seqlen_k > _FLASH_PREFILL_THRESHOLD:
            from flash_attn import flash_attn_varlen_func

            k_g, v_g, cu_seqlens_k, max_k = _gather_kv(
                key_cache, value_cache, block_table, seqused_k,
                max_seqlen_k)

            o = flash_attn_varlen_func(
                query[:num_actual_tokens],
                k_g,
                v_g,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_k,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
            )
            output[:num_actual_tokens].copy_(
                o.reshape(output[:num_actual_tokens].shape))
            return output

        # --- Short prefill & decode: Triton paged kernel (no gather) ---
        self._triton_attention(
            query[:num_actual_tokens], key_cache, value_cache,
            output[:num_actual_tokens], cu_seqlens_q, max_seqlen_q,
            seqused_k, max_seqlen_k, block_table, k_descale, v_descale,
            attn_metadata, output_scale,
        )
        return output
