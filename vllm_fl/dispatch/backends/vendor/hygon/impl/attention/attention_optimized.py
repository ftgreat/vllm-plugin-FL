# Copyright (c) 2025 BAAI. All rights reserved.
# Optimized attention backend using flash_attn kernels.

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

# Sparse attention threshold from environment variable.
_SPARSE_THRESHOLD_ENV = os.environ.get('VLLM_SPARSE_THRESHOLD', '0')
_SPARSE_THRESHOLD = (
    float(_SPARSE_THRESHOLD_ENV)
    if float(_SPARSE_THRESHOLD_ENV) > 0
    else None
)

if _SPARSE_THRESHOLD is not None:
    logger.info("Sparse attention enabled with threshold=%s", _SPARSE_THRESHOLD)
else:
    logger.info("Sparse attention disabled")


def _gather_paged_kv_and_attn(query, key_cache, value_cache, block_table,
                               seqused_k, max_seqlen_k, cu_seqlens_q,
                               max_seqlen_q, output, softmax_scale, causal,
                               alibi_slopes, window_size, softcap):
    """Gather paged KV into padded layout and run flash_attn with seqused_k.

    Uses padded cu_seqlens_k (stride=max_seqlen_k) + real seqused_k so the
    C kernel only attends to valid tokens. Pure GPU ops, CUDA-graph safe.
    """
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_size = key_cache.shape[3]
    num_seqs = seqused_k.shape[0]
    max_blocks_per_seq = block_table.shape[1]

    token_offsets = torch.arange(max_seqlen_k, device=key_cache.device)
    block_ids_per_token = token_offsets // block_size
    offset_in_block = token_offsets % block_size

    block_ids_clamped = block_ids_per_token.unsqueeze(0).expand(num_seqs, -1) \
        .clamp(max=max_blocks_per_seq - 1)

    physical_blocks = block_table.gather(1, block_ids_clamped)
    flat_indices = physical_blocks * block_size + offset_in_block.unsqueeze(0)

    seq_mask = token_offsets.unsqueeze(0) < seqused_k.unsqueeze(1)
    flat_indices = flat_indices.where(seq_mask, torch.zeros_like(flat_indices))

    flat_1d = flat_indices.reshape(-1)
    kv_total = key_cache.shape[0] * block_size
    flat_1d = flat_1d.clamp(max=kv_total - 1)

    k_flat = key_cache.reshape(-1, num_kv_heads, head_size)[flat_1d]
    v_flat = value_cache.reshape(-1, num_kv_heads, head_size)[flat_1d]

    cu_seqlens_k = torch.arange(
        0, (num_seqs + 1) * max_seqlen_k, max_seqlen_k,
        dtype=torch.int32, device=key_cache.device,
    )

    from flash_attn.flash_attn_interface import (
        _wrapped_flash_attn_varlen_forward,
    )

    out, _, _, _ = _wrapped_flash_attn_varlen_forward(
        query,
        k_flat,
        v_flat,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,       # actual per-seq lengths within padded slots
        None,             # leftpad_k
        None,             # block_table (KV already gathered)
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        0.0,              # dropout
        softmax_scale,
        False,            # zero_tensors
        causal,
        window_size[0],
        window_size[1],
        softcap,
        False,            # return_softmax
    )
    return out


class AttentionOptimizedMetadataBuilder(TritonAttentionMetadataBuilder):

    def build_for_cudagraph_capture(self, common_attn_metadata):
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        attn_metadata.max_seq_len = 1
        return attn_metadata


class AttentionOptimizedBackend(TritonAttentionBackend):
    """Optimized attention backend using flash_attn kernels."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AttentionOptimizedImpl"]:
        return AttentionOptimizedImpl

    @staticmethod
    def get_builder_cls() -> type["AttentionOptimizedMetadataBuilder"]:
        return AttentionOptimizedMetadataBuilder


class AttentionOptimizedImpl(TritonAttentionImpl):
    """Impl that uses hg_flash_attn_varlen_func for both prefill and decode."""

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

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        block_size = key_cache.shape[1]

        from flash_attn import hg_flash_attn_varlen_func

        # hg_flash_attn_varlen_func paged kernels only support block_size 64 or 128.
        # Hybrid models (e.g. Qwen3.5 with mamba) force block_size=784 to match
        # mamba page size. Gather KV and call varlen_fwd directly with seqused_k.
        if block_size not in (64, 128):
            _gather_paged_kv_and_attn(
                query[:num_actual_tokens], key_cache, value_cache,
                block_table, seqused_k, max_seqlen_k,
                cu_seqlens_q, max_seqlen_q, output[:num_actual_tokens],
                self.scale, True, self.alibi_slopes,
                self.sliding_window, self.logits_soft_cap,
            )
            return output

        # block_size 64 or 128: use native paged attention path
        descale_shape = (
            cu_seqlens_q.shape[0] - 1,
            key_cache.shape[2],
        )
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        hg_flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            seqused_k=seqused_k,
            out=output[:num_actual_tokens],
            k_descale=k_descale,
            v_descale=v_descale,
            s_aux=self.sinks,
        )
        return output
