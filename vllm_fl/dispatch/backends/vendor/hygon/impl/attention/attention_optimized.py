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


def _gather_paged_kv(key_cache, value_cache, block_table, seqused_k):
    """Gather paged KV cache into contiguous [total_k, num_kv_heads, head_size] tensors.

    Also returns cu_seqlens_k and max_seqlen_k for the gathered KV.
    """
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_size = key_cache.shape[3]
    num_seqs = seqused_k.shape[0]

    k_parts = []
    v_parts = []
    cu_seqlens_k_list = [0]
    total_k = 0
    for i in range(num_seqs):
        seq_len = int(seqused_k[i].item())
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
    max_seqlen_k = int(seqused_k.max().item()) if num_seqs > 0 else 0
    return k_gathered, v_gathered, cu_seqlens_k, max_seqlen_k


class AttentionOptimizedBackend(TritonAttentionBackend):
    """Optimized attention backend using flash_attn kernels.

    Inherits all metadata/builder/cache logic from TritonAttentionBackend,
    only overrides the impl class to use hg_flash_attn_varlen_func.
    """

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AttentionOptimizedImpl"]:
        return AttentionOptimizedImpl

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder


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
        block_table = attn_metadata.block_table
        block_size = key_cache.shape[1]

        from flash_attn import hg_flash_attn_varlen_func

        # hg_flash_attn_varlen_func paged kernels only support block_size 64 or 128.
        # Hybrid models (e.g. Qwen3.5 with mamba) force block_size=784 to match
        # mamba page size. Fall back to gathering KV into contiguous tensors.
        if block_size not in (64, 128):
            k_gathered, v_gathered, cu_seqlens_k, max_seqlen_k = \
                _gather_paged_kv(key_cache, value_cache, block_table, seqused_k)

            hg_flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=k_gathered,
                v=v_gathered,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                softcap=self.logits_soft_cap,
                out=output[:num_actual_tokens],
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
            max_seqlen_k=attn_metadata.max_seq_len,
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
