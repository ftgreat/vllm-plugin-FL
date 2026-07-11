# Copyright (c) 2025 BAAI. All rights reserved.
# Optimized attention backend using V1-style split 2D/3D Triton kernels.

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

# --- Dynamic 3D segment configuration ---
# Optimal (seg, tile_size, num_warps, num_stages) per num_seqs range,
# determined by grid search on recorded attention samples.
_3D_CONFIG_LOW = (32, 64, 4, 1)   # num_seqs <= 16
_3D_CONFIG_MID = (16, 64, 4, 1)   # num_seqs 17~56
_3D_CONFIG_HIGH = (8, 16, 4, 1)   # num_seqs 57~64

# Sparse attention threshold from environment variable.
# Set VLLM_SPARSE_THRESHOLD to a positive float (e.g. 0.001) to enable.
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

_USE_FLASH_PREFILL_ONE_SEQ = os.environ.get('VLLM_FL_USE_FLASH_PREFILL_ONE_SEQ', '0') == '1'
if _USE_FLASH_PREFILL_ONE_SEQ:
    logger.info("Flash prefill (single-seq) enabled via VLLM_FL_USE_FLASH_PREFILL_ONE_SEQ=1")


class AttentionOptimizedBackend(TritonAttentionBackend):
    """Optimized attention backend using V1-style split 2D/3D kernels.

    Inherits all metadata/builder/cache logic from TritonAttentionBackend,
    only overrides the impl class to use optimized kernels.
    """

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
    """Return (seg, tile_size, num_warps, num_stages) for given num_seqs."""
    if num_seqs <= 16:
        return _3D_CONFIG_LOW
    elif num_seqs >= 57:
        return _3D_CONFIG_HIGH
    else:
        return _3D_CONFIG_MID


class AttentionOptimizedMetadataBuilder(TritonAttentionMetadataBuilder):
    """Override builder to allocate dual segment buffers and select config
    based on num_seqs at build time."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        from triton import next_power_of_2
        headdim_padded = next_power_of_2(self.headdim)

        # Parent already allocated SEG=16 buffers
        # (self.softmax_segm_output / max / expsum)

        # SEG=32 buffers (for num_seqs <= 16)
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

        # SEG=8 buffers (for num_seqs 57~64)
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

        logger.info(
            "Dynamic 3D segments enabled: SEG=32 (num_seqs<=16), "
            "SEG=16 (17~56), SEG=8 (57~64). "
            "Buffers allocated for seq_threshold_3D=%d, num_heads_q=%d",
            self.seq_threshold_3D, self.num_heads_q,
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
        # else: 17~56 → keep parent's SEG=16 buffers

        return attn_metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata


class AttentionOptimizedImpl(TritonAttentionImpl):
    """Impl that uses optimized unified_attention with split 2D/3D kernels."""

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

        # Per-token-head quantized KV cache -> fallback to parent impl
        if self._is_per_token_head_quant:
            return super().forward(
                layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # Standard path: FP8 per-tensor / auto
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

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        # gfx936: the Hygon-built flash_attn 2.8.3 (FlashAttention-2) runs the
        # prefill full-attention ~2.7-2.9x faster than this kernel's Triton 2D path
        # (which only reaches ~18% of bf16 peak; flash_attn reaches ~45-51%). It is
        # exact attention (lossless, no token shift). Use it for the prefill case
        # only (max_query_len > 1); keep the tuned Triton 3D split-KV for pure decode.
        # Opt-in via VLLM_FL_USE_FLASH_PREFILL_ONE_SEQ=1. Reads the same paged KV cache via
        # block_table (k_cache/v_cache layout matches flash_attn's paged API).
        # flash_attn's paged API requires block_size==64, but the hybrid model forces
        # block_size=784 (>= mamba page). So instead gather the paged KV into a
        # contiguous [seq_len, n_kv_heads, head] tensor (cheap, ~0.06ms) and call
        # flash_attn_varlen WITHOUT block_table. Gated to a single sequence per step
        # (concurrency=1, the competition setting). num_seqs = cu_seqlens_q len - 1.
        if (
            _USE_FLASH_PREFILL_ONE_SEQ
            and max_seqlen_q > 1
            and cu_seqlens_q.shape[0] == 2  # exactly one sequence in this step
            and self.sinks is None
            and mm_prefix_range_tensor is None
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and not self.kv_cache_dtype.startswith("fp8")
            and output_scale is None
        ):
            from flash_attn import flash_attn_varlen_func

            bs = key_cache.shape[1]  # block_size
            seq_len = int(seqused_k[0].item())
            n_blk = (seq_len + bs - 1) // bs
            blk = block_table[0, :n_blk]
            k_g = key_cache[blk].reshape(-1, key_cache.shape[2], key_cache.shape[3])[:seq_len]
            v_g = value_cache[blk].reshape(-1, value_cache.shape[2], value_cache.shape[3])[:seq_len]
            cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=query.device)
            o = flash_attn_varlen_func(
                query[:num_actual_tokens],
                k_g,
                v_g,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
            )
            dst = output[:num_actual_tokens]
            dst.copy_(o.reshape(dst.shape))
            return output

        # Use optimized unified_attention with split 2D/3D kernels
        from .ops.triton_unified_attention import (
            unified_attention as optimized_unified_attention,
        )

        # Select optimal 3D kernel params based on num_seqs
        num_seqs = cu_seqlens_q.shape[0] - 1
        seg, tile_3d, warps_3d, stages_3d = _get_3d_config(num_seqs)

        # Log each distinct config once
        if not hasattr(self, '_logged_3d_config'):
            self._logged_3d_config = set()
        if seg not in self._logged_3d_config:
            self._logged_3d_config.add(seg)
            logger.info(
                "3D attention config: num_seqs=%d -> SEG=%d, TILE=%d, "
                "warps=%d, stages=%d",
                num_seqs, seg, tile_3d, warps_3d, stages_3d,
            )

        optimized_unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
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
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            softmax_threshold=_SPARSE_THRESHOLD,
            tile_size_3d=tile_3d,
            num_warps_3d=warps_3d,
            num_stages_3d=stages_3d,
        )

        return output
