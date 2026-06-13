# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon-specific monkey-patches for DCU optimization."""

import logging

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_hygon_patches():
    """Apply all Hygon-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True
    patch_ssm_state_dtype()
    patch_fla_packed_decode()
    patch_causal_conv1d_update()
    patch_chunk_delta_h()
    patch_prefill_l2norm_precision()


def patch_ssm_state_dtype():
    """SSM state dtype patch — keep original float32 to preserve numerical precision.

    Previous versions forced bfloat16 to halve HBM I/O, but the 7-bit mantissa
    causes compounding accumulation error in the recurrent state h across tokens.
    """
    logger.info("SSM state dtype: using upstream default (float32) for precision")


def patch_fla_packed_decode():
    """Patch packed_decode with num_warps=2 optimized version for Hygon DCU."""
    try:
        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_lib
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib

        from .impl.fla_packed_decode import (
            fused_recurrent_gated_delta_rule_packed_decode as hygon_packed_decode,
        )

        _fla_recurrent_lib.fused_recurrent_gated_delta_rule_packed_decode = (
            hygon_packed_decode
        )
        _gdn_lib.fused_recurrent_gated_delta_rule_packed_decode = hygon_packed_decode
        logger.info("Patched fused_recurrent_gated_delta_rule_packed_decode for Hygon DCU (num_warps=2)")
    except Exception as e:
        logger.warning("Failed to patch packed_decode for Hygon: %s", e)


def patch_causal_conv1d_update():
    """Patch causal_conv1d_update with num_warps=2, num_stages=1 for Hygon DCU."""
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib
        import vllm.model_executor.layers.mamba.short_conv as _short_conv_lib
        import vllm.model_executor.layers.mamba.mamba_mixer2 as _mamba_mixer2_lib
        import vllm.model_executor.layers.mamba.mamba_mixer as _mamba_mixer_lib

        from .impl.causal_conv1d_update import (
            causal_conv1d_update as hygon_conv1d_update,
        )

        _conv1d_lib.causal_conv1d_update = hygon_conv1d_update
        _gdn_lib.causal_conv1d_update = hygon_conv1d_update
        _short_conv_lib.causal_conv1d_update = hygon_conv1d_update
        _mamba_mixer2_lib.causal_conv1d_update = hygon_conv1d_update
        _mamba_mixer_lib.causal_conv1d_update = hygon_conv1d_update
        logger.info("Patched causal_conv1d_update for Hygon DCU (num_warps=2, num_stages=1)")
    except Exception as e:
        logger.warning("Failed to patch causal_conv1d_update for Hygon: %s", e)


def patch_chunk_delta_h():
    """Patch chunk_gated_delta_rule_fwd_kernel_h_blockdim64 with num_stages=1 for Hygon DCU."""
    try:
        import flag_gems.fused.FLA.chunk_delta_h as _chunk_h_lib

        from .impl.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64 as hygon_chunk_h,
        )

        _chunk_h_lib.chunk_gated_delta_rule_fwd_kernel_h_blockdim64 = hygon_chunk_h
        logger.info("Patched chunk_gated_delta_rule_fwd_kernel_h_blockdim64 for Hygon DCU (num_stages=1)")
    except Exception as e:
        logger.warning("Failed to patch chunk_delta_h for Hygon: %s", e)


def patch_prefill_l2norm_precision():
    """Move L2 normalization from fused_post_conv_prep into chunk_gated_delta_rule.

    Problem: The upstream fused_post_conv_prep applies L2 normalization to q/k
    in float32 but then truncates to bf16 when storing. The downstream
    chunk_gated_delta_rule receives these truncated values, losing precision.

    Fix: Skip L2norm in fused_post_conv_prep (replace with no-l2norm version),
    and force use_qk_l2norm_in_kernel=True in ChunkGatedDeltaRule.forward_native
    so that L2 normalization happens entirely in float32 inside the chunk kernel.
    """
    try:
        import vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv as _post_conv_lib
        import vllm.model_executor.layers.fla.ops as _fla_ops
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib

        from .impl.fused_post_conv_fp32 import (
            fused_post_conv_prep as no_l2norm_fused_post_conv_prep,
        )

        # 1. Replace fused_post_conv_prep with version that skips L2norm
        _post_conv_lib.fused_post_conv_prep = no_l2norm_fused_post_conv_prep
        _fla_ops.fused_post_conv_prep = no_l2norm_fused_post_conv_prep
        _gdn_lib.fused_post_conv_prep = no_l2norm_fused_post_conv_prep

        # 2. Patch ChunkGatedDeltaRule.forward_native to always use
        #    use_qk_l2norm_in_kernel=True (L2norm inside chunk kernel in fp32)
        _ChunkGDR = _gdn_lib.ChunkGatedDeltaRule
        _orig_forward_native = _ChunkGDR.forward_native

        def _patched_forward_native(self, q, k, v, g, beta, initial_state,
                                    output_final_state, cu_seqlens=None,
                                    chunk_indices=None, chunk_offsets=None,
                                    use_qk_l2norm_in_kernel=True):
            return _orig_forward_native(
                self, q, k, v, g, beta, initial_state,
                output_final_state, cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=True,
            )

        _ChunkGDR.forward_native = _patched_forward_native

        logger.info(
            "Patched prefill L2norm precision: fused_post_conv_prep skips L2norm, "
            "ChunkGatedDeltaRule.forward_native uses use_qk_l2norm_in_kernel=True"
        )
    except Exception as e:
        logger.warning("Failed to patch prefill L2norm precision: %s", e)
