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
    patch_fused_post_conv_fp32()


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


def patch_fused_post_conv_fp32():
    """Patch fused_post_conv_prep to output q/k in float32 when L2norm is applied.

    The upstream kernel truncates L2-normalized q/k to bf16 before passing them
    to the chunk_gated_delta_rule kernel, losing precision in the 7-bit mantissa.
    This patch keeps q/k in float32 after L2 normalization.
    """
    try:
        import vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv as _post_conv_lib
        import vllm.model_executor.layers.fla.ops as _fla_ops
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib

        from .impl.fused_post_conv_fp32 import (
            fused_post_conv_prep as fp32_fused_post_conv_prep,
        )

        _post_conv_lib.fused_post_conv_prep = fp32_fused_post_conv_prep
        _fla_ops.fused_post_conv_prep = fp32_fused_post_conv_prep
        _gdn_lib.fused_post_conv_prep = fp32_fused_post_conv_prep
        logger.info("Patched fused_post_conv_prep for float32 L2norm output (precision fix)")
    except Exception as e:
        logger.warning("Failed to patch fused_post_conv_prep: %s", e)
