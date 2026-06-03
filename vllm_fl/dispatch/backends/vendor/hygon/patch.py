# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon-specific monkey-patches for DCU optimization."""

import logging
import torch

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


def patch_ssm_state_dtype():
    """Override GDN SSM state dtype from fp32 to bf16 to halve HBM I/O for packed_decode."""
    try:
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

        _original_gdn_state_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype

        @classmethod
        def _bf16_gated_delta_net_state_dtype(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype="auto"):
            conv_state_dtype, _ = _original_gdn_state_dtype.__func__(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype)
            return (conv_state_dtype, torch.bfloat16)

        MambaStateDtypeCalculator.gated_delta_net_state_dtype = _bf16_gated_delta_net_state_dtype
        logger.info("Patched gated_delta_net_state_dtype: SSM temporal state forced to bfloat16")
    except Exception as e:
        logger.warning("Failed to patch SSM state dtype for Hygon: %s", e)


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
