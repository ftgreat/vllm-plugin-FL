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
    patch_fla_packed_decode()


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
