# Copyright (c) 2026 BAAI. All rights reserved.

"""
Hygon backend implementation.

This backend provides optimized operator implementations for Hygon GPUs.
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


class HygonBackend(Backend):
    """
    Hygon backend for operator implementations.

    Provides optimized attention using V1-style split 2D/3D Triton kernels.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "hygon"

    @property
    def vendor(self) -> Optional[str]:
        return "hygon"

    def is_available(self) -> bool:
        """Check if Hygon hardware is available."""
        if HygonBackend._available is None:
            HygonBackend._available = torch.cuda.is_available()
        return HygonBackend._available

    # ==================== Operator Implementations ====================

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for Hygon.

        Registers optimized V1-style attention backend and returns its path.
        """
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )

        if use_mla:
            raise NotImplementedError("MLA not supported on Hygon yet.")
        # Sparse attention is supported via softmax_threshold at runtime
        # (controlled by VLLM_SPARSE_THRESHOLD env var)

        register_backend(
            backend=AttentionBackendEnum.TRITON_ATTN,
            class_path=(
                "vllm_fl.dispatch.backends.vendor.hygon.impl.attention"
                ".attention_optimized.AttentionOptimizedBackend"
            ),
            is_mamba=False,
        )
        return AttentionBackendEnum.TRITON_ATTN.get_path()
