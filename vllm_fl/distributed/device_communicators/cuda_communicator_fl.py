# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hygon-enabled CudaCommunicator.

vLLM's ``CustomAllreduce.__init__`` guards itself with a single bare
``assert current_platform.is_cuda_alike()`` (custom_all_reduce.py:150).
Hygon DCU is a HIP/ROCm device on which the ``_C_custom_ar`` extension works,
but ``PlatformFL.is_cuda_alike()`` deliberately returns ``False`` for hygon so
that the plugin's own FlagGems/Triton kernels stay on the dispatch path.

Rather than flipping ``is_cuda_alike()`` globally (it is consulted by ~43 files
in vLLM, several at import time, which would change kernel selection on a
working deployment), this subclass makes it return ``True`` only for the
duration of ``CudaCommunicator.__init__``.

That window is safe: inside ``CudaCommunicator.__init__`` the only readers of
``is_cuda_alike()`` are ``CustomAllreduce`` (the assert above) and
``QuickAllReduce``, and the latter is never constructed on hygon because
``is_rocm()`` is ``False``. ``pynccl.py`` never reads it, while
``SymmMemCommunicator`` and ``FlashInferAllReduce`` are gated off by
``is_cuda()`` / env vars.
"""

from contextlib import contextmanager

import torch

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.logger import init_logger

from vllm_fl.platform import PlatformFL, hygon_custom_ar_enabled

logger = init_logger(__name__)

# The shipped `_C.abi3.so` only instantiates the custom_ar kernel for float32
# and float16 (verified via `nm`); there is NO bfloat16 kernel. Passing bf16
# straight through raises "custom allreduce only supports float32, float16 and
# bfloat16" on the first all-reduce. float32 is a loss-free superset of bf16, so
# these dtypes are up-converted to fp32 for the reduction and cast back after.
_UPCAST_DTYPES = (torch.bfloat16,)


@contextmanager
def _force_cuda_alike():
    """Temporarily make ``PlatformFL.is_cuda_alike()`` report ``True``."""
    original = PlatformFL.is_cuda_alike
    PlatformFL.is_cuda_alike = lambda self: True
    try:
        yield
    finally:
        PlatformFL.is_cuda_alike = original


class _Fp32CustomAllreduce:
    """Proxy around ``CustomAllreduce`` that reduces unsupported dtypes in fp32.

    Only ``should_custom_ar`` and ``custom_all_reduce`` are overridden -- both are
    the entry points ``CudaCommunicator.all_reduce`` calls. Everything else
    (``capture``, ``register_graph_buffers``, ``disabled``, ``close`` ...) is
    delegated to the wrapped object so CUDA-graph capture keeps working: the
    transient fp32 tensor is allocated from the graph pool exactly like any other
    activation and picked up by ``register_graph_buffers`` after capture.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        # Delegate anything not defined here (disabled, capture, max_size, ...).
        return getattr(self._inner, name)

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if inp.dtype not in _UPCAST_DTYPES:
            return self._inner.should_custom_ar(inp)
        # The kernel runs on an fp32 copy, so the effective payload is 4 B per
        # element. Replicate CustomAllreduce.should_custom_ar's checks against
        # that fp32 size WITHOUT allocating the fp32 tensor (which, in the
        # fallback path, would be pure wasted work on every large all-reduce).
        from vllm.distributed.device_communicators.custom_all_reduce import (
            is_weak_contiguous,
        )

        inner = self._inner
        if inner.disabled:
            return False
        fp32_size = inp.numel() * 4
        if fp32_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if inner.world_size == 2 or inner.fully_connected:
            return fp32_size < inner.max_size
        return False

    def custom_all_reduce(self, inp: torch.Tensor):
        if inp.dtype not in _UPCAST_DTYPES:
            return self._inner.custom_all_reduce(inp)
        out = self._inner.custom_all_reduce(inp.to(torch.float32))
        if out is None:
            return None
        return out.to(inp.dtype)


class CudaCommunicatorFL(CudaCommunicator):
    """CudaCommunicator that can enable custom allreduce on Hygon DCU."""

    def __init__(self, *args, **kwargs):
        if not (PlatformFL.vendor_name == "hygon" and hygon_custom_ar_enabled()):
            super().__init__(*args, **kwargs)
            return

        # `CudaCommunicator.__init__` lazily imports these. Import them *before*
        # entering the window so that no module-level `is_cuda_alike()` snapshot
        # (e.g. `vllm/kernels/vllm_c.py::CUDA_ALIKE`) can be latched to True by
        # an import that happens to be triggered inside it.
        import vllm.distributed.device_communicators.custom_all_reduce  # noqa: F401
        import vllm.distributed.device_communicators.flashinfer_all_reduce  # noqa: F401
        import vllm.distributed.device_communicators.pynccl  # noqa: F401
        import vllm.distributed.device_communicators.quick_all_reduce  # noqa: F401
        import vllm.distributed.device_communicators.symm_mem  # noqa: F401

        logger.info(
            "Enabling custom allreduce for Hygon DCU "
            "(set VLLM_FL_HYGON_CUSTOM_AR=0 to disable)."
        )
        with _force_cuda_alike():
            super().__init__(*args, **kwargs)

        ca = self.ca_comm
        if ca is None or ca.disabled:
            logger.warning(
                "Custom allreduce is NOT active on Hygon; falling back to "
                "NCCL/RCCL for tensor-parallel all-reduce."
            )
        else:
            logger.info(
                "Custom allreduce is active on Hygon (world_size=%d, max_size=%d).",
                ca.world_size,
                ca.max_size,
            )
            # The Hygon custom_ar kernel has no bf16 instantiation; wrap it so
            # bf16 all-reduces run losslessly in fp32. `should_custom_ar` /
            # `custom_all_reduce` are the only methods CudaCommunicator.all_reduce
            # invokes; the proxy delegates everything else (incl. capture()).
            self.ca_comm = _Fp32CustomAllreduce(ca)
            _log_coverage(ca.max_size)


def _log_coverage(max_size: int) -> None:
    """Report which batch sizes will actually be served by custom allreduce.

    ``should_custom_ar()`` only accepts payloads smaller than ``max_size``, so
    large prefill batches fall back to NCCL/RCCL. Logging the crossover point at
    startup makes it obvious how much of the workload the custom kernel covers.
    """
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None or config.model_config is None:
        return

    try:
        hidden_size = config.model_config.get_hidden_size()
        dtype_size = config.model_config.dtype.itemsize
    except Exception:  # diagnostics must never break startup
        logger.debug("Could not determine allreduce payload size.", exc_info=True)
        return

    # bf16 all-reduces are upcast to fp32 for the kernel, so the size checked
    # against max_size is the fp32 byte count (4 B/element), not the bf16 one.
    if dtype_size < 4 and config.model_config.dtype in _UPCAST_DTYPES:
        dtype_size = 4

    bytes_per_token = hidden_size * dtype_size
    if bytes_per_token <= 0:
        return

    max_tokens = (max_size - 1) // bytes_per_token
    max_batched = config.scheduler_config.max_num_batched_tokens

    logger.info(
        "Custom allreduce coverage: %d B/token (hidden_size=%d, %s, fp32 kernel) "
        "-> payloads up to %d tokens use the custom kernel; larger batches fall "
        "back to NCCL/RCCL.",
        bytes_per_token,
        hidden_size,
        config.model_config.dtype,
        max_tokens,
    )
    if max_batched is not None and max_batched > max_tokens:
        logger.info(
            "Custom allreduce coverage: decode is fully covered, but prefill "
            "batches above %d tokens (max_num_batched_tokens=%d) will use "
            "NCCL/RCCL.",
            max_tokens,
            max_batched,
        )
