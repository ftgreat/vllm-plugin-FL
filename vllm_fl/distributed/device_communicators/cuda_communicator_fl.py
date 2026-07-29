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
import os

import torch

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

from vllm_fl.dispatch.logger_manager import get_logger
from vllm_fl.platform import (
    PlatformFL,
    hygon_custom_ar_enabled,
    hygon_custom_ar_mode,
)

# NOTE: use the plugin's own get_logger, NOT vllm.logger.init_logger. The latter
# returns a `vllm_fl.*`-named logger, and only the `vllm` logger tree gets a
# handler/INFO level installed -- so `vllm_fl.*` inherits root's WARNING and
# every logger.info() here is silently dropped (which is exactly why none of the
# custom-allreduce startup diagnostics appeared in the exp_40/exp_41 logs).
logger = get_logger(__name__)

# The shipped `_C.abi3.so` only instantiates the custom_ar kernel for float32
# and float16 (verified via `nm`); there is NO bfloat16 kernel. Passing bf16
# straight through raises "custom allreduce only supports float32, float16 and
# bfloat16" on the first all-reduce. float32 is a loss-free superset of bf16, so
# these dtypes are up-converted to fp32 for the reduction and cast back after.
_UPCAST_DTYPES = (torch.bfloat16,)

# Standalone bf16-capable custom_ar extension (built at /workspace/hygon_custom_ar).
# Registers torch ops under the `_C_hygon_custom_ar` namespace (no collision with
# the installed `_C_custom_ar`). When loaded, we redirect vLLM's CustomAllreduce
# to it so bf16 all-reduces run on a NATIVE bf16 kernel (no fp32 upcast).
_HYGON_AR_SO_DEFAULT = "/workspace/hygon_custom_ar/build/_C_hygon_custom_ar.so"
_HYGON_AR_LOADED = None  # tri-state: None=untried, True/False=result


def _load_hygon_ar() -> bool:
    """Load the standalone bf16 custom_ar .so exactly once. Never raises."""
    global _HYGON_AR_LOADED
    if _HYGON_AR_LOADED is not None:
        return _HYGON_AR_LOADED
    _HYGON_AR_LOADED = False
    so_path = os.getenv("VLLM_FL_HYGON_AR_SO", _HYGON_AR_SO_DEFAULT)
    try:
        if not os.path.exists(so_path):
            logger.warning(
                "Hygon bf16 custom_ar .so not found at %s.",
                so_path,
            )
            return False
        torch.ops.load_library(so_path)
        size = torch.ops._C_hygon_custom_ar.meta_size()
        installed = torch.ops._C_custom_ar.meta_size()
        if size != installed:
            logger.warning(
                "Hygon bf16 custom_ar meta_size mismatch (%d != installed %d); "
                "ABI incompatible, not loading.",
                size,
                installed,
            )
            return False
        _HYGON_AR_LOADED = True
        logger.info("Loaded Hygon native bf16 custom_ar from %s.", so_path)
    except Exception:
        logger.warning(
            "Failed to load Hygon bf16 custom_ar.",
            exc_info=True,
        )
    return _HYGON_AR_LOADED


class _HygonAROps:
    """Drop-in replacement for vLLM's `_custom_ops` custom_ar functions that
    routes to the `_C_hygon_custom_ar` namespace (bf16/fp16/fp32 native).

    vLLM's ``custom_all_reduce`` module calls these as ``ops.<fn>`` at module
    scope; swapping the module's ``ops`` attribute for an instance of this class
    makes a plain ``CustomAllreduce`` drive our bf16 kernel end-to-end (init /
    buffers / all_reduce / cuda-graph registration all in one .so).
    """

    @staticmethod
    def init_custom_ar(ipc_tensors, rank_data, rank, fully_connected):
        return torch.ops._C_hygon_custom_ar.init_custom_ar(
            ipc_tensors, rank_data, rank, fully_connected
        )

    @staticmethod
    def all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes):
        torch.ops._C_hygon_custom_ar.all_reduce(
            fa, inp, out, reg_buffer, reg_buffer_sz_bytes
        )

    @staticmethod
    def dispose(fa):
        torch.ops._C_hygon_custom_ar.dispose(fa)

    @staticmethod
    def meta_size():
        return torch.ops._C_hygon_custom_ar.meta_size()

    @staticmethod
    def register_buffer(fa, ipc_tensors):
        return torch.ops._C_hygon_custom_ar.register_buffer(fa, ipc_tensors)

    @staticmethod
    def get_graph_buffer_ipc_meta(fa):
        return torch.ops._C_hygon_custom_ar.get_graph_buffer_ipc_meta(fa)

    @staticmethod
    def register_graph_buffers(fa, handles, offsets):
        torch.ops._C_hygon_custom_ar.register_graph_buffers(fa, handles, offsets)

    @staticmethod
    def allocate_shared_buffer_and_handle(size):
        return torch.ops._C_hygon_custom_ar.allocate_shared_buffer_and_handle(size)

    @staticmethod
    def open_mem_handle(mem_handle):
        return torch.ops._C_hygon_custom_ar.open_mem_handle(mem_handle)

    @staticmethod
    def free_shared_buffer(ptr):
        torch.ops._C_hygon_custom_ar.free_shared_buffer(ptr)


def _redirect_custom_ar_to_hygon() -> bool:
    """Point vLLM's custom_all_reduce module at the bf16 `_C_hygon_custom_ar`.

    Returns True if the native kernel is active. Must be called BEFORE
    `CudaCommunicator.__init__` constructs `ca_comm` so the object is built on
    our ops. Idempotent and safe: our kernel is an ABI-identical superset
    (meta_size verified equal) supporting fp32/fp16/bf16.
    """
    if not _load_hygon_ar():
        return False
    import vllm.distributed.device_communicators.custom_all_reduce as _car
    _car.ops = _HygonAROps()
    _car.custom_ar = True
    return True


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


def _ar_probe_enabled() -> bool:
    """Whether to wrap ca_comm in the in/out-of-graph call-counting probe."""
    return os.getenv("VLLM_FL_HYGON_AR_PROBE", "0") == "1"


class _ARCaptureProbe:
    """Diagnostic wrapper that counts custom-allreduce calls in/out of a graph.

    Enabled with ``VLLM_FL_HYGON_AR_PROBE=1``. Motivation: on Qwen3.6-27B TP=2
    the log shows ``Registering 0 cuda graph addresses``, and the kernel only
    records a graph buffer when it runs while ``cudaStreamIsCapturing()`` is
    Active. A 2-GPU controlled test proved neither the fp32 proxy nor
    ``_IS_CAPTURING`` propagation is at fault, which leaves one question: during
    real serving, is the all-reduce reached inside the captured decode graph at
    all, or only on the eager path?

    This wrapper answers it by tallying every ``custom_all_reduce`` call into
    three buckets and logging the totals periodically:

      ``captured``  -- called while the stream was actively capturing (the only
                       case that yields the CUDA-graph benefit)
      ``eager``     -- called outside any capture (pays cudaMemcpy, and for the
                       fp32 proxy the bf16<->fp32 conversion, for no benefit)
      ``warmup``    -- inside ``capture()`` but not yet capturing (vLLM's warmup
                       pass; ``CustomAllreduce`` returns an empty tensor here)

    A run dominated by ``eager`` with ``captured == 0`` confirms the all-reduce
    never enters the decode graph, which is what makes mode 1 a net loss.

    Purely observational: every call is delegated unchanged.
    """

    _LOG_EVERY = 2000

    def __init__(self, inner):
        self._inner = inner
        self._captured = 0
        self._eager = 0
        self._warmup = 0
        self._skipped = 0  # should_custom_ar() said no -> served by NCCL/RCCL
        self._capture_passes = 0
        self._registered_total = 0
        self._capture_seq = None  # recorded only while capturing

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @contextmanager
    def capture(self):
        """Instrument the capture lifecycle and verify cross-rank agreement.

        ``register_graph_buffers()`` runs at each ``capture()`` exit, advancing
        the kernel's ``d_rank_data_base_`` by the number of buffers recorded and
        then clearing the list (custom_all_reduce.cuh:491-515). Measured: exactly
        ONE pass registers, 11008 buffers, symmetric on both ranks -- so the
        double-registration theory is refuted, as is scale (a standalone test
        replayed 11008 registered buffers bit-exact) and duplicate addresses.

        What remains is ordering. During capture the kernel picks its rank-data
        slot as ``d_rank_data_base_ + graph_unreg_buffers_.size()``
        (custom_all_reduce.cuh:545) -- i.e. purely by call order. At replay each
        kernel reads the slot baked in at capture, and slot ``i`` is only
        meaningful if BOTH ranks recorded the same buffer at index ``i``.
        Handwritten tests can't break this because both ranks run identical
        Python loops, but in real serving the order comes from inductor-compiled
        PIECEWISE subgraphs, where any per-rank difference in compilation or
        scheduling would desynchronise the slots and corrupt peer pointers.

        So at capture exit, before the inner exit registers anything, each rank
        hashes its recorded (index, shape, dtype, numel) sequence and all-gathers
        the digests. A mismatch localises the first diverging index.
        """
        self._capture_passes += 1
        n = self._capture_passes
        self._capture_seq = []
        logger.info("[AR probe][capture] ENTER pass #%d", n)
        try:
            with self._inner.capture():
                yield
                # Read while still inside: the inner __exit__ calls
                # register_graph_buffers(), which clears the pending list.
                pending = self._pending_graph_buffers()
                self._registered_total += pending
                self._check_capture_order(n, pending)
                logger.info(
                    "[AR probe][capture] pass #%d about to register %s buffers "
                    "(running total %s). captured=%d warmup=%d eager=%d",
                    n,
                    pending,
                    self._registered_total,
                    self._captured,
                    self._warmup,
                    self._eager,
                )
        finally:
            self._capture_seq = None  # release; can be ~11k entries
            logger.info(
                "[AR probe][capture] EXIT pass #%d (registration done)", n
            )

    def _pending_graph_buffers(self):
        """Buffers recorded during this capture, not yet registered."""
        try:
            import vllm.distributed.device_communicators.custom_all_reduce as _car

            return len(_car.ops.get_graph_buffer_ipc_meta(self._inner._ptr)[1])
        except Exception:  # diagnostics must never break capture
            logger.debug("could not read pending graph buffers", exc_info=True)
            return -1

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        ok = self._inner.should_custom_ar(inp)
        if not ok:
            self._skipped += 1
            self._maybe_log()
        return ok

    def custom_all_reduce(self, inp: torch.Tensor):
        # Mirror CustomAllreduce.custom_all_reduce's own branch conditions so the
        # bucket reflects the path actually taken, without duplicating its work.
        if getattr(self._inner, "_IS_CAPTURING", False):
            if torch.cuda.is_current_stream_capturing():
                self._captured += 1
                # Record the slot-assignment order. The kernel's slot index is
                # graph_unreg_buffers_.size() at this moment, so appending here
                # mirrors it exactly.
                if self._capture_seq is not None:
                    self._capture_seq.append(
                        (tuple(inp.shape), str(inp.dtype), inp.numel())
                    )
            else:
                self._warmup += 1
        else:
            self._eager += 1
        self._maybe_log()
        return self._inner.custom_all_reduce(inp)

    def _check_capture_order(self, npass: int, pending) -> None:
        """All-gather each rank's capture sequence digest and compare.

        Slot ``i`` is only meaningful if every rank recorded the same buffer at
        index ``i``. Compares a per-prefix rolling digest so a mismatch reports
        the first diverging index rather than just "differs".
        """
        seq = self._capture_seq or []
        try:
            import hashlib

            import torch.distributed as dist

            group = getattr(self._inner, "group", None)
            if group is None or not dist.is_initialized():
                return

            def digest(items):
                h = hashlib.sha256()
                for it in items:
                    h.update(repr(it).encode())
                return h.hexdigest()

            payload = {
                "n": len(seq),
                "full": digest(seq),
                # checkpoints let us bisect to the first divergence
                "marks": {
                    i: digest(seq[:i])
                    for i in self._checkpoint_indices(len(seq))
                },
            }
            world = dist.get_world_size(group=group)
            gathered = [None] * world
            dist.all_gather_object(gathered, payload, group=group)

            ref = gathered[0]
            mismatched = [
                r for r, g in enumerate(gathered)
                if g["n"] != ref["n"] or g["full"] != ref["full"]
            ]
            if not mismatched:
                logger.info(
                    "[AR probe][order] pass #%d: capture sequence IDENTICAL "
                    "across all %d ranks (%d entries). Slot assignment is "
                    "consistent.",
                    npass,
                    world,
                    len(seq),
                )
                return

            logger.warning(
                "[AR probe][order] pass #%d: capture sequence DIVERGES across "
                "ranks %s -- rank-data slots do not describe the same buffers, "
                "so peer pointers are mismatched. lengths=%s",
                npass,
                mismatched,
                [g["n"] for g in gathered],
            )
            # Report the earliest checkpoint at which any rank differs.
            for i in sorted(ref["marks"]):
                bad = [r for r, g in enumerate(gathered)
                       if g["marks"].get(i) != ref["marks"][i]]
                if bad:
                    logger.warning(
                        "[AR probe][order] first divergence is at or before "
                        "capture index %d (ranks %s differ there). Local entry "
                        "at that index: %s",
                        i,
                        bad,
                        seq[i - 1] if 0 < i <= len(seq) else "n/a",
                    )
                    break
        except Exception:  # diagnostics must never break capture
            logger.debug("capture-order check failed", exc_info=True)

    @staticmethod
    def _checkpoint_indices(n: int, count: int = 64):
        """Evenly spaced prefix lengths used to bisect a divergence."""
        if n <= 0:
            return []
        step = max(1, n // count)
        marks = list(range(step, n + 1, step))
        if marks and marks[-1] != n:
            marks.append(n)
        return marks

    def _maybe_log(self) -> None:
        total = self._captured + self._eager + self._warmup + self._skipped
        if total == 0 or total % self._LOG_EVERY:
            return
        self._log("[AR probe]")

    def _log(self, prefix: str) -> None:
        logger.info(
            "%s custom_all_reduce calls: captured=%d eager=%d warmup=%d "
            "| should_custom_ar rejected (NCCL/RCCL)=%d "
            "| capture passes=%d, buffers registered across passes=%d.",
            prefix,
            self._captured,
            self._eager,
            self._warmup,
            self._skipped,
            self._capture_passes,
            self._registered_total,
        )

    def close(self):
        # Final tally, so short runs that never hit _LOG_EVERY still report.
        self._log("[AR probe][final]")
        return self._inner.close()


class CudaCommunicatorFL(CudaCommunicator):
    """CudaCommunicator that can enable custom allreduce on Hygon DCU."""

    # Entry-level accounting for VLLM_FL_HYGON_AR_PROBE=1. The ca_comm-level
    # probe showed zero calls after ~110k expected all-reduces, which leaves two
    # very different explanations. Counting here tells them apart:
    #   * entry count 0  -> tensor-parallel reduction never reaches this
    #     communicator at all, so custom allreduce has no attachment point in
    #     this model and the feature cannot help it.
    #   * entry count > 0 but ca_comm untouched -> the call arrives but an
    #     earlier branch inside CudaCommunicator.all_reduce (symm-mem, quick
    #     reduce, flashinfer) or a should_custom_ar rejection diverts it.
    _ENTRY_LOG_EVERY = 2000

    def all_reduce(self, input_):
        if not getattr(self, "_probe_entry", False):
            return super().all_reduce(input_)

        self._entry_calls += 1
        n = self._entry_calls
        if n == 1 or n % self._ENTRY_LOG_EVERY == 0:
            ca = self.ca_comm
            # Ask the UNDERLYING communicator, not the probe wrapper, so this
            # diagnostic does not inflate the probe's own `skipped` counter.
            target = getattr(ca, "_inner", ca)
            try:
                covered = target.should_custom_ar(input_) if ca is not None else "n/a"
            except Exception:  # diagnostics must never break the forward pass
                covered = "error"
            logger.info(
                "[AR probe][entry] %s.all_reduce calls=%d shape=%s dtype=%s "
                "ca_comm=%s should_custom_ar=%s",
                self.unique_name,
                n,
                tuple(input_.shape),
                input_.dtype,
                type(ca).__name__ if ca is not None else None,
                covered,
            )
        return super().all_reduce(input_)

    def __init__(self, *args, **kwargs):
        self._probe_entry = _ar_probe_enabled()
        self._entry_calls = 0
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

        # Try to activate the standalone NATIVE bf16 kernel. Must happen before
        # super().__init__ builds ca_comm so the object is constructed on our
        # ops. Only attempted for VLLM_FL_HYGON_CUSTOM_AR=2; mode 1 keeps the
        # shipped fp32-upcast proxy.
        mode = hygon_custom_ar_mode()
        native_bf16 = False
        if mode >= 2:
            native_bf16 = _redirect_custom_ar_to_hygon()
            if not native_bf16:
                # Requested native bf16 but the .so could not be loaded. Per
                # spec, do NOT silently downgrade to the fp32 proxy; disable
                # custom allreduce and let NCCL/RCCL serve the all-reduce.
                logger.warning(
                    "VLLM_FL_HYGON_CUSTOM_AR=2 requested the native bf16 "
                    "custom_ar kernel but it could not be loaded (set "
                    "VLLM_FL_HYGON_AR_SO or build "
                    "/workspace/hygon_custom_ar). Falling back to NCCL/RCCL "
                    "for tensor-parallel all-reduce (use "
                    "VLLM_FL_HYGON_CUSTOM_AR=1 for the fp32-upcast proxy)."
                )
                # super().__init__ still trips the bare is_cuda_alike() assert
                # in CustomAllreduce.__init__, so keep the window; then dispose
                # the (installed-ops) ca_comm it builds and null it so all
                # tensor-parallel all-reduces go through NCCL/RCCL.
                with _force_cuda_alike():
                    super().__init__(*args, **kwargs)
                if self.ca_comm is not None:
                    try:
                        self.ca_comm.close()
                    except Exception:  # cleanup must never break startup
                        logger.debug("ca_comm.close() failed", exc_info=True)
                    self.ca_comm = None
                return

        logger.info(
            "Enabling custom allreduce for Hygon DCU (mode=%d; "
            "set VLLM_FL_HYGON_CUSTOM_AR=0 to disable, 1=fp32 proxy, "
            "2=native bf16).",
            mode,
        )
        with _force_cuda_alike():
            super().__init__(*args, **kwargs)

        ca = self.ca_comm
        if ca is None or ca.disabled:
            logger.warning(
                "Custom allreduce is NOT active on Hygon; falling back to "
                "NCCL/RCCL for tensor-parallel all-reduce."
            )
        elif native_bf16:
            # ca_comm was built on the _C_hygon_custom_ar ops: bf16/fp16/fp32 all
            # run on the native kernel, no fp32 upcast, no proxy needed.
            logger.info(
                "Native bf16 custom allreduce is active on Hygon "
                "(world_size=%d, max_size=%d).",
                ca.world_size,
                ca.max_size,
            )
            _log_coverage(ca.max_size, native_bf16=True)
        else:
            logger.info(
                "Custom allreduce is active on Hygon via fp32 upcast "
                "(world_size=%d, max_size=%d).",
                ca.world_size,
                ca.max_size,
            )
            # No native bf16 kernel; wrap so bf16 all-reduces run losslessly in
            # fp32. `should_custom_ar` / `custom_all_reduce` are the only methods
            # CudaCommunicator.all_reduce invokes; proxy delegates the rest.
            self.ca_comm = _Fp32CustomAllreduce(ca)
            _log_coverage(ca.max_size, native_bf16=False)

        # Optional in/out-of-graph call accounting. Wraps whatever ca_comm ended
        # up being (native CustomAllreduce or the fp32 proxy) so both modes are
        # measurable with the same counters.
        if self.ca_comm is not None and _ar_probe_enabled():
            logger.info(
                "[AR probe] VLLM_FL_HYGON_AR_PROBE=1: counting custom "
                "allreduce calls inside vs outside CUDA graph capture."
            )
            self.ca_comm = _ARCaptureProbe(self.ca_comm)


def _log_coverage(max_size: int, native_bf16: bool = False) -> None:
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

    # The native bf16 kernel reduces in-dtype (2 B/element for bf16). The fp32
    # fallback upcasts, so the size checked against max_size is 4 B/element.
    if not native_bf16 and dtype_size < 4 and config.model_config.dtype in _UPCAST_DTYPES:
        dtype_size = 4

    kernel = "native" if native_bf16 else "fp32-upcast"
    bytes_per_token = hidden_size * dtype_size
    if bytes_per_token <= 0:
        return

    max_tokens = (max_size - 1) // bytes_per_token
    max_batched = config.scheduler_config.max_num_batched_tokens

    logger.info(
        "Custom allreduce coverage: %d B/token (hidden_size=%d, %s, %s kernel) "
        "-> payloads up to %d tokens use the custom kernel; larger batches fall "
        "back to NCCL/RCCL.",
        bytes_per_token,
        hidden_size,
        config.model_config.dtype,
        kernel,
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
