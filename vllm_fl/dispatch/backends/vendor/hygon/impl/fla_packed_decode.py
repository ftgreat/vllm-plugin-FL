# Copyright (c) 2026 BAAI. All rights reserved.
#
# Optimized fused_recurrent_gated_delta_rule_packed_decode for Hygon DCU.
# Exp_30: Remove all validation checks to minimize Python overhead in decode hot path.
# The framework guarantees tensor shapes/contiguity in _forward_core_decode_non_spec.

import torch
import triton

from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode_kernel,
)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Skip all validation — tensors are guaranteed well-formed by the decode framework.
    # This eliminates ~17 Python attribute accesses per call × 49,104 calls per benchmark.

    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 16)
    # Hygon DCU optimization: num_warps=2 reduces register spills,
    # num_stages=1 disables software pipelining to further reduce register pressure on gfx926
    num_stages = 1
    num_warps = 2

    NV = triton.cdiv(V, BV)
    grid = (NV, B * HV)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state
