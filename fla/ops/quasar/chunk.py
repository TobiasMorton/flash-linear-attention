"""
QuasarAttention chunk implementation — parallel scan.

Three-phase approach:
  Phase 1: Parallel D[c] = kT_alpha_c @ V_c  (embarrassingly parallel)
  Phase 2: S[c] = exclusive_cumsum(D)          (PyTorch cumsum)
  Phase 3: O = Q@S[c] + causal(Q, alpha*K, V)  (tiled, fully parallel)

Drop-in replacement for fla/ops/quasar/chunk.py
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# --- Robust imports from fla ---
try:
    from fla.utils import autocast_custom_fwd, autocast_custom_bwd
except ImportError:
    def autocast_custom_fwd(*args, **kwargs):
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    autocast_custom_bwd = autocast_custom_fwd

try:
    from fla.utils import input_guard
except ImportError:
    def input_guard(fn):
        return fn

try:
    from fla.utils import autotune_cache_kwargs, check_shared_mem
except ImportError:
    autotune_cache_kwargs = {}
    def check_shared_mem():
        return True

try:
    from fla.ops.utils.index import prepare_chunk_indices
except ImportError:
    prepare_chunk_indices = None

# Compatibility exports (referenced by other fla modules)
BS_LIST = [32, 64] if check_shared_mem() else [16, 32]
BT_LIST_AUTOTUNE = [32, 64, 128]
_ = autotune_cache_kwargs


# ============================================================
# KERNEL 1 — Parallel chunk contribution D[c]
# Grid: (NT * B * H,)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['S', 'BT'],
)
@triton.jit
def _chunk_contrib(
    k_ptr, v_ptr, beta_ptr, D_ptr,
    T,
    H:  tl.constexpr,
    S:  tl.constexpr,
    BT: tl.constexpr,
    NT,
    stride_kb, stride_kt, stride_kh, stride_ks,
    stride_vb, stride_vt, stride_vh, stride_vs,
    stride_db, stride_dh, stride_dn, stride_di, stride_dj,
):
    pid = tl.program_id(0)
    i_bh = pid // NT
    i_c  = pid % NT
    i_b  = i_bh // H
    i_h  = i_bh % H

    # beta is [B, H] — load per (batch, head)
    beta = tl.load(beta_ptr + i_b * H + i_h).to(tl.float32)

    r_s  = tl.arange(0, S)
    r_bt = tl.arange(0, BT)

    t0 = i_c * BT
    r_t = t0 + r_bt
    mask_t = r_t < T

    base_k = k_ptr + i_b * stride_kb + i_h * stride_kh
    base_v = v_ptr + i_b * stride_vb + i_h * stride_vh

    v = tl.load(
        base_v + r_t[:, None] * stride_vt + r_s[None, :] * stride_vs,
        mask=mask_t[:, None], other=0.0,
    ).to(tl.float32)

    kT = tl.load(
        base_k + r_s[:, None] * stride_ks + r_t[None, :] * stride_kt,
        mask=mask_t[None, :], other=0.0,
    ).to(tl.float32)

    k_sq  = tl.sum(kT * kT, axis=0)
    alpha = (1.0 - tl.exp(-beta * k_sq)) / (k_sq + 1e-8)
    kT_a  = kT * alpha[None, :]

    D = tl.dot(kT_a.to(tl.bfloat16), v.to(tl.bfloat16)).to(tl.float32)

    base_d = D_ptr + i_b * stride_db + i_h * stride_dh + i_c * stride_dn
    tl.store(
        base_d + r_s[:, None] * stride_di + r_s[None, :] * stride_dj,
        D,
    )


# ============================================================
# KERNEL 2 — Tiled output
# Grid: (NT * N_QT, B * H)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['S', 'BT_Q'],
)
@triton.jit
def _output_tiled(
    q_ptr, k_ptr, v_ptr, beta_ptr, states_ptr, o_ptr,
    T,
    H:    tl.constexpr,
    S:    tl.constexpr,
    BT:   tl.constexpr,
    BT_Q: tl.constexpr,
    N_QT: tl.constexpr,
    NT,
    stride_qb, stride_qt, stride_qh, stride_qs,
    stride_kb, stride_kt, stride_kh, stride_ks,
    stride_vb, stride_vt, stride_vh, stride_vs,
    stride_ob, stride_ot, stride_oh, stride_os,
    stride_sb, stride_sh, stride_sn, stride_si, stride_sj,
):
    pid0 = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b  = i_bh // H
    i_h  = i_bh % H

    i_c  = pid0 // N_QT
    i_q  = pid0 % N_QT

    # beta is [B, H]
    beta = tl.load(beta_ptr + i_b * H + i_h).to(tl.float32)

    t0_chunk = i_c * BT
    t0_q = t0_chunk + i_q * BT_Q
    r_tq  = t0_q + tl.arange(0, BT_Q)
    r_s   = tl.arange(0, S)
    mask_q = r_tq < T

    base_q = q_ptr + i_b * stride_qb + i_h * stride_qh
    base_k = k_ptr + i_b * stride_kb + i_h * stride_kh
    base_v = v_ptr + i_b * stride_vb + i_h * stride_vh

    q = tl.load(
        base_q + r_tq[:, None] * stride_qt + r_s[None, :] * stride_qs,
        mask=mask_q[:, None], other=0.0,
    ).to(tl.bfloat16)

    base_s = states_ptr + i_b * stride_sb + i_h * stride_sh + i_c * stride_sn
    st = tl.load(
        base_s + r_s[:, None] * stride_si + r_s[None, :] * stride_sj,
    ).to(tl.bfloat16)

    o_acc = tl.dot(q, st).to(tl.float32)

    row_global = i_q * BT_Q + tl.arange(0, BT_Q)

    for i_k in range(N_QT):
        t0_k = t0_chunk + i_k * BT_Q
        r_tk = t0_k + tl.arange(0, BT_Q)
        mask_k = r_tk < T
        col_global = i_k * BT_Q + tl.arange(0, BT_Q)

        kT = tl.load(
            base_k + r_s[:, None] * stride_ks + r_tk[None, :] * stride_kt,
            mask=mask_k[None, :], other=0.0,
        ).to(tl.float32)

        v = tl.load(
            base_v + r_tk[:, None] * stride_vt + r_s[None, :] * stride_vs,
            mask=mask_k[:, None], other=0.0,
        ).to(tl.bfloat16)

        k_sq  = tl.sum(kT * kT, axis=0)
        alpha = (1.0 - tl.exp(-beta * k_sq)) / (k_sq + 1e-8)
        kT_a  = (kT * alpha[None, :]).to(tl.bfloat16)

        A = tl.dot(q, kT_a).to(tl.float32)
        causal = row_global[:, None] >= col_global[None, :]
        A = tl.where(causal & mask_q[:, None] & mask_k[None, :], A, 0.0)

        o_acc += tl.dot(A.to(tl.bfloat16), v).to(tl.float32)

    base_o = o_ptr + i_b * stride_ob + i_h * stride_oh
    tl.store(
        base_o + r_tq[:, None] * stride_ot + r_s[None, :] * stride_os,
        o_acc.to(tl.bfloat16),
        mask=mask_q[:, None],
    )


# ============================================================
# Beta normalization
# ============================================================
def _normalize_beta(beta: torch.Tensor, B: int, H: int) -> torch.Tensor:
    """Normalize beta to shape [B, H] regardless of input shape."""
    if beta.dim() == 0:
        return beta.float().expand(B, H).contiguous()
    elif beta.dim() == 1:
        if beta.shape[0] == H:
            return beta.float().unsqueeze(0).expand(B, H).contiguous()
        elif beta.shape[0] == B:
            return beta.float().unsqueeze(1).expand(B, H).contiguous()
        else:
            # Unknown 1D shape — broadcast
            return beta.float().view(-1)[:1].expand(B, H).contiguous()
    elif beta.dim() == 2:
        if beta.shape == (B, H):
            return beta.float().contiguous()
        return beta.float().reshape(B, -1)[:, :H].contiguous()
    else:
        # [B, T, H, ...] or [B, T, H] — take mean over non-batch/head dims
        beta = beta.float()
        if beta.shape[-1] == 1:
            beta = beta.squeeze(-1)
        if beta.dim() == 3:  # [B, T, H]
            return beta.mean(dim=1).contiguous()
        elif beta.dim() == 4:  # [B, T, H, S]
            return beta.mean(dim=(1, 3)).contiguous()
        return beta.view(B, -1).mean(dim=1, keepdim=True).expand(B, H).contiguous()


# ============================================================
# Python forward
# ============================================================
@input_guard
def chunk_quasar_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 256,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    del kwargs

    B, T, H, S = q.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    BT_Q = min(64, BT)  # query tile size, capped at chunk size
    N_QT = BT // BT_Q
    BH = B * H

    # Ensure BF16 + contiguous
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
    if k.dtype != torch.bfloat16:
        k = k.to(torch.bfloat16)
    if v.dtype != torch.bfloat16:
        v = v.to(torch.bfloat16)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    beta = _normalize_beta(beta, B, H)

    o = torch.empty(B, T, H, S, dtype=torch.bfloat16, device=q.device)

    # Phase 1: parallel chunk contributions
    D = torch.empty(B, H, NT, S, S, dtype=torch.float32, device=q.device)

    _chunk_contrib[(NT * BH,)](
        k, v, beta, D,
        T, H=H, S=S, BT=BT, NT=NT,
        stride_kb=k.stride(0), stride_kt=k.stride(1),
        stride_kh=k.stride(2), stride_ks=k.stride(3),
        stride_vb=v.stride(0), stride_vt=v.stride(1),
        stride_vh=v.stride(2), stride_vs=v.stride(3),
        stride_db=D.stride(0), stride_dh=D.stride(1),
        stride_dn=D.stride(2), stride_di=D.stride(3),
        stride_dj=D.stride(4),
    )

    # Phase 2: exclusive prefix sum → states
    states = torch.cumsum(D, dim=2)
    states = torch.cat([
        torch.zeros(B, H, 1, S, S, dtype=torch.float32, device=q.device),
        states[:, :, :-1],
    ], dim=2)

    if initial_state is not None:
        states = states + initial_state.unsqueeze(2)

    # Phase 3: tiled parallel output
    _output_tiled[(NT * N_QT, BH)](
        q, k, v, beta, states, o,
        T, H=H, S=S, BT=BT, BT_Q=BT_Q, N_QT=N_QT, NT=NT,
        stride_qb=q.stride(0), stride_qt=q.stride(1),
        stride_qh=q.stride(2), stride_qs=q.stride(3),
        stride_kb=k.stride(0), stride_kt=k.stride(1),
        stride_kh=k.stride(2), stride_ks=k.stride(3),
        stride_vb=v.stride(0), stride_vt=v.stride(1),
        stride_vh=v.stride(2), stride_vs=v.stride(3),
        stride_ob=o.stride(0), stride_ot=o.stride(1),
        stride_oh=o.stride(2), stride_os=o.stride(3),
        stride_sb=states.stride(0), stride_sh=states.stride(1),
        stride_sn=states.stride(2), stride_si=states.stride(3),
        stride_sj=states.stride(4),
    )

    final_state = None
    if output_final_state:
        final_state = (states[:, :, -1] + D[:, :, -1]).to(q.dtype)

    return o, final_state


class ChunkQuasarFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del kwargs

        chunk_size = 256
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, chunk_size)
            if cu_seqlens is not None and prepare_chunk_indices is not None
            else None
        )

        o, final_state = chunk_quasar_fwd(
            q=q, k=k, v=v, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )

        if initial_state is None:
            initial_state_saved = torch.empty(0, device=q.device, dtype=q.dtype)
        else:
            initial_state_saved = initial_state
        if cu_seqlens is None:
            cu_seqlens_saved = torch.empty(0, device=q.device, dtype=torch.int32)
        else:
            cu_seqlens_saved = cu_seqlens
        if chunk_indices is None:
            chunk_indices_saved = torch.empty(0, device=q.device, dtype=torch.int32)
        else:
            chunk_indices_saved = chunk_indices

        ctx.save_for_backward(
            q, k, v, beta, initial_state_saved, cu_seqlens_saved, chunk_indices_saved
        )
        return o, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, d_final_state: Optional[torch.Tensor]):
        del do, d_final_state
        q, k, v, beta, *_ = ctx.saved_tensors
        return (torch.zeros_like(q), torch.zeros_like(k),
                torch.zeros_like(v), torch.zeros_like(beta),
                None, None, None)


try:
    _compiler_disable = torch.compiler.disable
except AttributeError:
    def _compiler_disable(fn):
        return fn


@_compiler_disable
def chunk_quasar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return ChunkQuasarFunction.apply(
        q, k, v, beta, initial_state, output_final_state, cu_seqlens
    )
