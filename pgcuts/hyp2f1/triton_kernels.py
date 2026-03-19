"""
Triton kernel for 2F1(-m, b; c; z) using Kummer transformation + associative scan.

Uses the same (carry, sum) monoid as the CUDA par_hyp2f1 kernel, but leverages
Triton's tl.associative_scan for the tree reduction.

Broadcasting is handled via strides — no materialized flat copies of b, c, z.

Prefactor Γ(c)Γ(c-b+m)/(Γ(c+m)Γ(c-b)) is precomputed via vectorised PyTorch
lgamma+exp before the kernel launch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _combine_fn(p_l, s_l, p_r, s_r):
    """(carry, sum) monoid: combine(L, R) = (L.p * R.p, L.s + L.p * R.s)"""
    return p_l * p_r, s_l + p_l * s_r


@triton.jit
def _triton_hyp2f1_kernel(
    b_ptr,
    c_ptr,
    z_ptr,
    pref_ptr,   # [N] float64
    out_ptr,    # [N] float64
    m,          # int: positive, a = -m
    m_f,        # float64: float(m)
    D1,         # int: inner dimension for 2D index decomposition
    b_s0, b_s1, # int: strides for b
    c_s0, c_s1, # int: strides for c
    z_s0, z_s1, # int: strides for z
    N,          # int: total output elements
    BLOCK_SIZE: tl.constexpr,     # P: lanes per element
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Decompose flat index into 2D coordinates
    i = pid // D1
    j = pid % D1

    # Stride-based loads (no materialized flat copies)
    b_val = tl.load(b_ptr + i * b_s0 + j * b_s1)
    c_val = tl.load(c_ptr + i * c_s0 + j * c_s1)
    z_val = tl.load(z_ptr + i * z_s0 + j * z_s1)

    pref = tl.load(pref_ptr + pid)

    # Edge: z ≈ 0 → 1.0
    if z_val < 1e-14:
        tl.store(out_ptr + pid, 1.0)
        return
    # Edge: z ≈ 1 → prefactor
    if z_val > 1.0 - 1e-14:
        tl.store(out_ptr + pid, pref)
        return

    # Kummer recurrence: 2F1(-m, b; C'; 1-z),  C' = -m + b - c + 1
    a_f = -tl.cast(m, tl.float64)
    z_k = 1.0 - z_val
    C_prime = a_f + b_val - c_val + 1.0

    # Partition m iterations across BLOCK_SIZE lanes
    lane = tl.arange(0, BLOCK_SIZE)
    chunk_size = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    k_start = lane * chunk_size

    # Local (carry, sum) per lane
    carry = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float64)
    local_sum = tl.where(lane == 0, 1.0, 0.0).to(tl.float64)

    for j in range(chunk_size):
        k = k_start + j
        valid = k < m
        kd = k.to(tl.float64)
        ratio = (a_f + kd) * (b_val + kd) * z_k / ((C_prime + kd) * (kd + 1.0))
        carry = carry * tl.where(valid, ratio, 1.0)
        local_sum = local_sum + tl.where(valid, carry, 0.0)

    # Tree reduction via associative scan
    _, scanned_sum = tl.associative_scan(
        (carry, local_sum), axis=0, combine_fn=_combine_fn
    )

    # Last lane holds the full series sum
    total = tl.sum(
        tl.where(lane == BLOCK_SIZE - 1, scanned_sum, 0.0), axis=0
    )

    tl.store(out_ptr + pid, pref * total)


# ── shared setup ──────────────────────────────────────────────────────────

def _prepare_args(a, b, c, z):
    """Convert inputs, compute broadcast shape and 2-D strides."""
    m = -int(a)
    assert m >= 0, "a must be non-positive integer"

    if isinstance(z, torch.Tensor):
        device = z.device
    else:
        device = torch.device("cuda:0")

    def _t(x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        return torch.tensor(x, dtype=dtype, device=device)

    z = _t(z, torch.float64)
    b = _t(b, torch.float64)
    c = _t(c, torch.float64)

    out_shape = torch.broadcast_shapes(b.shape, c.shape, z.shape)
    N = 1
    for s in out_shape:
        N *= s

    b = b.expand(out_shape)
    c = c.expand(out_shape)
    z = z.expand(out_shape)

    ndim = len(out_shape)
    if ndim == 0:
        D1 = 1
        b_s0, b_s1, c_s0, c_s1, z_s0, z_s1 = 0, 0, 0, 0, 0, 0
    elif ndim == 1:
        D1 = 1
        b_s0, b_s1 = b.stride(0), 0
        c_s0, c_s1 = c.stride(0), 0
        z_s0, z_s1 = z.stride(0), 0
    elif ndim == 2:
        D1 = out_shape[1]
        b_s0, b_s1 = b.stride(0), b.stride(1)
        c_s0, c_s1 = c.stride(0), c.stride(1)
        z_s0, z_s1 = z.stride(0), z.stride(1)
    else:
        D1 = out_shape[-1]
        b = b.contiguous().view(-1, D1)
        c = c.contiguous().view(-1, D1)
        z = z.contiguous().view(-1, D1)
        b_s0, b_s1 = b.stride(0), b.stride(1)
        c_s0, c_s1 = c.stride(0), c.stride(1)
        z_s0, z_s1 = z.stride(0), z.stride(1)

    return m, N, out_shape, device, b, c, z, D1, b_s0, b_s1, c_s0, c_s1, z_s0, z_s1


# ── public API ────────────────────────────────────────────────────────────

def triton_hyp2f1(a, b, c, z, P=32):
    """Triton 2F1(-m, b; c; z) with precomputed prefactor.

    Prefactor Γ(c)Γ(c-b+m)/(Γ(c+m)Γ(c-b)) is computed once via vectorised
    PyTorch lgamma+exp before the kernel launch.
    """
    m, N, out_shape, device, b, c, z, D1, b_s0, b_s1, c_s0, c_s1, z_s0, z_s1 = \
        _prepare_args(a, b, c, z)

    if m == 0:
        return torch.ones(out_shape, dtype=torch.float64, device=device)

    m_f = float(m)
    prefactor = torch.exp(
        torch.lgamma(c) + torch.lgamma(c - b + m_f)
        - torch.lgamma(c + m_f) - torch.lgamma(c - b)
    ).reshape(-1)

    out = torch.empty(N, dtype=torch.float64, device=device)

    _triton_hyp2f1_kernel[(N,)](
        b, c, z, prefactor, out,
        m, m_f,
        D1, b_s0, b_s1, c_s0, c_s1, z_s0, z_s1,
        N, BLOCK_SIZE=P, num_warps=1,
    )
    return out.reshape(out_shape)
