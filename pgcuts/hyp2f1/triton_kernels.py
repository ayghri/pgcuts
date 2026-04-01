"""Triton kernel for 2F1(-m, b; c; z).

Uses Kummer transformation + associative scan.
Broadcasting handled via strides.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _combine_fn(p_l, s_l, p_r, s_r):
    """Monoid combine for (carry, sum)."""
    return p_l * p_r, s_l + p_l * s_r


@triton.jit
def _triton_hyp2f1_kernel(
    b_ptr,
    c_ptr,
    z_ptr,
    pref_ptr,
    out_ptr,
    m,
    dim1,
    b_s0,
    b_s1,
    c_s0,
    c_s1,
    z_s0,
    z_s1,
    num_elements,
    block_size: tl.constexpr,  # pylint: disable=invalid-name
):
    """Triton kernel for 2F1 computation."""
    pid = tl.program_id(0)

    if pid >= num_elements:
        return

    # Decompose flat index into 2D coordinates
    i = pid // dim1
    j = pid % dim1

    # Stride-based loads
    b_val = tl.load(b_ptr + i * b_s0 + j * b_s1)
    c_val = tl.load(c_ptr + i * c_s0 + j * c_s1)
    z_val = tl.load(z_ptr + i * z_s0 + j * z_s1)

    pref = tl.load(pref_ptr + pid)

    # Edge: z ~ 0 -> 1.0
    if z_val < 1e-14:
        tl.store(out_ptr + pid, 1.0)
        return
    # Edge: z ~ 1 -> prefactor
    if z_val > 1.0 - 1e-14:
        tl.store(out_ptr + pid, pref)
        return

    # Kummer recurrence
    a_f = -tl.cast(m, tl.float64)
    b_val = b_val.to(tl.float64)
    c_val = c_val.to(tl.float64)
    z_val = z_val.to(tl.float64)
    z_k = 1.0 - z_val
    c_prime = a_f + b_val - c_val + 1.0

    # Partition m iterations across block_size lanes
    lane = tl.arange(0, block_size)
    chunk_size = (m + block_size - 1) // block_size
    k_start = lane * chunk_size

    # Local (carry, sum) per lane
    carry = tl.full(
        (block_size,), 1.0, dtype=tl.float64
    )
    local_sum = tl.where(
        lane == 0, 1.0, 0.0
    ).to(tl.float64)

    for step in range(chunk_size):
        k = k_start + step
        valid = k < m
        ratio = (
            (a_f + k)
            * (b_val + k)
            * z_k
            / ((c_prime + k) * (k + 1.0))
        )
        carry = carry * tl.where(valid, ratio, 1.0)
        local_sum = local_sum + tl.where(
            valid, carry, 0.0
        )

    # Reduction via associative scan
    _, scanned_sum = tl.associative_scan(
        (carry, local_sum),
        axis=0,
        combine_fn=_combine_fn,
    )

    # Last lane holds the full series sum
    total = tl.sum(
        tl.where(
            lane == block_size - 1, scanned_sum, 0.0
        ),
        axis=0,
    )

    tl.store(out_ptr + pid, pref * total)


def _prepare_args(a, b, c, z):
    """Convert inputs, compute broadcast shape."""
    m = -int(a)
    assert m >= 0, "a must be non-positive integer"

    device = z.device
    dtype = z.dtype

    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        return torch.tensor(
            x, dtype=dtype, device=device
        )

    b = _t(b)
    c = _t(c)

    out_shape = torch.broadcast_shapes(
        b.shape, c.shape, z.shape
    )

    b = b.expand(out_shape)
    c = c.expand(out_shape)
    z = z.expand(out_shape)

    ndim = len(out_shape)
    dim1 = 1
    if ndim == 0:
        b = b.unsqueeze(0)
        c = c.unsqueeze(0)
        z = z.unsqueeze(0)
        b_s0, b_s1 = 0, 0
        c_s0, c_s1 = 0, 0
        z_s0, z_s1 = 0, 0
    elif ndim == 1:
        b_s0, b_s1 = b.stride(0), 0
        c_s0, c_s1 = c.stride(0), 0
        z_s0, z_s1 = z.stride(0), 0
    elif ndim == 2:
        dim1 = out_shape[-1]
        b_s0, b_s1 = b.stride(0), b.stride(1)
        c_s0, c_s1 = c.stride(0), c.stride(1)
        z_s0, z_s1 = z.stride(0), z.stride(1)
    else:
        raise ValueError(
            "hyp2f1: inputs must be <=2D "
            f"after broadcast, got {ndim}D"
        )

    return (
        m,
        out_shape,
        b,
        c,
        z,
        dim1,
        b_s0,
        b_s1,
        c_s0,
        c_s1,
        z_s0,
        z_s1,
    )


def triton_hyp2f1(a, b, c, z, num_lanes=32):
    """Triton 2F1(-m, b; c; z) with precomputed prefactor.

    Args:
        a: Non-positive integer parameter.
        b: Second parameter.
        c: Third parameter.
        z: Values in [0, 1].
        num_lanes: Block size for parallel lanes.

    Returns:
        Tensor with 2F1 values.
    """
    (
        m,
        out_shape,
        b,
        c,
        z,
        dim1,
        b_s0,
        b_s1,
        c_s0,
        c_s1,
        z_s0,
        z_s1,
    ) = _prepare_args(a, b, c, z)

    if m == 0:
        return torch.ones(
            out_shape, dtype=z.dtype, device=z.device
        )

    # Prefactor in fp64 for accuracy
    c64 = c.double()
    b64 = b.double()
    prefactor = (
        torch.exp(
            torch.lgamma(c64)
            + torch.lgamma(c64 - b64 + m)
            - torch.lgamma(c64 + m)
            - torch.lgamma(c64 - b64)
        )
        .to(z.dtype)
        .reshape(-1)
    )

    num_elements = 1
    for s in out_shape:
        num_elements *= s
    out = torch.empty(
        out_shape, dtype=z.dtype, device=z.device
    )

    _triton_hyp2f1_kernel[(num_elements,)](
        b,
        c,
        z,
        prefactor,
        out,
        m,
        dim1,
        b_s0,
        b_s1,
        c_s0,
        c_s1,
        z_s0,
        z_s1,
        num_elements,
        block_size=num_lanes,
        num_warps=1,
    )
    return out
