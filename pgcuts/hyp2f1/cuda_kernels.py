"""
GPU-accelerated 2F1(-m, b; c; z) for non-positive integer a.

Uses CUDA kernels (JIT-compiled via torch.utils.cpp_extension.load_inline)
with N-D stride-based tensor access — no .contiguous() calls, no materialized
flat copies of b, c, z for any number of dimensions.

Supports full PyTorch broadcasting: b, c, z can be any broadcastable shapes.
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_NDIM 8

// Stride info passed by value to kernels (lives in constant memory).
struct StridesInfo {
    int ndim;
    int64_t shape[MAX_NDIM];
    int64_t b_s[MAX_NDIM];
    int64_t c_s[MAX_NDIM];
    int64_t z_s[MAX_NDIM];
};

// Convert flat element index to memory offset for a strided tensor.
__device__ __forceinline__ int64_t flat_to_offset(
    int64_t flat_idx, int ndim,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ strides
) {
    int64_t offset = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        offset += (flat_idx % shape[d]) * strides[d];
        flat_idx /= shape[d];
    }
    return offset;
}

// Helper: build StridesInfo from CPU tensors (called from forward functions).
static StridesInfo build_strides_info(
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides
) {
    StridesInfo si;
    si.ndim = ndim;
    auto sp = shape.data_ptr<int64_t>();
    auto bp = b_strides.data_ptr<int64_t>();
    auto cp = c_strides.data_ptr<int64_t>();
    auto zp = z_strides.data_ptr<int64_t>();
    for (int d = 0; d < ndim && d < MAX_NDIM; d++) {
        si.shape[d] = sp[d];
        si.b_s[d] = bp[d];
        si.c_s[d] = cp[d];
        si.z_s[d] = zp[d];
    }
    return si;
}

// ---- hyp2f1: Pfaff/Kummer branching kernel ----
__global__ void hyp2f1_kernel(
    int a_val,
    const double* __restrict__ b_data,
    const double* __restrict__ c_data,
    const double* __restrict__ z_data,
    double*       __restrict__ out_data,
    StridesInfo si,
    int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double b_val = b_data[flat_to_offset(idx, si.ndim, si.shape, si.b_s)];
    const double c_val = c_data[flat_to_offset(idx, si.ndim, si.shape, si.c_s)];
    const double z_val = z_data[flat_to_offset(idx, si.ndim, si.shape, si.z_s)];

    constexpr double EPS       = 1e-14;
    constexpr double THRESHOLD = 0.3819660112501051;

    if (a_val == 0 || fabs(z_val) < EPS) {
        out_data[idx] = 1.0;
        return;
    }

    const int    m   = -a_val;
    const double m_d = (double)m;
    const double a_d = (double)a_val;

    if (fabs(z_val - 1.0) < EPS) {
        out_data[idx] = exp(
            lgamma(c_val) + lgamma(c_val - b_val + m_d)
            - lgamma(c_val + m_d) - lgamma(c_val - b_val)
        );
        return;
    }

    const double pfaff_limit = 1.0 - exp(-700.0 / m_d);
    const double threshold   = fmin(THRESHOLD, pfaff_limit);

    double result;

    if (z_val < threshold) {
        const double one_minus_z = 1.0 - z_val;
        const double z_pfaff     = z_val / (z_val - 1.0);
        const double c_minus_b   = c_val - b_val;

        double term = pow(one_minus_z, m_d);
        double sum  = term;
        for (int k = 0; k < m; ++k) {
            const double kd = (double)k;
            term *= (a_d + kd) * (c_minus_b + kd) * z_pfaff
                    / ((c_val + kd) * (kd + 1.0));
            sum += term;
        }
        result = sum;
    } else {
        const double z_k     = 1.0 - z_val;
        const double C_prime = a_d + b_val - c_val + 1.0;

        const double prefactor = exp(
            lgamma(c_val) + lgamma(c_val - b_val + m_d)
            - lgamma(c_val + m_d) - lgamma(c_val - b_val)
        );

        double term = prefactor;
        double sum  = term;
        for (int k = 0; k < m; ++k) {
            const double kd  = (double)k;
            const double num = (a_d + kd) * (b_val + kd) * z_k;
            const double den = (C_prime + kd) * (kd + 1.0);
            term *= num / den;
            sum  += term;
        }
        result = sum;
    }

    out_data[idx] = result;
}

// ---- fast_hyp2f1: branchless Kummer-only kernel ----
__global__ void fast_hyp2f1_kernel(
    int a_val,
    const double* __restrict__ b_data,
    const double* __restrict__ c_data,
    const double* __restrict__ z_data,
    double*       __restrict__ out_data,
    StridesInfo si,
    int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double b_val = b_data[flat_to_offset(idx, si.ndim, si.shape, si.b_s)];
    const double c_val = c_data[flat_to_offset(idx, si.ndim, si.shape, si.c_s)];
    const double z_val = z_data[flat_to_offset(idx, si.ndim, si.shape, si.z_s)];

    constexpr double EPS = 1e-14;

    if (a_val == 0 || fabs(z_val) < EPS) {
        out_data[idx] = 1.0;
        return;
    }

    const int    m   = -a_val;
    const double m_d = (double)m;
    const double a_d = (double)a_val;

    const double prefactor = exp(
        lgamma(c_val) + lgamma(c_val - b_val + m_d)
        - lgamma(c_val + m_d) - lgamma(c_val - b_val)
    );

    if (fabs(z_val - 1.0) < EPS) {
        out_data[idx] = prefactor;
        return;
    }

    const double z_k     = 1.0 - z_val;
    const double C_prime = a_d + b_val - c_val + 1.0;

    double term = prefactor;
    double sum  = term;
    for (int k = 0; k < m; ++k) {
        const double kd  = (double)k;
        term *= (a_d + kd) * (b_val + kd) * z_k
                / ((C_prime + kd) * (kd + 1.0));
        sum  += term;
    }

    out_data[idx] = sum;
}

// ---- mp_hyp2f1: direct series (no transformations) ----
__global__ void mp_hyp2f1_kernel(
    int a_val,
    const double* __restrict__ b_data,
    const double* __restrict__ c_data,
    const double* __restrict__ z_data,
    double*       __restrict__ out_data,
    StridesInfo si,
    int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double b_val = b_data[flat_to_offset(idx, si.ndim, si.shape, si.b_s)];
    const double c_val = c_data[flat_to_offset(idx, si.ndim, si.shape, si.c_s)];
    const double z_val = z_data[flat_to_offset(idx, si.ndim, si.shape, si.z_s)];

    constexpr double EPS = 1e-14;

    if (a_val == 0 || fabs(z_val) < EPS) {
        out_data[idx] = 1.0;
        return;
    }

    const int    m   = -a_val;
    const double a_d = (double)a_val;

    if (fabs(z_val - 1.0) < EPS) {
        const double m_d = (double)m;
        out_data[idx] = exp(
            lgamma(c_val) + lgamma(c_val - b_val + m_d)
            - lgamma(c_val + m_d) - lgamma(c_val - b_val)
        );
        return;
    }

    double sum  = 1.0;
    double term = 1.0;
    for (int k = 0; k < m; ++k) {
        const double kd = (double)k;
        term *= (a_d + kd) * (b_val + kd) * z_val
                / ((c_val + kd) * (kd + 1.0));
        sum += term;
    }

    out_data[idx] = sum;
}

// ---- C++ forward functions ----

torch::Tensor hyp2f1_forward(
    int a_val,
    torch::Tensor b, torch::Tensor c, torch::Tensor z,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N
) {
    auto out = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat64).device(z.device()));
    if (N == 0) return out;

    auto si = build_strides_info(ndim, shape, b_strides, c_strides, z_strides);
    constexpr int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    hyp2f1_kernel<<<blocks, threads>>>(
        a_val, b.data_ptr<double>(), c.data_ptr<double>(), z.data_ptr<double>(),
        out.data_ptr<double>(), si, N
    );
    return out;
}

torch::Tensor fast_hyp2f1_forward(
    int a_val,
    torch::Tensor b, torch::Tensor c, torch::Tensor z,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N
) {
    auto out = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat64).device(z.device()));
    if (N == 0) return out;

    auto si = build_strides_info(ndim, shape, b_strides, c_strides, z_strides);
    constexpr int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    fast_hyp2f1_kernel<<<blocks, threads>>>(
        a_val, b.data_ptr<double>(), c.data_ptr<double>(), z.data_ptr<double>(),
        out.data_ptr<double>(), si, N
    );
    return out;
}

torch::Tensor mp_hyp2f1_forward(
    int a_val,
    torch::Tensor b, torch::Tensor c, torch::Tensor z,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N
) {
    auto out = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat64).device(z.device()));
    if (N == 0) return out;

    auto si = build_strides_info(ndim, shape, b_strides, c_strides, z_strides);
    constexpr int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    mp_hyp2f1_kernel<<<blocks, threads>>>(
        a_val, b.data_ptr<double>(), c.data_ptr<double>(), z.data_ptr<double>(),
        out.data_ptr<double>(), si, N
    );
    return out;
}

// ---- par_hyp2f1: P-thread-per-element parallel Kummer kernel ----
template <int P>
__global__ void par_hyp2f1_kernel(
    int a_val,
    const double* __restrict__ b_data,
    const double* __restrict__ c_data,
    const double* __restrict__ z_data,
    double*       __restrict__ out_data,
    StridesInfo si,
    int64_t N
) {
    const int64_t tid  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t elem = tid / P;
    const int     lane = tid % P;

    double b_val = 1.0, c_val = 2.0, z_val = 0.0;
    if (elem < N) {
        b_val = b_data[flat_to_offset(elem, si.ndim, si.shape, si.b_s)];
        c_val = c_data[flat_to_offset(elem, si.ndim, si.shape, si.c_s)];
        z_val = z_data[flat_to_offset(elem, si.ndim, si.shape, si.z_s)];
    }

    constexpr double EPS = 1e-14;
    const int    m   = -a_val;
    const double m_d = (double)m;
    const double a_d = (double)a_val;

    const bool is_zero   = (a_val == 0 || fabs(z_val) < EPS);
    const bool is_one    = (!is_zero && fabs(z_val - 1.0) < EPS);
    const bool need_loop = (!is_zero && !is_one);

    const double z_k     = 1.0 - z_val;
    const double C_prime = a_d + b_val - c_val + 1.0;

    const int h       = m / P;
    const int k_start = lane * h;
    const int k_end   = (lane == P - 1) ? m : (lane + 1) * h;

    double carry  = 1.0;
    double my_sum = (lane == 0) ? 1.0 : 0.0;

    if (need_loop) {
        for (int k = k_start; k < k_end; ++k) {
            const double kd = (double)k;
            carry *= (a_d + kd) * (b_val + kd) * z_k
                     / ((C_prime + kd) * (kd + 1.0));
            my_sum += carry;
        }
    }

    const unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int delta = 1; delta < P; delta <<= 1) {
        double p_sum   = __shfl_xor_sync(mask, my_sum, delta);
        double p_carry = __shfl_xor_sync(mask, carry,  delta);
        if ((lane & delta) == 0) {
            my_sum = my_sum + carry * p_sum;
            carry  = carry  * p_carry;
        }
    }

    if (lane == 0 && elem < N) {
        if (is_zero) {
            out_data[elem] = 1.0;
        } else {
            const double prefactor = exp(
                lgamma(c_val) + lgamma(c_val - b_val + m_d)
                - lgamma(c_val + m_d) - lgamma(c_val - b_val)
            );
            out_data[elem] = is_one ? prefactor : prefactor * my_sum;
        }
    }
}

torch::Tensor par_hyp2f1_forward(
    int a_val,
    torch::Tensor b, torch::Tensor c, torch::Tensor z,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N, int64_t P
) {
    auto out = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat64).device(z.device()));
    if (N == 0) return out;

    auto si = build_strides_info(ndim, shape, b_strides, c_strides, z_strides);
    constexpr int threads = 256;
    const int blocks = (P * N + threads - 1) / threads;

    auto b_p = b.data_ptr<double>();
    auto c_p = c.data_ptr<double>();
    auto z_p = z.data_ptr<double>();
    auto o_p = out.data_ptr<double>();

    switch (P) {
        case 1:  par_hyp2f1_kernel<1> <<<blocks, threads>>>(a_val, b_p, c_p, z_p, o_p, si, N); break;
        case 2:  par_hyp2f1_kernel<2> <<<blocks, threads>>>(a_val, b_p, c_p, z_p, o_p, si, N); break;
        case 4:  par_hyp2f1_kernel<4> <<<blocks, threads>>>(a_val, b_p, c_p, z_p, o_p, si, N); break;
        case 8:  par_hyp2f1_kernel<8> <<<blocks, threads>>>(a_val, b_p, c_p, z_p, o_p, si, N); break;
        case 16: par_hyp2f1_kernel<16><<<blocks, threads>>>(a_val, b_p, c_p, z_p, o_p, si, N); break;
        case 32: par_hyp2f1_kernel<32><<<blocks, threads>>>(a_val, b_p, c_p, z_p, o_p, si, N); break;
        default: TORCH_CHECK(false, "P must be 1, 2, 4, 8, 16, or 32");
    }
    return out;
}

// ---- par_hyp2f1_precomp: same as par_hyp2f1 but with precomputed prefactor ----
template <int P>
__global__ void par_hyp2f1_precomp_kernel(
    int a_val,
    const double* __restrict__ b_data,
    const double* __restrict__ c_data,
    const double* __restrict__ z_data,
    const double* __restrict__ pref_data,
    double*       __restrict__ out_data,
    StridesInfo si,
    int64_t N
) {
    const int64_t tid  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t elem = tid / P;
    const int     lane = tid % P;

    double b_val = 1.0, c_val = 2.0, z_val = 0.0;
    if (elem < N) {
        b_val = b_data[flat_to_offset(elem, si.ndim, si.shape, si.b_s)];
        c_val = c_data[flat_to_offset(elem, si.ndim, si.shape, si.c_s)];
        z_val = z_data[flat_to_offset(elem, si.ndim, si.shape, si.z_s)];
    }

    constexpr double EPS = 1e-14;
    const int    m   = -a_val;
    const double a_d = (double)a_val;

    const bool is_zero   = (a_val == 0 || fabs(z_val) < EPS);
    const bool is_one    = (!is_zero && fabs(z_val - 1.0) < EPS);
    const bool need_loop = (!is_zero && !is_one);

    const double z_k     = 1.0 - z_val;
    const double C_prime = a_d + b_val - c_val + 1.0;

    const int h       = m / P;
    const int k_start = lane * h;
    const int k_end   = (lane == P - 1) ? m : (lane + 1) * h;

    double carry  = 1.0;
    double my_sum = (lane == 0) ? 1.0 : 0.0;

    if (need_loop) {
        for (int k = k_start; k < k_end; ++k) {
            const double kd = (double)k;
            carry *= (a_d + kd) * (b_val + kd) * z_k
                     / ((C_prime + kd) * (kd + 1.0));
            my_sum += carry;
        }
    }

    const unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int delta = 1; delta < P; delta <<= 1) {
        double p_sum   = __shfl_xor_sync(mask, my_sum, delta);
        double p_carry = __shfl_xor_sync(mask, carry,  delta);
        if ((lane & delta) == 0) {
            my_sum = my_sum + carry * p_sum;
            carry  = carry  * p_carry;
        }
    }

    if (lane == 0 && elem < N) {
        if (is_zero) {
            out_data[elem] = 1.0;
        } else {
            const double prefactor = pref_data[elem];
            out_data[elem] = is_one ? prefactor : prefactor * my_sum;
        }
    }
}

torch::Tensor par_hyp2f1_precomp_forward(
    int a_val,
    torch::Tensor b, torch::Tensor c, torch::Tensor z,
    torch::Tensor pref,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N, int64_t P
) {
    auto out = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat64).device(z.device()));
    if (N == 0) return out;

    auto si = build_strides_info(ndim, shape, b_strides, c_strides, z_strides);
    constexpr int threads = 256;
    const int blocks = (P * N + threads - 1) / threads;

    auto b_p = b.data_ptr<double>();
    auto c_p = c.data_ptr<double>();
    auto z_p = z.data_ptr<double>();
    auto pf  = pref.data_ptr<double>();
    auto o_p = out.data_ptr<double>();

    switch (P) {
        case 1:  par_hyp2f1_precomp_kernel<1> <<<blocks, threads>>>(a_val, b_p, c_p, z_p, pf, o_p, si, N); break;
        case 2:  par_hyp2f1_precomp_kernel<2> <<<blocks, threads>>>(a_val, b_p, c_p, z_p, pf, o_p, si, N); break;
        case 4:  par_hyp2f1_precomp_kernel<4> <<<blocks, threads>>>(a_val, b_p, c_p, z_p, pf, o_p, si, N); break;
        case 8:  par_hyp2f1_precomp_kernel<8> <<<blocks, threads>>>(a_val, b_p, c_p, z_p, pf, o_p, si, N); break;
        case 16: par_hyp2f1_precomp_kernel<16><<<blocks, threads>>>(a_val, b_p, c_p, z_p, pf, o_p, si, N); break;
        case 32: par_hyp2f1_precomp_kernel<32><<<blocks, threads>>>(a_val, b_p, c_p, z_p, pf, o_p, si, N); break;
        default: TORCH_CHECK(false, "P must be 1, 2, 4, 8, 16, or 32");
    }
    return out;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor hyp2f1_forward(
    int a_val, torch::Tensor b, torch::Tensor c, torch::Tensor z,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N);
torch::Tensor fast_hyp2f1_forward(
    int a_val, torch::Tensor b, torch::Tensor c, torch::Tensor z,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N);
torch::Tensor mp_hyp2f1_forward(
    int a_val, torch::Tensor b, torch::Tensor c, torch::Tensor z,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N);
torch::Tensor par_hyp2f1_forward(
    int a_val, torch::Tensor b, torch::Tensor c, torch::Tensor z,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N, int64_t P);
torch::Tensor par_hyp2f1_precomp_forward(
    int a_val, torch::Tensor b, torch::Tensor c, torch::Tensor z,
    torch::Tensor pref,
    int ndim, torch::Tensor shape,
    torch::Tensor b_strides, torch::Tensor c_strides, torch::Tensor z_strides,
    int64_t N, int64_t P);
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="hyp2f1_jit",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["hyp2f1_forward", "fast_hyp2f1_forward",
                        "mp_hyp2f1_forward", "par_hyp2f1_forward",
                        "par_hyp2f1_precomp_forward"],
            verbose=False,
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
        )
    return _module


def _prepare_args(a, b, c, z):
    """Convert inputs, compute broadcast shape and N-D strides.

    Returns stride metadata as CPU int64 tensors (passed by value to kernel
    via StridesInfo struct — no GPU allocation, no .contiguous() calls).
    """
    a_val = int(a)
    m = -a_val
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
    # CPU tensors — tiny, no GPU alloc; read in C++ host code to fill struct
    shape_t = torch.tensor(list(out_shape), dtype=torch.int64)
    b_strides_t = torch.tensor(list(b.stride()), dtype=torch.int64) if ndim > 0 else torch.empty(0, dtype=torch.int64)
    c_strides_t = torch.tensor(list(c.stride()), dtype=torch.int64) if ndim > 0 else torch.empty(0, dtype=torch.int64)
    z_strides_t = torch.tensor(list(z.stride()), dtype=torch.int64) if ndim > 0 else torch.empty(0, dtype=torch.int64)

    return (a_val, m, N, out_shape, device, b, c, z,
            ndim, shape_t, b_strides_t, c_strides_t, z_strides_t)


def _stride_args(prepared):
    """Extract the stride arguments tuple from _prepare_args result."""
    return prepared[8:]  # ndim, shape_t, b_strides_t, c_strides_t, z_strides_t


def hyp2f1(a, b, c, z):
    """Compute 2F1(a, b; c; z) for non-positive integer *a* on GPU.

    Uses Pfaff transformation for small z and Kummer connection for z near 1,
    with automatic crossover selection.

    Parameters
    ----------
    a : int   Non-positive integer.
    b : float | Tensor  Positive real(s).
    c : float | Tensor  Positive real(s), c > b.
    z : Tensor  Values in [0, 1].

    Returns
    -------
    Tensor  Same shape as ``torch.broadcast_shapes(b, c, z)``.
    """
    p = _prepare_args(a, b, c, z)
    a_val, m, N, out_shape, device, b, c, z = p[:8]
    ndim, shape_t, b_st, c_st, z_st = p[8:]
    module = _get_module()
    result = module.hyp2f1_forward(
        a_val, b, c, z, ndim, shape_t, b_st, c_st, z_st, N)
    return result.reshape(out_shape)


def fast_hyp2f1(a, b, c, z):
    """Branchless Kummer-only GPU 2F1.  No warp divergence.

    Same API as :func:`hyp2f1`.  Uses Kummer connection for all z.
    Fastest single-thread variant; accurate for typical b values (b <= ~5).
    """
    p = _prepare_args(a, b, c, z)
    a_val, m, N, out_shape, device, b, c, z = p[:8]
    ndim, shape_t, b_st, c_st, z_st = p[8:]
    module = _get_module()
    result = module.fast_hyp2f1_forward(
        a_val, b, c, z, ndim, shape_t, b_st, c_st, z_st, N)
    return result.reshape(out_shape)


def mp_hyp2f1(a, b, c, z):
    """Direct-series GPU 2F1 (mpmath-style, no transformations).

    Uses the plain forward recurrence without Pfaff/Kummer transformations.
    Accurate for small |a| but produces NaN/Inf for large |a| and z near 1.
    """
    p = _prepare_args(a, b, c, z)
    a_val, m, N, out_shape, device, b, c, z = p[:8]
    ndim, shape_t, b_st, c_st, z_st = p[8:]
    module = _get_module()
    result = module.mp_hyp2f1_forward(
        a_val, b, c, z, ndim, shape_t, b_st, c_st, z_st, N)
    return result.reshape(out_shape)


def par_hyp2f1(a, b, c, z, P=8):  # noqa: N803
    """P-thread-per-element parallel Kummer kernel.

    Splits the m-iteration recurrence across P threads using warp shuffles
    and tree reduction.  P must be 1, 2, 4, 8, 16, or 32.
    Default P=8 (best balance of parallelism and overhead for N~128K).
    """
    p = _prepare_args(a, b, c, z)
    a_val, m, N, out_shape, device, b, c, z = p[:8]
    ndim, shape_t, b_st, c_st, z_st = p[8:]
    if m == 0:
        return torch.ones(out_shape, dtype=torch.float64, device=device)
    module = _get_module()
    result = module.par_hyp2f1_forward(
        a_val, b, c, z, ndim, shape_t, b_st, c_st, z_st, N, P)
    return result.reshape(out_shape)


def par_hyp2f1_precomp(a, b, c, z, P=8):  # noqa: N803
    """par_hyp2f1 with precomputed prefactor (lgamma+exp done outside kernel).

    Same algorithm as par_hyp2f1, but the Kummer prefactor
    Gamma(c)*Gamma(c-b+m) / (Gamma(c+m)*Gamma(c-b)) is computed via
    vectorised PyTorch lgamma+exp before the kernel launch, removing
    the in-kernel lgamma bottleneck on lane 0.
    """
    p = _prepare_args(a, b, c, z)
    a_val, m, N, out_shape, device, b, c, z = p[:8]
    ndim, shape_t, b_st, c_st, z_st = p[8:]
    if m == 0:
        return torch.ones(out_shape, dtype=torch.float64, device=device)

    m_f = float(m)
    # Prefactor: PyTorch handles strides internally, result is a new contiguous tensor
    prefactor = torch.exp(
        torch.lgamma(c) + torch.lgamma(c - b + m_f)
        - torch.lgamma(c + m_f) - torch.lgamma(c - b)
    )
    pref_flat = prefactor.view(-1)

    module = _get_module()
    result = module.par_hyp2f1_precomp_forward(
        a_val, b, c, z, pref_flat,
        ndim, shape_t, b_st, c_st, z_st, N, P)
    return result.reshape(out_shape)
