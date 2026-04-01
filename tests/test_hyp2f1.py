"""Tests for pgcuts.hyp2f1 triton kernel and autograd wrapper against SciPy."""
import torch
import numpy as np
from scipy.special import hyp2f1 as scipy_hyp2f1
from pgcuts.hyp2f1.autograd import Hyp2F1
from pgcuts.hyp2f1.triton_kernels import triton_hyp2f1

DEVICE = "cuda:0"


def _check(a, b, c, z, rtol=1e-5, atol=1e-8):
    """Run triton kernel and Hyp2F1.apply, compare both to SciPy.

    Returns (ok, max_abs_err, max_rel_err) for triton kernel.
    """
    out_tri = triton_hyp2f1(a, b, c, z)
    out_auto = Hyp2F1.apply(a, b, c, z)

    # SciPy reference
    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().to(torch.float64).numpy()
        return np.array(x, dtype=np.float64)

    a_np, b_np, c_np, z_np = _np(a), _np(b), _np(c), _np(z)
    a_np, b_np, c_np, z_np = np.broadcast_arrays(a_np, b_np, c_np, z_np)
    ref = scipy_hyp2f1(a_np, b_np, c_np, z_np)

    gpu = out_tri.detach().cpu().to(torch.float64).numpy()
    abs_err = np.abs(gpu - ref)
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-30)

    ok = np.allclose(gpu, ref, rtol=rtol, atol=atol)

    # Also check autograd wrapper matches triton
    auto_gpu = out_auto.detach().cpu().to(torch.float64).numpy()
    auto_match = np.allclose(gpu, auto_gpu, rtol=1e-12, atol=1e-14)

    return ok, abs_err.max(), rel_err.max(), auto_match


def _report(name, ok, abs_e, rel_e, auto_ok):
    status = "PASS" if ok else "FAIL"
    auto_s = "" if auto_ok else " [AUTOGRAD MISMATCH]"
    print(f"  [{status}] {name:<40s} max_abs={abs_e:.2e}  max_rel={rel_e:.2e}{auto_s}")
    return ok


# ── Edge cases ────────────────────────────────────────────────────────

def test_z_zero():
    """z=0 → 2F1=1 for any a,b,c."""
    z = torch.zeros(10, dtype=torch.float32, device=DEVICE)
    out = triton_hyp2f1(-20, 1.0, 2.0, z)
    ok = torch.allclose(out, torch.ones_like(out))
    print(f"  [{'PASS' if ok else 'FAIL'}] z=0 → 1.0")
    assert ok


def test_z_near_one():
    """z≈1 → should match SciPy prefactor."""
    z = torch.tensor([1.0 - 1e-10, 1.0 - 1e-7, 0.999], device=DEVICE)
    ok, abs_e, rel_e, auto_ok = _check(-512, 1.0, 2.0, z, rtol=1e-4)
    _report("z near 1, m=512", ok, abs_e, rel_e, auto_ok)
    assert ok


def test_a_zero():
    """a=0 → 2F1=1."""
    z = torch.rand(50, device=DEVICE) * 0.9
    out = triton_hyp2f1(0, 1.5, 3.0, z)
    ok = torch.allclose(out, torch.ones_like(out))
    print(f"  [{'PASS' if ok else 'FAIL'}] a=0 → 1.0")
    assert ok


def test_a_minus_one():
    """a=-1 → 2F1(-1,b;c;z) = 1 - bz/c."""
    z = torch.rand(100, device=DEVICE) * 0.9
    b, c = 1.5, 3.0
    out = triton_hyp2f1(-1, b, c, z)
    expected = 1.0 - b * z / c
    ok = torch.allclose(out, expected, rtol=1e-5)
    err = (out - expected).abs().max().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] a=-1 → 1-bz/c  max_err={err:.2e}")
    assert ok


# ── Scalar params, sweep m ────────────────────────────────────────────

def test_scalar_params():
    """Scalar a,b,c with 1D z, sweep m values."""
    z = torch.rand(500, device=DEVICE) * 0.95 + 0.01
    all_ok = True
    for m in [1, 5, 10, 50, 100, 512]:
        ok, abs_e, rel_e, auto_ok = _check(-m, 1.0, 2.0, z, rtol=1e-4)
        passed = _report(f"m={m}, b=1, c=2", ok, abs_e, rel_e, auto_ok)
        all_ok = all_ok and passed
    assert all_ok


def test_various_bc():
    """Different b,c values. rtol=1e-3 for float32 with large m."""
    z = torch.rand(200, device=DEVICE) * 0.9 + 0.05
    all_ok = True
    for b, c in [(1.0, 2.0), (0.5, 1.5), (2.0, 5.0), (1.0, 10.0)]:
        ok, abs_e, rel_e, auto_ok = _check(-512, b, c, z, rtol=1e-3)
        passed = _report(f"m=512, b={b}, c={c}", ok, abs_e, rel_e, auto_ok)
        all_ok = all_ok and passed
    assert all_ok


# ── Broadcasting ───────────────────────────────────────────────────────

def test_broadcast_2d():
    """2D broadcast: c (E,1) with z (1,K)."""
    c = torch.rand(50, 1, device=DEVICE) * 5 + 1.5
    z = torch.rand(1, 20, device=DEVICE) * 0.9 + 0.05
    b = torch.ones(1, device=DEVICE)
    ok, abs_e, rel_e, auto_ok = _check(-512, b, c, z, rtol=1e-3)
    _report("broadcast (50,1)*(1,20) m=512", ok, abs_e, rel_e, auto_ok)
    assert ok


def test_broadcast_scalar_bc():
    """Scalar b,c with 2D z."""
    z = torch.rand(10, 20, device=DEVICE) * 0.9 + 0.05
    ok, abs_e, rel_e, auto_ok = _check(-64, 1.0, 2.0, z, rtol=1e-4)
    _report("scalar b,c with 2D z", ok, abs_e, rel_e, auto_ok)
    assert ok


def test_broadcast_ncut_style():
    """NCut-style: c varies per (edge, bin), z varies per (bin, cluster).

    c shape (E, d, 1), z shape (1, d, K) → would be 3D.
    Since we cap at 2D, caller must reshape. Test the reshaped version.
    """
    E, d, K = 32, 4, 10
    c_flat = torch.rand(E * d, 1, device=DEVICE) * 5 + 1.5   # (E*d, 1)
    z_flat = torch.rand(1, K, device=DEVICE) * 0.9 + 0.05     # (1, K) tiled

    # Tile z for each bin
    z_tiled = z_flat.expand(E * d, K)  # (E*d, K)
    ok, abs_e, rel_e, auto_ok = _check(-512, 1.0, c_flat, z_tiled, rtol=1e-3)
    _report(f"NCut-style (E*d={E*d}, K={K})", ok, abs_e, rel_e, auto_ok)
    assert ok


# ── Dtype ──────────────────────────────────────────────────────────────

def test_dtype_preserved():
    """Output dtype matches z dtype."""
    for dt in [torch.float32, torch.float64]:
        z = torch.rand(50, dtype=dt, device=DEVICE) * 0.9
        out = triton_hyp2f1(-10, 1.0, 2.0, z)
        assert out.dtype == dt, f"expected {dt}, got {out.dtype}"

        out_auto = Hyp2F1.apply(-10, 1.0, 2.0, z)
        assert out_auto.dtype == dt, f"autograd: expected {dt}, got {out_auto.dtype}"
    print("  [PASS] dtype preserved (float32 and float64)")


def test_fp32_vs_fp64_relative():
    """fp32 should be within ~1e-5 relative error of fp64."""
    z64 = torch.rand(500, dtype=torch.float64, device=DEVICE) * 0.9 + 0.05
    z32 = z64.float()

    for m in [64, 512]:
        out64 = triton_hyp2f1(-m, 1.0, 2.0, z64)
        out32 = triton_hyp2f1(-m, 1.0, 2.0, z32)

        rel_err = ((out32.double() - out64).abs() / out64.abs().clamp(min=1e-30)).max().item()
        ok = rel_err < 1e-4
        print(f"  [{'PASS' if ok else 'FAIL'}] fp32 vs fp64 m={m}: max_rel={rel_err:.2e}")
        assert ok, f"fp32 too imprecise: rel_err={rel_err}"


# ── Autograd ───────────────────────────────────────────────────────────

def test_gradcheck_1d():
    """gradcheck for 1D z with various m."""
    for m in [0, 1, 5, 10, 50]:
        z = (torch.rand(20, dtype=torch.float64, device=DEVICE) * 0.8 + 0.1).requires_grad_(True)
        ok = torch.autograd.gradcheck(
            lambda z: Hyp2F1.apply(-m, 1.5, 3.0, z), (z,), fast_mode=True
        )
        print(f"  [PASS] gradcheck m={m}")
        assert ok


def test_gradcheck_2d_broadcast():
    """gradcheck for 2D z with broadcast c."""
    c = torch.rand(1, 4, dtype=torch.float64, device=DEVICE) * 3 + 1.5
    z = (torch.rand(8, 1, dtype=torch.float64, device=DEVICE) * 0.8 + 0.1).requires_grad_(True)
    ok = torch.autograd.gradcheck(
        lambda z: Hyp2F1.apply(-10, 1.0, c, z), (z,), fast_mode=True
    )
    print("  [PASS] gradcheck 2D broadcast (8,1)*(1,4)")
    assert ok


def test_gradient_value():
    """Verify gradient matches the derivative identity: d/dz = (ab/c) * 2F1(a+1,b+1;c+1;z)."""
    m = 50
    z = (torch.rand(100, dtype=torch.float64, device=DEVICE) * 0.8 + 0.1).requires_grad_(True)
    out = Hyp2F1.apply(-m, 1.5, 3.0, z)
    out.sum().backward()
    analytical = z.grad.clone()

    # Manual: (-m * 1.5 / 3.0) * 2F1(-m+1, 2.5; 4.0; z)
    with torch.no_grad():
        expected = (-m * 1.5 / 3.0) * triton_hyp2f1(-m + 1, 2.5, 4.0, z)

    rel_err = ((analytical - expected).abs() / expected.abs().clamp(min=1e-30)).max().item()
    ok = rel_err < 1e-10
    print(f"  [{'PASS' if ok else 'FAIL'}] gradient = (ab/c)*2F1(a+1,b+1;c+1;z)  max_rel={rel_err:.2e}")
    assert ok


# ── Manual series reference ───────────────────────────────────────────

def test_manual_series():
    """Compare kernel output against manual Horner-style series loop (from notebook)."""
    m = 16
    b = 1.0
    c = torch.tensor(1.5, dtype=torch.float64, device=DEVICE)
    z = (torch.rand(4, 10, dtype=torch.float64, device=DEVICE) * 0.9 + 0.05).requires_grad_(True)

    # Manual series: sum_{k=0}^m (a)_k (b)_k / (c)_k * z^k / k!
    a = -m
    coef = torch.ones_like(z)
    h2f1_ref = torch.ones_like(z)
    for k in range(1, m + 2):
        coef = coef * (a + k - 1) * (b + k - 1) * z / (c + k - 1) / k
        h2f1_ref = h2f1_ref + coef

    # Kernel output
    h2f1_kernel = Hyp2F1.apply(-m, b, c, z)
    abs_err = (h2f1_kernel - h2f1_ref).abs().max().item()
    ok = abs_err < 1e-12
    print(f"  [{'PASS' if ok else 'FAIL'}] manual series vs kernel  max_abs={abs_err:.2e}")
    assert ok, f"manual series mismatch: {abs_err}"

    # Compare gradients
    h2f1_ref.sum().backward()
    g_ref = z.grad.clone()
    z.grad.zero_()
    h2f1_kernel.sum().backward()
    g_kernel = z.grad.clone()
    grad_err = (g_ref - g_kernel).abs().max().item()
    grad_ok = grad_err < 1e-10
    print(f"  [{'PASS' if grad_ok else 'FAIL'}] manual series grad  max_abs={grad_err:.2e}")
    assert grad_ok, f"gradient mismatch: {grad_err}"


def test_large_m():
    """Test with large m values (m=1024, 4096) — stress test for numerical stability."""
    z = (torch.rand(100, dtype=torch.float64, device=DEVICE) * 0.9 + 0.05)
    for m in [1024, 4096]:
        ok, abs_e, rel_e, auto_ok = _check(-m, 1.0, 2.0, z, rtol=1e-3)
        _report(f"large m={m}", ok, abs_e, rel_e, auto_ok)
        assert ok


def test_broadcast_c_per_row():
    """c varies per row (like NCut bins), z varies per column (like clusters)."""
    c = torch.rand(16, 1, dtype=torch.float64, device=DEVICE) * 4 + 1.5
    z = (torch.rand(1, 10, dtype=torch.float64, device=DEVICE) * 0.9 + 0.05).requires_grad_(True)
    ok = torch.autograd.gradcheck(
        lambda z: Hyp2F1.apply(-64, 1.0, c, z), (z,), fast_mode=True
    )
    print(f"  [PASS] gradcheck c per-row (16,1) z (1,10)")
    assert ok


# ── 3D rejected ────────────────────────────────────────────────────────

def test_3d_raises():
    """3D input should raise ValueError."""
    z = torch.rand(2, 3, 4, device=DEVICE)
    try:
        triton_hyp2f1(-10, 1.0, 2.0, z)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"  [PASS] 3D raises: {e}")


if __name__ == "__main__":
    print("=== Edge cases ===")
    test_z_zero()
    test_z_near_one()
    test_a_zero()
    test_a_minus_one()

    print("\n=== Scalar params ===")
    test_scalar_params()
    test_various_bc()

    print("\n=== Broadcasting ===")
    test_broadcast_2d()
    test_broadcast_scalar_bc()
    test_broadcast_ncut_style()

    print("\n=== Dtype ===")
    test_dtype_preserved()
    test_fp32_vs_fp64_relative()

    print("\n=== Autograd ===")
    test_gradcheck_1d()
    test_gradcheck_2d_broadcast()
    test_gradient_value()

    print("\n=== Manual series ===")
    test_manual_series()
    test_large_m()
    test_broadcast_c_per_row()

    print("\n=== Constraints ===")
    test_3d_raises()

    print("\n" + "=" * 50)
    print("All tests passed!")
