"""Tests for FlashCutRCut: verify gradients match full autograd reference.

The key test: FlashCutRCut (manual backward with two terms) must produce
the same gradients as a reference that lets autograd differentiate through
H(alpha) where alpha = P_left.mean(0) (not detached).
"""
import torch
import torch.nn.functional as F
from pgcuts.hyp2f1.autograd import Hyp2F1
from pgcuts.losses.flashcut import FlashCutRCut, flashcut_rcut

_h = Hyp2F1.apply


def reference_rcut_loss(P_left, log_P_right, w, m):
    """Full autograd reference: alpha is computed from P_left, NOT detached.

    This lets autograd differentiate through both the cut term AND the
    ₂F₁ envelope, producing the exact gradient we want FlashCut to match.
    """
    alpha = P_left.mean(0)  # (K,) — NOT detached!
    z = alpha.clamp(1e-7, 1 - 1e-7)
    H = _h(-m, 1.0, 2.0, z)  # (K,) — autograd tracks gradient through z→alpha→P_left
    C = (w.unsqueeze(-1) * (-P_left * log_P_right)).mean(0)  # (K,)
    return (C * H).sum()  # scalar


def make_test_data(E=64, K=5, m=32, device='cuda:0', dtype=torch.float64):
    """Create random test data."""
    torch.manual_seed(42)

    # Random logits → softmax/log_softmax for valid probabilities
    logits_left = torch.randn(E, K, dtype=dtype, device=device)
    logits_right = torch.randn(E, K, dtype=dtype, device=device)

    P_left = torch.softmax(logits_left, dim=-1).requires_grad_(True)
    log_P_right = F.log_softmax(logits_right, dim=-1).requires_grad_(True)

    w = torch.rand(E, dtype=dtype, device=device).clamp(min=0.1)

    return P_left, log_P_right, w, m


def test_forward_matches():
    """FlashCut forward values match the reference computation."""
    P_left, log_P_right, w, m = make_test_data()
    alpha_ema = P_left.detach().mean(0)

    # FlashCut forward
    flash_cut = flashcut_rcut(P_left, log_P_right, w, alpha_ema, m)

    # Reference forward (same computation, just inline)
    z = alpha_ema.clamp(1e-7, 1 - 1e-7)
    with torch.no_grad():
        H = _h(-m, 1.0, 2.0, z).to(P_left.dtype)
    C = (w.unsqueeze(-1) * (-P_left * log_P_right)).mean(0)
    ref_cut = C * H

    err = (flash_cut - ref_cut).abs().max().item()
    print(f'Forward match: max_err={err:.2e}')
    assert err < 1e-10, f'Forward mismatch: {err}'
    print('  PASSED')


def test_gradcheck():
    """torch.autograd.gradcheck on the full-autograd reference.

    gradcheck verifies that the *reference* function (where alpha = P_left.mean(0)
    is in the graph) has consistent forward/backward. This confirms the
    mathematical formulation is correct. FlashCutRCut intentionally decouples
    alpha_ema from P_left (like PRCut's ov_P), so its gradcheck would fail
    by design — the reference test below verifies it matches instead.
    """
    P_left, log_P_right, w, m = make_test_data(E=16, K=4, m=8)

    ok = torch.autograd.gradcheck(
        lambda pl, lpr: reference_rcut_loss(pl, lpr, w, m),
        (P_left, log_P_right),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    print(f'gradcheck (reference): {"PASSED" if ok else "FAILED"}')
    assert ok


def test_grads_match_autograd_reference():
    """The critical test: FlashCut grads match full autograd through H(alpha).

    Reference: alpha = P_left.mean(0) (not detached), autograd through ₂F₁.
    FlashCut: alpha_ema = P_left.detach().mean(0), manual Term A + Term B.

    Gradients w.r.t. P_left should match (both Term A and Term B).
    Gradients w.r.t. log_P_right should match (Term A only, since alpha
    doesn't depend on log_P_right).
    """
    for E, K, m in [(32, 5, 16), (64, 10, 32), (128, 3, 64)]:
        P_left, log_P_right, w, m = make_test_data(E=E, K=K, m=m)

        # ── Reference: full autograd ──────────────────────────
        P_left_ref = P_left.detach().clone().requires_grad_(True)
        log_P_right_ref = log_P_right.detach().clone().requires_grad_(True)

        loss_ref = reference_rcut_loss(P_left_ref, log_P_right_ref, w, m)
        loss_ref.backward()
        grad_P_ref = P_left_ref.grad.clone()
        grad_lP_ref = log_P_right_ref.grad.clone()

        # ── FlashCut: manual backward ─────────────────────────
        P_left_flash = P_left.detach().clone().requires_grad_(True)
        log_P_right_flash = log_P_right.detach().clone().requires_grad_(True)

        # Use P_left's current mean as alpha (same as reference)
        alpha_ema = P_left_flash.detach().mean(0)
        cut_per_k = FlashCutRCut.apply(
            P_left_flash, log_P_right_flash, w, alpha_ema, m
        )
        loss_flash = cut_per_k.sum()
        loss_flash.backward()
        grad_P_flash = P_left_flash.grad.clone()
        grad_lP_flash = log_P_right_flash.grad.clone()

        # ── Compare ───────────────────────────────────────────
        # Forward values
        fwd_err = abs(loss_ref.item() - loss_flash.item())

        # Gradients w.r.t. P_left (should match: Term A + Term B)
        err_P = (grad_P_ref - grad_P_flash).abs()
        rel_P = err_P / (grad_P_ref.abs().clamp(min=1e-12))

        # Gradients w.r.t. log_P_right (should match: Term A only)
        err_lP = (grad_lP_ref - grad_lP_flash).abs()
        rel_lP = err_lP / (grad_lP_ref.abs().clamp(min=1e-12))

        print(f'\nE={E}, K={K}, m={m}:')
        print(f'  Forward: ref={loss_ref.item():.6f} flash={loss_flash.item():.6f} err={fwd_err:.2e}')
        print(f'  grad_P_left:      max_abs_err={err_P.max():.2e}  max_rel_err={rel_P.max():.2e}')
        print(f'  grad_log_P_right: max_abs_err={err_lP.max():.2e}  max_rel_err={rel_lP.max():.2e}')

        assert fwd_err < 1e-8, f'Forward mismatch: {fwd_err}'
        assert err_P.max() < 1e-5, f'grad_P_left mismatch: {err_P.max()}'
        assert err_lP.max() < 1e-5, f'grad_log_P_right mismatch: {err_lP.max()}'
        print('  PASSED')


def test_term_b_is_nonzero():
    """Verify Term B (envelope gradient) contributes meaningfully."""
    P_left, log_P_right, w, m = make_test_data(E=64, K=5, m=32)
    alpha_ema = P_left.detach().mean(0)

    # FlashCut with Term B
    P1 = P_left.detach().clone().requires_grad_(True)
    lP1 = log_P_right.detach().clone().requires_grad_(True)
    loss1 = FlashCutRCut.apply(P1, lP1, w, alpha_ema, m).sum()
    loss1.backward()
    grad_with_B = P1.grad.clone()

    # Term A only (what current autograd-with-detached-alpha gives)
    P2 = P_left.detach().clone().requires_grad_(True)
    lP2 = log_P_right.detach().clone().requires_grad_(True)
    z = alpha_ema.clamp(1e-7, 1 - 1e-7)
    with torch.no_grad():
        H = _h(-m, 1.0, 2.0, z).to(P2.dtype)
    C = (w.unsqueeze(-1) * (-P2 * lP2)).mean(0)
    loss2 = (C * H).sum()
    loss2.backward()
    grad_without_B = P2.grad.clone()

    # Term B contribution
    diff = (grad_with_B - grad_without_B).abs()
    term_b_norm = diff.norm().item()
    term_a_norm = grad_without_B.norm().item()
    ratio = term_b_norm / (term_a_norm + 1e-12)

    print(f'\nTerm B analysis:')
    print(f'  |Term A| = {term_a_norm:.6f}')
    print(f'  |Term B| = {term_b_norm:.6f}')
    print(f'  |B|/|A| = {ratio:.4f}')
    assert term_b_norm > 1e-10, f'Term B is zero: {term_b_norm}'
    print('  PASSED (Term B is nonzero)')


if __name__ == '__main__':
    test_forward_matches()
    test_gradcheck()
    test_grads_match_autograd_reference()
    test_term_b_is_nonzero()
    print('\n' + '='*50)
    print('All tests passed!')
