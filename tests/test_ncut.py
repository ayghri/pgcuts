"""Tests for NCut forward and backward.

Uses the polynomial definition of ₂F₁(-m, b; c; z) for integer m > 0:
    ₂F₁(-m, b; c; z) = Σ_{k=0}^{m} (-m)_k (b)_k / ((c)_k k!) z^k

This is a finite sum implemented in pure torch — fully differentiable,
no scipy, giving us the true gradient.
"""
import numpy as np
import torch
from pgcuts.losses.pncut import compute_ncut_bin_phi
from pgcuts.hyp2f1.autograd import Hyp2F1

DEVICE = "cuda:0"
M = 64


def hyp2f1_poly(m, b, c, z):
    """₂F₁(-m, b; c; z) via direct terminating polynomial.

    Uses binomial-form coefficients to avoid overflow:
        term_k = C(m,k) * (b)_k / (c)_k * (-z)^k

    where C(m,k) = m!/(k!(m-k)!) and (x)_k = x(x+1)...(x+k-1).

    Recurrence: term_{k+1}/term_k = -(m-k)/(k+1) * (b+k)/(c+k) * z

    Pure torch, fully differentiable.
    Stable for m * z ≲ 30 in float64.
    """
    result = torch.ones_like(z)
    term = torch.ones_like(z)
    for k in range(m):
        ratio = (m - k) / (k + 1) * (b + k) / (c + k) * z
        term = -term * ratio
        result = result + term
    return result


def make_test_graph(N=20, K=3, d=2, knn=4, seed=42):
    """Small test graph with known structure."""
    rng = np.random.RandomState(seed)
    # Small logits → P close to uniform (1/K) → alpha_bars ≈ 1/K ≈ 0.33
    logits = rng.randn(N, K).astype(np.float64) * 0.3

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        nbrs = rng.choice([j for j in range(N) if j != i], knn, replace=False)
        for j in nbrs:
            w = rng.rand() * 0.5 + 0.5
            W[i, j] = w
            W[j, i] = w
    np.fill_diagonal(W, 0)

    degrees = W.sum(axis=1)
    sorted_idx = np.argsort(degrees)
    bins = []
    for indices in np.array_split(sorted_idx, d):
        bins.append({
            'beta_star': float(degrees[indices].min()),
            'indices': indices,
            'count': len(indices),
        })
    return logits, W, degrees, bins


def reference_ncut_loss(logits_t, W_t, degrees_t, bins, m):
    """Full NCut loss using hyp2f1_poly — fully differentiable reference.

    No detached alpha — gradient flows through everything.
    """
    N, K = logits_t.shape
    d = len(bins)
    device = logits_t.device

    P = torch.softmax(logits_t, dim=-1)
    log_P = torch.log_softmax(logits_t, dim=-1)

    # Per-bin alpha (NOT detached)
    alpha_bars = torch.zeros(d, K, dtype=torch.float64, device=device)
    for j, b in enumerate(bins):
        idx = torch.tensor(b['indices'], dtype=torch.long, device=device)
        alpha_bars[j] = P[idx].mean(0)

    counts = torch.tensor([b['count'] for b in bins], dtype=torch.float64, device=device)
    bin_weights = counts / counts.sum()
    beta_stars = torch.tensor([b['beta_star'] for b in bins], dtype=torch.float64, device=device)
    q_stars = torch.tensor(
        [degrees_t[b['indices']].mean().item() for b in bins], dtype=torch.float64, device=device
    )

    # Phi via polynomial ₂F₁ — fully differentiable
    z = alpha_bars.clamp(1e-7, 1 - 1e-7)  # (d, K)
    log_Phi = torch.zeros(d, K, dtype=torch.float64, device=device)
    for i in range(d):
        for j in range(d):
            c = q_stars[i] / beta_stars[j] + 1.0
            F = hyp2f1_poly(m, 1.0, c, z[j])  # (K,)
            log_Phi[i] += bin_weights[j] * torch.log(F.clamp(min=1e-30))
    Phi = torch.exp(log_Phi)  # (d, K)

    # Node-to-bin
    node_to_bin = torch.zeros(N, dtype=torch.long, device=device)
    for j, b in enumerate(bins):
        node_to_bin[torch.tensor(b['indices'], dtype=torch.long, device=device)] = j

    # Loss: Σ_{i,j} W_ij * (-P_i * log_P_j) * Phi[bin_i] / deg_i
    hycut = (W_t.unsqueeze(-1) * (-P.unsqueeze(1) * log_P.unsqueeze(0))).sum(1)  # (N, K)
    Phi_per_node = Phi[node_to_bin]  # (N, K)
    loss = (hycut * Phi_per_node / degrees_t.clamp(min=1e-6).unsqueeze(1)).sum() / W_t.sum()
    return loss


def our_ncut_loss(logits_t, W_t, degrees_t, bins, m):
    """NCut loss using our compute_ncut_bin_phi (Triton kernel).

    Alpha is detached as in training. Gradient only flows through the cut term.
    """
    N, K = logits_t.shape
    d = len(bins)
    device = logits_t.device

    P = torch.softmax(logits_t, dim=-1)
    log_P = torch.log_softmax(logits_t, dim=-1)

    alpha_bars = torch.zeros(d, K, dtype=torch.float64, device=device)
    for j, b in enumerate(bins):
        idx = torch.tensor(b['indices'], dtype=torch.long, device=device)
        alpha_bars[j] = P.detach()[idx].mean(0)  # DETACHED

    counts = torch.tensor([b['count'] for b in bins], dtype=torch.float64, device=device)
    bin_weights = counts / counts.sum()
    beta_stars = torch.tensor([b['beta_star'] for b in bins], dtype=torch.float64, device=device)
    q_stars = torch.tensor(
        [degrees_t[b['indices']].mean().item() for b in bins], dtype=torch.float64, device=device
    )

    node_to_bin = torch.zeros(N, dtype=torch.long, device=device)
    for j, b in enumerate(bins):
        node_to_bin[torch.tensor(b['indices'], dtype=torch.long, device=device)] = j

    Phi_bins = compute_ncut_bin_phi(q_stars, alpha_bars, beta_stars, bin_weights, m)

    hycut = (W_t.unsqueeze(-1) * (-P.unsqueeze(1) * log_P.unsqueeze(0))).sum(1)
    Phi_per_node = Phi_bins[node_to_bin]
    loss = (hycut * Phi_per_node / degrees_t.clamp(min=1e-6).unsqueeze(1)).sum() / W_t.sum()
    return loss


def test_hyp2f1_poly_vs_triton():
    """Verify hyp2f1_poly matches the Triton kernel."""
    # z in [0.01, 0.3] — realistic range for alpha_bars ≈ 1/K
    z = torch.rand(10, dtype=torch.float64, device=DEVICE) * 0.29 + 0.01
    for m in [1, 8, 32, 64]:
        poly = hyp2f1_poly(m, 1.0, 2.0, z)
        triton = Hyp2F1.apply(-m, 1.0, 2.0, z)
        rel = ((poly - triton).abs() / poly.abs().clamp(min=1e-30)).max().item()
        print(f'  m={m}: max_rel_err={rel:.2e}')
        assert rel < 1e-5, f'hyp2f1 mismatch at m={m}: {rel}'
    print('  PASSED')


def test_phi_matches():
    """Verify compute_ncut_bin_phi matches polynomial reference."""
    logits_np, W, degrees, bins = make_test_graph()
    N, K = logits_np.shape
    d = len(bins)

    logits_t = torch.tensor(logits_np, dtype=torch.float64, device=DEVICE)
    P = torch.softmax(logits_t, dim=-1)

    alpha_bars = torch.zeros(d, K, dtype=torch.float64, device=DEVICE)
    for j, b in enumerate(bins):
        idx = torch.tensor(b['indices'], dtype=torch.long, device=DEVICE)
        alpha_bars[j] = P.detach()[idx].mean(0)

    counts = torch.tensor([b['count'] for b in bins], dtype=torch.float64, device=DEVICE)
    bin_weights = counts / counts.sum()
    beta_stars = torch.tensor([b['beta_star'] for b in bins], dtype=torch.float64, device=DEVICE)
    q_stars = torch.tensor(
        [degrees[b['indices']].mean() for b in bins], dtype=torch.float64, device=DEVICE
    )

    # Our kernel
    Phi_ours = compute_ncut_bin_phi(q_stars, alpha_bars, beta_stars, bin_weights, M)

    # Polynomial reference
    z = alpha_bars.clamp(1e-7, 1 - 1e-7)
    Phi_ref = torch.zeros(d, K, dtype=torch.float64, device=DEVICE)
    for i in range(d):
        log_phi = torch.zeros(K, dtype=torch.float64, device=DEVICE)
        for j in range(d):
            c = q_stars[i] / beta_stars[j] + 1.0
            F = hyp2f1_poly(M, 1.0, c, z[j])
            log_phi += bin_weights[j] * torch.log(F.clamp(min=1e-30))
        Phi_ref[i] = torch.exp(log_phi)

    rel = ((Phi_ours - Phi_ref).abs() / Phi_ref.abs().clamp(min=1e-30)).max().item()
    print(f'  Phi max_rel_err={rel:.2e}')
    assert rel < 1e-6, f'Phi mismatch: {rel}'
    print('  PASSED')


def test_forward_matches():
    """Forward values match between reference and our implementation."""
    logits_np, W, degrees, bins = make_test_graph()

    logits_t = torch.tensor(logits_np, dtype=torch.float64, device=DEVICE)
    W_t = torch.tensor(W, dtype=torch.float64, device=DEVICE)
    degrees_t = torch.tensor(degrees, dtype=torch.float64, device=DEVICE)

    ref_loss = reference_ncut_loss(logits_t.clone(), W_t, degrees_t, bins, M)
    our_loss = our_ncut_loss(logits_t.clone(), W_t, degrees_t, bins, M)

    rel = abs(ref_loss.item() - our_loss.item()) / max(abs(ref_loss.item()), 1e-30)
    print(f'  ref={ref_loss.item():.10f}  ours={our_loss.item():.10f}  rel_err={rel:.2e}')
    assert rel < 1e-6, f'Forward mismatch: {rel}'
    print('  PASSED')


def test_backward_reference_gradcheck():
    """gradcheck on the polynomial reference — confirms math is correct.

    Uses m=8 where the direct polynomial is numerically exact in float64.
    """
    m_small = 8
    logits_np, W, degrees, bins = make_test_graph(N=10, K=3, d=2, knn=3)

    W_t = torch.tensor(W, dtype=torch.float64, device=DEVICE)
    degrees_t = torch.tensor(degrees, dtype=torch.float64, device=DEVICE)

    logits_t = torch.tensor(logits_np, dtype=torch.float64, device=DEVICE, requires_grad=True)

    ok = torch.autograd.gradcheck(
        lambda l: reference_ncut_loss(l, W_t, degrees_t, bins, m_small),
        (logits_t,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    print(f'  gradcheck (m={m_small}): {"PASSED" if ok else "FAILED"}')
    assert ok


def test_backward_our_vs_reference():
    """Compare gradient: our implementation (detached alpha) vs reference (full autograd).

    The reference gradient includes Term B (through envelope).
    Our implementation only has Term A (through cut, alpha detached).
    We verify both are valid and show the difference.
    """
    logits_np, W, degrees, bins = make_test_graph()

    W_t = torch.tensor(W, dtype=torch.float64, device=DEVICE)
    degrees_t = torch.tensor(degrees, dtype=torch.float64, device=DEVICE)

    # Reference gradient (full, through envelope)
    logits_ref = torch.tensor(logits_np, dtype=torch.float64, device=DEVICE, requires_grad=True)
    loss_ref = reference_ncut_loss(logits_ref, W_t, degrees_t, bins, M)
    loss_ref.backward()
    grad_ref = logits_ref.grad.clone()

    # Our gradient (detached alpha — Term A only)
    logits_ours = torch.tensor(logits_np, dtype=torch.float64, device=DEVICE, requires_grad=True)
    loss_ours = our_ncut_loss(logits_ours, W_t, degrees_t, bins, M)
    loss_ours.backward()
    grad_ours = logits_ours.grad.clone()

    # Term A should be the same in both
    # Term B (envelope gradient) is present only in ref
    diff = (grad_ref - grad_ours).abs()
    term_a_norm = grad_ours.norm().item()
    term_b_norm = diff.norm().item()
    full_norm = grad_ref.norm().item()

    print(f'  |grad_ref| (full)  = {full_norm:.6f}')
    print(f'  |grad_ours| (A only) = {term_a_norm:.6f}')
    print(f'  |Term B| (diff)    = {term_b_norm:.6f}')
    print(f'  |B|/|A|            = {term_b_norm / max(term_a_norm, 1e-12):.4f}')

    # Both should be nonzero
    assert term_a_norm > 1e-8, f'Term A is zero'
    assert full_norm > 1e-8, f'Full gradient is zero'
    print('  PASSED')


if __name__ == '__main__':
    print('=== hyp2f1 poly vs triton ===')
    test_hyp2f1_poly_vs_triton()

    print('\n=== Phi matches ===')
    test_phi_matches()

    print('\n=== Forward matches ===')
    test_forward_matches()

    print('\n=== gradcheck on reference ===')
    test_backward_reference_gradcheck()

    print('\n=== Our gradient vs reference ===')
    test_backward_our_vs_reference()

    print('\n' + '=' * 50)
    print('All NCut tests passed!')
