"""Benchmark FlashCut vs current autograd-through-₂F₁ approach."""
import time, torch, torch.nn.functional as F
from pgcuts.hyp2f1.autograd import Hyp2F1
from pgcuts.losses.flashcut import flashcut_rcut

_h = Hyp2F1.apply


def bench_current(P_left, log_P_right, w, alpha_ema, m, steps=200):
    """Current approach: autograd through ₂F₁ with detached alpha."""
    P_left = P_left.detach().requires_grad_(True)
    log_P_right = log_P_right.detach().requires_grad_(True)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        z = alpha_ema.clamp(1e-7, 1 - 1e-7)
        H = _h(-m, 1.0, 2.0, z)
        C = (w.unsqueeze(-1) * (-P_left * log_P_right)).mean(0)
        loss = (C * H.to(P_left.dtype)).sum() / w.sum()
        loss.backward()
        P_left.grad = None
        log_P_right.grad = None
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


def bench_current_no_detach(P_left, log_P_right, w, m, steps=200):
    """Current approach but alpha NOT detached — autograd through ₂F₁ backward."""
    P_left = P_left.detach().requires_grad_(True)
    log_P_right = log_P_right.detach().requires_grad_(True)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        alpha = P_left.mean(0)  # NOT detached
        z = alpha.clamp(1e-7, 1 - 1e-7)
        H = _h(-m, 1.0, 2.0, z)
        C = (w.unsqueeze(-1) * (-P_left * log_P_right)).mean(0)
        loss = (C * H.to(P_left.dtype)).sum() / w.sum()
        loss.backward()
        P_left.grad = None
        log_P_right.grad = None
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


def bench_flash(P_left, log_P_right, w, alpha_ema, m, steps=200):
    """FlashCut: custom autograd.Function with manual backward."""
    P_left = P_left.detach().requires_grad_(True)
    log_P_right = log_P_right.detach().requires_grad_(True)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        cut_per_k = flashcut_rcut(P_left, log_P_right, w, alpha_ema, m)
        loss = cut_per_k.sum() / w.sum()
        loss.backward()
        P_left.grad = None
        log_P_right.grad = None
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


def bench_reference(P_left, log_P_right, w, m, steps=200):
    """Reference: full autograd through H(alpha) with alpha = P_left.mean(0)."""
    P_left = P_left.detach().requires_grad_(True)
    log_P_right = log_P_right.detach().requires_grad_(True)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        alpha = P_left.mean(0)
        z = alpha.clamp(1e-7, 1 - 1e-7)
        H = _h(-m, 1.0, 2.0, z)
        C = (w.unsqueeze(-1) * (-P_left * log_P_right)).mean(0)
        loss = (C * H.to(P_left.dtype)).sum() / w.sum()
        loss.backward()
        P_left.grad = None
        log_P_right.grad = None
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


if __name__ == '__main__':
    device = 'cuda:0'
    m = 512

    print(f'{"E":>8} {"K":>6} {"detached":>12} {"no_detach":>12} {"flash":>12} {"flash/detach":>12} {"flash/nodet":>12}')
    print('-' * 80)

    for E, K in [(1024, 10), (8192, 10), (8192, 100), (16384, 100), (8192, 200)]:
        torch.manual_seed(0)
        logits_l = torch.randn(E, K, device=device)
        logits_r = torch.randn(E, K, device=device)
        P_left = torch.softmax(logits_l, dim=-1)
        log_P_right = F.log_softmax(logits_r, dim=-1)
        w = torch.rand(E, device=device).clamp(min=0.1)
        alpha_ema = torch.ones(K, device=device) / K

        # Warmup
        bench_flash(P_left, log_P_right, w, alpha_ema, m, steps=5)
        bench_current(P_left, log_P_right, w, alpha_ema, m, steps=5)
        bench_current_no_detach(P_left, log_P_right, w, m, steps=5)

        t_detach = bench_current(P_left, log_P_right, w, alpha_ema, m)
        t_nodetach = bench_current_no_detach(P_left, log_P_right, w, m)
        t_flash = bench_flash(P_left, log_P_right, w, alpha_ema, m)

        print(f'{E:>8} {K:>6} {t_detach*1000:>10.2f}ms {t_nodetach*1000:>10.2f}ms {t_flash*1000:>10.2f}ms {t_flash/t_detach:>11.2f}x {t_nodetach/t_flash:>11.2f}x')
