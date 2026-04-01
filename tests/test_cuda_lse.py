import torch
import pytest
from attempts.cuda_lse import hyper2f1_negint


def cpu_reference(z, a_int, b, c):
    assert a_int <= 0
    m = -int(a_int)
    z = z.to(torch.float64)
    k = torch.arange(m + 1, dtype=torch.float64)
    lg_m1 = torch.lgamma(torch.tensor(m + 1.0))
    log_binom = (
        lg_m1
        - torch.lgamma(k + 1)
        - torch.lgamma(torch.tensor(m + 0.0) - k + 1)
    )
    cb = torch.tensor(c - b, dtype=torch.float64)
    c_t = torch.tensor(c, dtype=torch.float64)
    log_r = (
        torch.lgamma(cb + k)
        - torch.lgamma(cb)
        + torch.lgamma(c_t)
        - torch.lgamma(c_t + k)
    )
    logz = torch.log(z)
    logq = torch.log1p(-z)
    T = []
    for kk in range(m + 1):
        lw = 0.0
        if kk > 0:
            lw += kk * logz
        if kk < m:
            lw += (m - kk) * logq
        T.append(log_binom[kk] + log_r[kk] + lw)
    T = torch.stack(T, dim=-1)  # (K, m+1) after broadcasting
    max_log = T.max(dim=-1, keepdim=True).values
    return torch.exp(max_log) * torch.exp(T - max_log).sum(
        dim=-1, keepdim=True
    ).squeeze(-1)


@pytest.mark.parametrize("m,b,c", [(20, 2.5, 5.0), (50, 1.75, 4.2)])
def test_cuda_matches_cpu(m, b, c):
    from scipy.special import hyp2f1
    a_int = -m
    K = 257
    z = torch.linspace(0, 1, K, dtype=torch.float64, device="cuda")
    y_cuda = hyper2f1_negint(z, a_int, b, c).double().cpu()
    # y_cpu = cpu_reference(z.cpu(), a_int, b, c)
    y_cpu = hyp2f1(a_int, b,c, z.cpu())
    rel_err = (y_cuda - y_cpu).abs() / y_cpu.clamp_min(1e-30)
    # print((y_cuda - y_cpu).abs())
    # print(y_cuda)
    # print(y_cpu)
    assert rel_err.max().item() < 1e-12
