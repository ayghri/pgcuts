"""Non-parametric PGCut: softmax logits + Adam + temp annealing + GradMixer.

What worked: softmax(logits) + Adam + full graph + GradientMixer → ncut=0.664
Now adding: temperature annealing T: 1→0.1 to sharpen assignments over time.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
import scipy.sparse as sp

from embedata import load_embeddings
from pgcuts.metrics import evaluate_clustering, compute_rcut_ncut
from pgcuts.graph import build_rbf_knn_graph
from pgcuts.optim import GradientMixer
from pgcuts.hyp2f1.autograd import Hyp2F1
from pgcuts.losses.pncut import equal_size_bins, log_kmeans_bins
from sklearn.cluster import SpectralClustering

_hyp2f1 = Hyp2F1.apply
REPR_DIR = Path("/buckets/representations/")


def remap_labels(y):
    unique = np.unique(y)
    lmap = {old: new for new, old in enumerate(unique)}
    return np.array([lmap[l] for l in y]), len(unique)


def sparse_to_torch(W_sp, device):
    W_coo = sp.coo_matrix(W_sp)
    indices = torch.tensor(np.vstack([W_coo.row, W_coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(W_coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, W_coo.shape).coalesce()


def run(
    X, y, K, W_exp, degrees, device="cuda",
    m=512, num_bins=4, binning="log_kmeans",
    lr=0.1, epochs=500, balance_weight=1.0,
    temp_start=1.0, temp_end=0.1,
):
    n = X.shape[0]
    device = torch.device(device)

    W_t = sparse_to_torch(W_exp, device)
    W_sum = W_exp.sum()

    bins = log_kmeans_bins(degrees, num_bins) if binning == "log_kmeans" else equal_size_bins(degrees, num_bins)
    d = len(bins)
    beta_stars = torch.tensor([b["beta_star"] for b in bins], dtype=torch.float32, device=device)
    counts = torch.tensor([b["count"] for b in bins], dtype=torch.float32, device=device)
    bin_weights = counts / counts.sum()
    degrees_t = torch.tensor(degrees, dtype=torch.float32, device=device).clamp(min=1e-6)
    bin_indices_list = [torch.tensor(b["indices"], dtype=torch.long, device=device) for b in bins]
    q_stars = torch.tensor([degrees[b["indices"]].mean() for b in bins], dtype=torch.float32, device=device)
    c_per_bin = q_stars / beta_stars + 1.0

    # Random logits init
    logits = torch.nn.Parameter(torch.randn(n, K, device=device))
    optimizer = optim.Adam([logits], lr=lr)
    grad_mix = GradientMixer([("logits", logits)], loss_scale={"ncut": 1.0, "balance": balance_weight})

    # EMA for per-bin means
    alpha_ema = torch.ones(d, K, device=device) / K

    t0 = time.time()
    for epoch in range(epochs):
        # Temperature annealing
        frac = epoch / max(epochs - 1, 1)
        temp = temp_start * (temp_end / temp_start) ** frac

        P = torch.softmax(logits / temp, dim=-1)

        # M = P · (W @ (1-P))
        M = P * torch.sparse.mm(W_t, 1.0 - P)

        # Per-bin means via EMA
        with torch.no_grad():
            for j, idx in enumerate(bin_indices_list):
                alpha_ema[j] = 0.9 * alpha_ema[j] + 0.1 * P[idx].mean(0)

        # Φ via binned ₂F₁
        z = alpha_ema.clamp(1e-7, 1 - 1e-7)
        c_dk = c_per_bin.unsqueeze(1).expand(d, K)
        F = _hyp2f1(-m, 1.0, c_dk, z)
        log_F = torch.log(F.clamp(min=1e-30))
        log_Phi_base = (bin_weights.unsqueeze(1) * log_F).sum(0)
        Phi = torch.exp(log_Phi_base).unsqueeze(0) / degrees_t.unsqueeze(1)

        U = (M * Phi).sum() / W_sum
        balance = -torch.special.entr(P.mean(0)).sum()

        optimizer.zero_grad()
        with grad_mix("ncut"):
            U.backward(retain_graph=True)
        with grad_mix("balance"):
            balance.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                pred = logits.argmax(dim=-1).cpu().numpy()
                res = evaluate_clustering(y, pred, K)
                rc, nc = compute_rcut_ncut(W_exp, pred)
                print(f"  ep {epoch:>3} T={temp:.3f}: ncut={nc:.4f} rcut={rc:.2f} acc={res['accuracy']:.4f} U={U.item():.6f}")

    elapsed = time.time() - t0
    with torch.no_grad():
        pred = logits.argmax(dim=-1).cpu().numpy()
        res = evaluate_clustering(y, pred, K)
    rc, nc = compute_rcut_ncut(W_exp, pred)
    return {"accuracy": float(res["accuracy"]), "nmi": float(res["nmi"]),
            "rcut": float(rc), "ncut": float(nc), "time_s": round(elapsed, 1)}


def main():
    train_ds = load_embeddings("cifar10", "clipvitL14", REPR_DIR, split="train")
    X, y = train_ds.feats, train_ds.labels
    y, K = remap_labels(y)
    n = X.shape[0]
    print(f"cifar10/clipvitL14: n={n}, K={K}")

    all_results = []

    for knn in [50]:
        W_exp = build_rbf_knn_graph(X, n_neighbors=min(knn, n - 1))
        degrees = np.array(W_exp.sum(axis=1)).flatten().astype(np.float32)

        # Spectral
        print("\nSpectral...")
        sc = SpectralClustering(n_clusters=K, affinity="precomputed", eigen_solver="amg", assign_labels="kmeans", n_jobs=-1)
        sc_pred = sc.fit_predict(W_exp)
        sc_res = evaluate_clustering(y, sc_pred, K)
        sc_rc, sc_nc = compute_rcut_ncut(W_exp, sc_pred)
        print(f"  Spectral: ncut={sc_nc:.4f} rcut={sc_rc:.2f} acc={sc_res['accuracy']:.4f}")
        all_results.append({"method": "Spectral", "knn": knn, "accuracy": float(sc_res["accuracy"]),
                           "nmi": float(sc_res["nmi"]), "rcut": float(sc_rc), "ncut": float(sc_nc)})

        configs = [
            # Sweep lr with temp annealing 1→0.1
            {"lr": 0.001, "epochs": 500, "temp_start": 1.0, "temp_end": 0.1, "balance_weight": 1.0},
            {"lr": 0.01, "epochs": 500, "temp_start": 1.0, "temp_end": 0.1, "balance_weight": 1.0},
            {"lr": 0.05, "epochs": 500, "temp_start": 1.0, "temp_end": 0.1, "balance_weight": 1.0},
            {"lr": 0.1, "epochs": 500, "temp_start": 1.0, "temp_end": 0.1, "balance_weight": 1.0},
            {"lr": 0.5, "epochs": 500, "temp_start": 1.0, "temp_end": 0.1, "balance_weight": 1.0},
            # No annealing (T=1 constant, like before)
            {"lr": 0.1, "epochs": 500, "temp_start": 1.0, "temp_end": 1.0, "balance_weight": 1.0},
            # Different temp ranges
            {"lr": 0.1, "epochs": 500, "temp_start": 2.0, "temp_end": 0.1, "balance_weight": 1.0},
            {"lr": 0.1, "epochs": 500, "temp_start": 1.0, "temp_end": 0.01, "balance_weight": 1.0},
            # Balance sweep
            {"lr": 0.1, "epochs": 500, "temp_start": 1.0, "temp_end": 0.1, "balance_weight": 0.5},
            {"lr": 0.1, "epochs": 500, "temp_start": 1.0, "temp_end": 0.1, "balance_weight": 2.0},
        ]

        for i, cfg in enumerate(configs):
            tag = f"lr={cfg['lr']} T={cfg['temp_start']}->{cfg['temp_end']} bal={cfg['balance_weight']}"
            print(f"\n[{i+1}/{len(configs)}] {tag}")
            r = run(X, y, K, W_exp, degrees, device="cuda", **cfg)
            r["method"] = "PGCut"
            r["config"] = tag
            all_results.append(r)
            print(f"  FINAL: ncut={r['ncut']:.4f} rcut={r['rcut']:.2f} acc={r['accuracy']:.4f} ({r['time_s']}s)")

            with open("nonparam_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Report
    lines = ["# Non-Parametric PGCut: Softmax + Adam + Temp Annealing", ""]
    subset = sorted(all_results, key=lambda r: r["ncut"])
    lines.append("| Method | Config | NCut ↓ | RCut ↓ | ACC | NMI | Time |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in subset:
        lines.append(f"| {r['method']} | {r.get('config','-')} | {r['ncut']:.4f} | {r['rcut']:.2f} | "
                     f"{r['accuracy']:.4f} | {r.get('nmi',0):.4f} | {r.get('time_s','-')}s |")
    with open("nonparam_report.md", "w") as f:
        f.write("\n".join(lines))
    print("\nReport: nonparam_report.md")


if __name__ == "__main__":
    main()
