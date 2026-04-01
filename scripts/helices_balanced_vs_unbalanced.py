"""Compare H-RCut vs H-NCut on balanced and unbalanced 3-helices.

Hypothesis: H-RCut (beta=1) fails when clusters are unbalanced because it
normalizes by cluster size |C_k| (RatioCut), not volume (NCut).  H-NCut
with the Holder bound adapts to heterogeneous degrees and handles imbalance.

Produces a single figure with 2 rows x 5 columns:
  Row 1: Balanced   (400, 400, 400) — Ground Truth, Spectral, H-RCut, H-NCut
  Row 2: Unbalanced (200, 400, 400) — Ground Truth, Spectral, H-RCut, H-NCut
"""

import sys
import time
import numpy as np
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from sklearn.cluster import SpectralClustering

sys.stdout.reconfigure(line_buffering=True)

from pgcuts.hyp2f1.autograd import Hyp2F1
from pgcuts.losses.pncut import log_kmeans_bins
from pgcuts.metrics import compute_rcut_ncut

_h = Hyp2F1.apply

SAVE_DIR = "./outputs"
import os; os.makedirs(SAVE_DIR, exist_ok=True)


# -- Dataset -----------------------------------------------------------------

def make_helices(n_samples=1200, noise=0.2, n_helices=3):
    t = np.linspace(np.pi, 4 * np.pi, n_samples // n_helices)
    X, y = [], []
    for i in range(n_helices):
        theta = t + (i * 2 * np.pi / n_helices)
        helix = np.stack([t * np.cos(theta), t * np.sin(theta)], axis=1)
        helix += np.random.normal(scale=noise, size=helix.shape)
        X.append(helix); y.append(np.full(len(t), i))
    return np.vstack(X), np.concatenate(y)


def make_graph(X):
    dists = pairwise_distances(X, metric="euclidean")
    sigma = np.mean(dists) / 12.0
    W = np.exp(-(dists / sigma) ** 2 / 2.0)
    np.fill_diagonal(W, 0)
    W = (W + W.T) / 2.0
    W = W * (W > 0.5)
    D = W.sum(axis=1).astype(np.float32)
    return W, D


# -- PGCut training ----------------------------------------------------------

def train_pgcut(W_np, D_np, y_true, K, mode="hncut", m=512, num_bins=16,
                lr=0.5, wd=0.17, bal_weight=0.0005, epochs=30000,
                n_restarts=10, basis_r=50, basis_hops=1000):
    N = W_np.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    W_coo = sp.coo_matrix(W_np)
    W_t = torch.sparse_coo_tensor(
        torch.tensor(np.vstack([W_coo.row, W_coo.col]), dtype=torch.long, device=device),
        torch.tensor(W_coo.data, dtype=torch.float32, device=device), W_coo.shape
    ).coalesce()
    W_sum = W_np.sum()
    degrees_t = torch.tensor(D_np, dtype=torch.float32, device=device).clamp(min=1e-6)
    D_inv = (1.0 / degrees_t).unsqueeze(1)

    bins = log_kmeans_bins(D_np, num_bins)
    d = len(bins)
    counts = torch.tensor([b["count"] for b in bins], dtype=torch.float32, device=device)
    bin_weights = counts / counts.sum()
    bin_idx = [torch.tensor(b["indices"], dtype=torch.long, device=device) for b in bins]
    q_stars = torch.tensor([D_np[b["indices"]].mean() for b in bins],
                           dtype=torch.float32, device=device)
    beta_stars = torch.tensor([b["beta_star"] for b in bins],
                              dtype=torch.float32, device=device)
    c_per_bin = q_stars / beta_stars + 1.0

    best_pred, best_score, best_hist = None, float("inf"), None

    for restart in range(n_restarts):
        torch.manual_seed(restart)
        V = torch.randn(N, basis_r, device=device)
        for _ in range(basis_hops):
            V = D_inv * torch.sparse.mm(W_t, V)
            V = V / (V.norm(dim=0, keepdim=True) + 1e-8)
        V = V.detach()

        theta = torch.nn.Parameter(0.01 * torch.randn(basis_r, K, device=device))
        alpha_ema = torch.ones(d, K, device=device) / K
        optimizer = torch.optim.AdamW([theta], lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)

        history = {"epoch": [], "ncut": [], "rcut": [], "ari": []}

        for epoch in range(epochs):
            logits = V @ theta
            P = torch.softmax(logits, dim=-1)
            M = P * torch.sparse.mm(W_t, 1.0 - P)

            if mode == "hrcut":
                ab = P.detach().mean(0)
                with torch.no_grad():
                    alpha_ema[0] = 0.9 * alpha_ema[0] + 0.1 * ab
                H = _h(-m, 1.0, 2.0, alpha_ema[0].clamp(1e-7, 1 - 1e-7))
                U = (M.sum(0) * H).sum() / W_sum
            elif mode == "hncut":
                with torch.no_grad():
                    for j, ix in enumerate(bin_idx):
                        alpha_ema[j] = 0.9 * alpha_ema[j] + 0.1 * P[ix].mean(0)
                z = alpha_ema.clamp(1e-7, 1 - 1e-7)
                F = _h(-m, 1.0, c_per_bin.unsqueeze(1).expand(d, K), z)
                log_Phi = (bin_weights.unsqueeze(1) * torch.log(F.clamp(min=1e-30))).sum(0)
                Phi = torch.exp(log_Phi).unsqueeze(0) / degrees_t.unsqueeze(1)
                U = (M * Phi).sum() / W_sum

            pi_k = P.mean(0)
            barrier = -torch.log(pi_k.clamp(min=1e-6)).sum()
            loss = U + bal_weight * barrier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 1000 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    pred = (V @ theta).argmax(dim=-1).cpu().numpy()
                    rc, nc = compute_rcut_ncut(sp.csr_matrix(W_np), pred)
                    ari = adjusted_rand_score(y_true, pred)
                    history["epoch"].append(epoch)
                    history["ncut"].append(nc)
                    history["rcut"].append(rc)
                    history["ari"].append(ari)

        pred = (V @ theta).detach().argmax(dim=-1).cpu().numpy()
        ari = adjusted_rand_score(y_true, pred)
        rc, nc = compute_rcut_ncut(sp.csr_matrix(W_np), pred)
        sizes = np.bincount(pred, minlength=K)
        print(f"    restart {restart}: ncut={nc:.4f} rcut={rc:.2f} ARI={ari:.4f} "
              f"sizes={sizes.tolist()}", flush=True)

        score = rc if mode == "hrcut" else nc
        if score < best_score:
            best_pred, best_score, best_hist = pred, score, history

    best_ari = adjusted_rand_score(y_true, best_pred)
    best_rc, best_nc = compute_rcut_ncut(sp.csr_matrix(W_np), best_pred)
    return best_pred, best_hist, best_ari, best_rc, best_nc


# -- Run both scenarios ------------------------------------------------------

np.random.seed(42)
X_full, y_full = make_helices(1200, noise=0.2)
K = 3

scenarios = {}

# Balanced: all 3 clusters have 400 points
print("=" * 60, flush=True)
print("BALANCED (400, 400, 400)", flush=True)
print("=" * 60, flush=True)
X_bal, y_bal = X_full.copy(), y_full.copy()
N_bal = X_bal.shape[0]
print(f"N={N_bal}, sizes={[np.sum(y_bal==k) for k in range(K)]}", flush=True)
W_bal, D_bal = make_graph(X_bal)

sc = SpectralClustering(n_clusters=K, affinity="precomputed",
                        assign_labels="kmeans", n_jobs=-1, random_state=42)
sc_pred_bal = sc.fit_predict(W_bal)
sc_rc_bal, sc_nc_bal = compute_rcut_ncut(sp.csr_matrix(W_bal), sc_pred_bal)
sc_ari_bal = adjusted_rand_score(y_bal, sc_pred_bal)
print(f"Spectral: ncut={sc_nc_bal:.4f} rcut={sc_rc_bal:.2f} ARI={sc_ari_bal:.4f}", flush=True)

print("  H-RCut:", flush=True)
hrcut_pred_bal, hrcut_hist_bal, hrcut_ari_bal, hrcut_rc_bal, hrcut_nc_bal = train_pgcut(
    W_bal, D_bal, y_bal, K, mode="hrcut")

print("  H-NCut:", flush=True)
hncut_pred_bal, hncut_hist_bal, hncut_ari_bal, hncut_rc_bal, hncut_nc_bal = train_pgcut(
    W_bal, D_bal, y_bal, K, mode="hncut")

scenarios["balanced"] = {
    "X": X_bal, "y": y_bal,
    "sc_pred": sc_pred_bal, "sc_ari": sc_ari_bal,
    "hrcut_pred": hrcut_pred_bal, "hrcut_ari": hrcut_ari_bal,
    "hrcut_rc": hrcut_rc_bal, "hrcut_nc": hrcut_nc_bal,
    "hncut_pred": hncut_pred_bal, "hncut_ari": hncut_ari_bal,
    "hncut_rc": hncut_rc_bal, "hncut_nc": hncut_nc_bal,
    "hrcut_hist": hrcut_hist_bal, "hncut_hist": hncut_hist_bal,
    "sc_nc": sc_nc_bal, "sc_ari_val": sc_ari_bal,
}

# Unbalanced: remove 200 from cluster 0 -> (200, 400, 400)
print("\n" + "=" * 60, flush=True)
print("UNBALANCED (200, 400, 400)", flush=True)
print("=" * 60, flush=True)
norms_c0 = np.linalg.norm(X_full[y_full == 0], axis=1)
thresh = np.sort(norms_c0)[-200]
keep = ~((y_full == 0) & (np.linalg.norm(X_full, axis=1) < thresh))
X_unb, y_unb = X_full[keep], y_full[keep]
N_unb = X_unb.shape[0]
print(f"N={N_unb}, sizes={[np.sum(y_unb==k) for k in range(K)]}", flush=True)
W_unb, D_unb = make_graph(X_unb)

sc_pred_unb = SpectralClustering(
    n_clusters=K, affinity="precomputed",
    assign_labels="kmeans", n_jobs=-1, random_state=42
).fit_predict(W_unb)
sc_rc_unb, sc_nc_unb = compute_rcut_ncut(sp.csr_matrix(W_unb), sc_pred_unb)
sc_ari_unb = adjusted_rand_score(y_unb, sc_pred_unb)
print(f"Spectral: ncut={sc_nc_unb:.4f} rcut={sc_rc_unb:.2f} ARI={sc_ari_unb:.4f}", flush=True)

print("  H-RCut:", flush=True)
hrcut_pred_unb, hrcut_hist_unb, hrcut_ari_unb, hrcut_rc_unb, hrcut_nc_unb = train_pgcut(
    W_unb, D_unb, y_unb, K, mode="hrcut")

print("  H-NCut:", flush=True)
hncut_pred_unb, hncut_hist_unb, hncut_ari_unb, hncut_rc_unb, hncut_nc_unb = train_pgcut(
    W_unb, D_unb, y_unb, K, mode="hncut")

scenarios["unbalanced"] = {
    "X": X_unb, "y": y_unb,
    "sc_pred": sc_pred_unb, "sc_ari": sc_ari_unb,
    "hrcut_pred": hrcut_pred_unb, "hrcut_ari": hrcut_ari_unb,
    "hrcut_rc": hrcut_rc_unb, "hrcut_nc": hrcut_nc_unb,
    "hncut_pred": hncut_pred_unb, "hncut_ari": hncut_ari_unb,
    "hncut_rc": hncut_rc_unb, "hncut_nc": hncut_nc_unb,
    "hrcut_hist": hrcut_hist_unb, "hncut_hist": hncut_hist_unb,
    "sc_nc": sc_nc_unb, "sc_ari_val": sc_ari_unb,
}


# -- Summary table -----------------------------------------------------------

print(f"\n{'=' * 70}", flush=True)
print(f"{'Scenario':<14} {'Method':<12} {'NCut':>8} {'RCut':>8} {'ARI':>8}", flush=True)
print(f"{'-' * 70}", flush=True)
for label, s in scenarios.items():
    print(f"{label:<14} {'Spectral':<12} {s['sc_nc']:.4f}{'':>4} {s['sc_ari']:>8.4f}", flush=True)
    print(f"{'':14} {'H-RCut':<12} {s['hrcut_nc']:>8.4f} {s['hrcut_rc']:>8.2f} {s['hrcut_ari']:>8.4f}", flush=True)
    print(f"{'':14} {'H-NCut':<12} {s['hncut_nc']:>8.4f} {s['hncut_rc']:>8.2f} {s['hncut_ari']:>8.4f}", flush=True)
    print(flush=True)


# -- Figure: 2 rows x 4 columns ---------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(20, 9))

for row, (label, s) in enumerate(scenarios.items()):
    X_s = s["X"]
    title_prefix = "Balanced" if label == "balanced" else "Unbalanced"
    ms = 12

    # Ground Truth
    axes[row, 0].scatter(X_s[:, 0], X_s[:, 1], c=s["y"], cmap="brg", s=ms,
                         edgecolors="white", linewidths=0.2)
    sizes = [np.sum(s["y"] == k) for k in range(K)]
    axes[row, 0].set_title(f"{title_prefix}: Ground Truth\n{sizes}")
    axes[row, 0].axis("off")

    # Spectral
    axes[row, 1].scatter(X_s[:, 0], X_s[:, 1], c=s["sc_pred"], cmap="brg", s=ms,
                         edgecolors="white", linewidths=0.2)
    axes[row, 1].set_title(f"Spectral\nARI={s['sc_ari']:.3f}")
    axes[row, 1].axis("off")

    # H-RCut
    axes[row, 2].scatter(X_s[:, 0], X_s[:, 1], c=s["hrcut_pred"], cmap="brg", s=ms,
                         edgecolors="white", linewidths=0.2)
    axes[row, 2].set_title(f"H-RCut\nARI={s['hrcut_ari']:.3f}  RCut={s['hrcut_rc']:.2f}")
    axes[row, 2].axis("off")

    # H-NCut
    axes[row, 3].scatter(X_s[:, 0], X_s[:, 1], c=s["hncut_pred"], cmap="brg", s=ms,
                         edgecolors="white", linewidths=0.2)
    axes[row, 3].set_title(f"H-NCut\nARI={s['hncut_ari']:.3f}  NCut={s['hncut_nc']:.4f}")
    axes[row, 3].axis("off")

fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/helices_balanced_vs_unbalanced.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved to {SAVE_DIR}/helices_balanced_vs_unbalanced.png", flush=True)


# -- Convergence figure: 2 rows x 2 columns ---------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row, (label, s) in enumerate(scenarios.items()):
    title_prefix = "Balanced" if label == "balanced" else "Unbalanced"

    for hist, name, color in [(s["hrcut_hist"], "H-RCut", "tab:red"),
                               (s["hncut_hist"], "H-NCut", "tab:green")]:
        axes[row, 0].plot(hist["epoch"], hist["ncut"], "-o", ms=2, label=name, color=color)
        axes[row, 1].plot(hist["epoch"], hist["ari"], "-o", ms=2, label=name, color=color)

    axes[row, 0].axhline(s["sc_nc"], ls="--", color="tab:blue",
                          label=f"Spectral ({s['sc_nc']:.4f})")
    axes[row, 0].set_xlabel("Epoch"); axes[row, 0].set_ylabel("NCut")
    axes[row, 0].set_title(f"{title_prefix}: NCut Convergence")
    axes[row, 0].legend(); axes[row, 0].grid(True, alpha=0.3)

    axes[row, 1].axhline(s["sc_ari_val"], ls="--", color="tab:blue",
                          label=f"Spectral ({s['sc_ari_val']:.3f})")
    axes[row, 1].set_xlabel("Epoch"); axes[row, 1].set_ylabel("ARI")
    axes[row, 1].set_title(f"{title_prefix}: ARI Convergence")
    axes[row, 1].legend(); axes[row, 1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/helices_convergence_balanced_vs_unbalanced.png",
            bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved to {SAVE_DIR}/helices_convergence_balanced_vs_unbalanced.png", flush=True)
