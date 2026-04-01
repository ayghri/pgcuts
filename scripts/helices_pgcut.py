"""Train H-RCut and H-NCut on the 3-helices toy dataset.

Non-parametric PGCut with diffusion basis parameterization:
  logits = V @ theta,  where V = (D^{-1}W)^{hops} @ random_vectors
This gives multi-hop graph structure without spectral decomposition.

Anti-collapse via log-barrier on cluster proportions (not entropy balance,
which would force equal-sized clusters on an unbalanced dataset).

Compares: Spectral, H-RCut (Theorem 1, beta=1), H-NCut (Theorem 2, Holder binned).
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

np.random.seed(42)
X, y_true = make_helices(1200, noise=0.2)
K = 3

# Unbalance: remove 200 from cluster 0
norms_c0 = np.linalg.norm(X[y_true == 0], axis=1)
thresh = np.sort(norms_c0)[-200]
keep = ~((y_true == 0) & (np.linalg.norm(X, axis=1) < thresh))
X, y_true = X[keep], y_true[keep]
N = X.shape[0]
print(f"N={N}, K={K}, sizes={[np.sum(y_true==k) for k in range(K)]}", flush=True)


# -- Graph -------------------------------------------------------------------

dists = pairwise_distances(X, metric="euclidean")
sigma = np.mean(dists) / 12.0
ADJ_THRESH = 0.5
W_np = np.exp(-(dists / sigma) ** 2 / 2.0)
np.fill_diagonal(W_np, 0)
W_np = (W_np + W_np.T) / 2.0
W_np = W_np * (W_np > ADJ_THRESH)
D_np = W_np.sum(axis=1).astype(np.float32)
print(f"Sigma={sigma:.4f}, Threshold={ADJ_THRESH}", flush=True)
print(f"Edges={W_np.astype(bool).sum()//2}, Mean deg={D_np.mean():.1f}", flush=True)


# -- Spectral baseline -------------------------------------------------------

sc = SpectralClustering(n_clusters=K, affinity="precomputed",
                        assign_labels="kmeans", n_jobs=-1, random_state=42)
sc_pred = sc.fit_predict(W_np)
sc_rc, sc_nc = compute_rcut_ncut(sp.csr_matrix(W_np), sc_pred)
sc_ari = adjusted_rand_score(y_true, sc_pred)
print(f"Spectral: ncut={sc_nc:.4f} rcut={sc_rc:.2f} ARI={sc_ari:.4f}", flush=True)


# -- GPU setup ---------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W_coo = sp.coo_matrix(W_np)
W_t = torch.sparse_coo_tensor(
    torch.tensor(np.vstack([W_coo.row, W_coo.col]), dtype=torch.long, device=device),
    torch.tensor(W_coo.data, dtype=torch.float32, device=device), W_coo.shape
).coalesce()
W_sum = W_np.sum()
degrees_t = torch.tensor(D_np, dtype=torch.float32, device=device).clamp(min=1e-6)
D_inv = (1.0 / degrees_t).unsqueeze(1)  # (N, 1) for broadcasting


# -- Diffusion basis ---------------------------------------------------------

def make_diffusion_basis(r, n_hops, seed=0):
    """Create basis by diffusing r random vectors through the graph.

    V = normalize( (D^{-1}W)^{n_hops} @ V0 )

    Uses sparse matmul: each hop is D_inv * sparse_mm(W, V), which is
    O(nnz * r) instead of O(N^2 * r) for the dense version.  This scales
    to large graphs (e.g., CIFAR-10 with N=50k).

    As n_hops -> infinity the columns of V converge to the dominant
    eigenvectors of D^{-1}W (power iteration on r vectors simultaneously).
    """
    torch.manual_seed(seed)
    V = torch.randn(N, r, device=device)
    for _ in range(n_hops):
        V = D_inv * torch.sparse.mm(W_t, V)
        # Periodic re-normalization to prevent overflow/underflow
        V = V / (V.norm(dim=0, keepdim=True) + 1e-8)
    return V.detach()


# -- Training function -------------------------------------------------------

def train_pgcut(mode="hncut", m=512, num_bins=16, lr=0.5, wd=0.17,
                bal_weight=0.0005, epochs=30000, n_restarts=10,
                basis_r=50, basis_hops=1000):
    """Train non-parametric PGCut with diffusion basis.

    Parameters
    ----------
    mode : 'hrcut' (Theorem 1, beta=1) or 'hncut' (Theorem 2, Holder binned)
    m : polynomial degree for 2F1 envelope
    num_bins : number of log-kmeans bins for Holder bound (hncut only)
    lr, wd : AdamW learning rate and weight decay
    bal_weight : log-barrier weight (anti-collapse, NOT entropy balance).
        ~0.0005 is the sweet spot: prevents collapse without forcing uniformity.
    epochs : training epochs per restart
    n_restarts : take best of n random restarts (pick by lowest NCut/RCut)
    basis_r : number of random vectors for diffusion basis
    basis_hops : number of diffusion steps
    """
    # Binning setup
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
        V = make_diffusion_basis(basis_r, basis_hops, seed=restart)
        theta = torch.nn.Parameter(0.01 * torch.randn(basis_r, K, device=device))
        alpha_ema = torch.ones(d, K, device=device) / K

        optimizer = torch.optim.AdamW([theta], lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        history = {"epoch": [], "ncut": [], "rcut": [], "ari": []}
        t0 = time.time()

        for epoch in range(epochs):
            logits = V @ theta
            P = torch.softmax(logits, dim=-1)
            M = P * torch.sparse.mm(W_t, 1.0 - P)

            if mode == "hrcut":
                # Theorem 1: H_l = 2F1(-m, 1; 2; alpha_bar_l)
                ab = P.detach().mean(0)
                with torch.no_grad():
                    alpha_ema[0] = 0.9 * alpha_ema[0] + 0.1 * ab
                H = _h(-m, 1.0, 2.0, alpha_ema[0].clamp(1e-7, 1 - 1e-7))
                U = (M.sum(0) * H).sum() / W_sum

            elif mode == "hncut":
                # Theorem 2: Holder-binned, per-bin 2F1
                with torch.no_grad():
                    for j, ix in enumerate(bin_idx):
                        alpha_ema[j] = 0.9 * alpha_ema[j] + 0.1 * P[ix].mean(0)
                z = alpha_ema.clamp(1e-7, 1 - 1e-7)
                F = _h(-m, 1.0, c_per_bin.unsqueeze(1).expand(d, K), z)
                log_Phi = (bin_weights.unsqueeze(1) * torch.log(F.clamp(min=1e-30))).sum(0)
                Phi = torch.exp(log_Phi).unsqueeze(0) / degrees_t.unsqueeze(1)
                U = (M * Phi).sum() / W_sum

            # Anti-collapse: log-barrier on cluster proportions
            # Unlike entropy balance, this does NOT force equal-sized clusters.
            pi_k = P.mean(0)
            barrier = -torch.log(pi_k.clamp(min=1e-6)).sum()

            loss = U + bal_weight * barrier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 1000 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    logits_eval = V @ theta
                    pred = logits_eval.argmax(dim=-1).cpu().numpy()
                    rc, nc = compute_rcut_ncut(sp.csr_matrix(W_np), pred)
                    ari = adjusted_rand_score(y_true, pred)
                    history["epoch"].append(epoch)
                    history["ncut"].append(nc)
                    history["rcut"].append(rc)
                    history["ari"].append(ari)

        elapsed = time.time() - t0
        logits_final = (V @ theta).detach()
        pred = logits_final.argmax(dim=-1).cpu().numpy()
        ari = adjusted_rand_score(y_true, pred)
        rc, nc = compute_rcut_ncut(sp.csr_matrix(W_np), pred)
        sizes = np.bincount(pred, minlength=K)
        print(f"  restart {restart}: ncut={nc:.4f} rcut={rc:.2f} ARI={ari:.4f} "
              f"sizes={sizes.tolist()} ({elapsed:.1f}s)", flush=True)

        # Select best restart by the objective being optimized:
        # H-RCut -> pick by lowest RCut, H-NCut -> pick by lowest NCut
        score = rc if mode == "hrcut" else nc
        if score < best_score:
            best_pred, best_score, best_hist = pred, score, history

    # Report the ARI of the selected restart
    best_ari = adjusted_rand_score(y_true, best_pred)
    return best_pred, best_hist, best_ari


# -- Run experiments ---------------------------------------------------------

print("\n--- H-RCut (Theorem 1, beta=1, diffusion basis) ---", flush=True)
hrcut_pred, hrcut_hist, hrcut_ari = train_pgcut(
    mode="hrcut", m=512,
    lr=0.5, wd=0.17, bal_weight=0.0005, epochs=30000,
    n_restarts=10, basis_r=50, basis_hops=1000,
)

print(f"\n--- H-NCut (Theorem 2, Holder, 16 bins, diffusion basis) ---", flush=True)
hncut_pred, hncut_hist, hncut_ari = train_pgcut(
    mode="hncut", m=512, num_bins=16,
    lr=0.5, wd=0.17, bal_weight=0.0005, epochs=30000,
    n_restarts=10, basis_r=50, basis_hops=1000,
)


# -- Summary -----------------------------------------------------------------

print(f"\n{'='*50}", flush=True)
print(f"{'Method':<15} {'NCut':>8} {'RCut':>8} {'ARI':>8}", flush=True)
print(f"{'-'*50}", flush=True)

rc, nc = compute_rcut_ncut(sp.csr_matrix(W_np), sc_pred)
print(f"{'Spectral':<15} {nc:>8.4f} {rc:>8.2f} {sc_ari:>8.4f}", flush=True)

rc, nc = compute_rcut_ncut(sp.csr_matrix(W_np), hrcut_pred)
print(f"{'H-RCut':<15} {nc:>8.4f} {rc:>8.2f} {hrcut_ari:>8.4f}", flush=True)

rc, nc = compute_rcut_ncut(sp.csr_matrix(W_np), hncut_pred)
print(f"{'H-NCut':<15} {nc:>8.4f} {rc:>8.2f} {hncut_ari:>8.4f}", flush=True)


# -- Plots -------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="brg", s=12,
                edgecolors="white", linewidths=0.2)
axes[0].set_title("Ground Truth"); axes[0].axis("off")

axes[1].scatter(X[:, 0], X[:, 1], c=sc_pred, cmap="brg", s=12,
                edgecolors="white", linewidths=0.2)
axes[1].set_title(f"Spectral (ARI={sc_ari:.3f})"); axes[1].axis("off")

axes[2].scatter(X[:, 0], X[:, 1], c=hrcut_pred, cmap="brg", s=12,
                edgecolors="white", linewidths=0.2)
axes[2].set_title(f"H-RCut (ARI={hrcut_ari:.3f})"); axes[2].axis("off")

axes[3].scatter(X[:, 0], X[:, 1], c=hncut_pred, cmap="brg", s=12,
                edgecolors="white", linewidths=0.2)
axes[3].set_title(f"H-NCut (ARI={hncut_ari:.3f})"); axes[3].axis("off")

fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/helices_clustering.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nSaved to {SAVE_DIR}/helices_clustering.png", flush=True)

# Convergence plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for hist, name, color in [(hrcut_hist, "H-RCut", "tab:red"),
                           (hncut_hist, "H-NCut", "tab:green")]:
    axes[0].plot(hist["epoch"], hist["ncut"], "-o", ms=3, label=name, color=color)
    axes[1].plot(hist["epoch"], hist["ari"], "-o", ms=3, label=name, color=color)

axes[0].axhline(sc_nc, ls="--", color="tab:blue", label=f"Spectral ({sc_nc:.4f})")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("NCut"); axes[0].set_title("NCut Convergence")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].axhline(sc_ari, ls="--", color="tab:blue", label=f"Spectral ({sc_ari:.3f})")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ARI"); axes[1].set_title("ARI Convergence")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/helices_convergence.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved to {SAVE_DIR}/helices_convergence.png", flush=True)
