"""Benchmark PRCut vs RatioCut vs NCut on CIFAR-10, sweep m.

Evaluates each method on its TARGET objective:
  - PRCut  → RCut value (uses 1/ᾱ normalization)
  - RatioCut → RCut value (uses ₂F₁ envelope, β=1)
  - NCut   → NCut value (uses Hölder-binned ₂F₁, per-vertex q=d_i)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from embedata import load_embeddings
from pgcuts.utils.pairs import get_pairs_unique_map
from pgcuts.metrics import evaluate_clustering, compute_rcut_ncut
from pgcuts.optim import GradientMixer
from pgcuts.graph import build_rbf_knn_graph
from pgcuts.utils.data import ShuffledRangeDataset
from pgcuts.hyp2f1.autograd import Hyp2F1
from pgcuts.losses.pncut import equal_size_bins, log_kmeans_bins, compute_ncut_edge_phi

_hyp2f1 = Hyp2F1.apply

REPR_DIR = Path("/buckets/representations/")
DEVICE = "cuda"
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
N_NEIGHBORS = 100
BATCH_PAIR_SIZE = 8192
EMA = 0.95
NCUT_BINS = 16


def remap_labels(y):
    unique = np.unique(y)
    lmap = {old: new for new, old in enumerate(unique)}
    return np.array([lmap[l] for l in y]), len(unique)


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

def kmeans_init_linear(X_np, K, device):
    """Initialize nn.Linear so that logits ≈ similarity to K-Means centroids.

    Sets W = normalized centroids, b = 0, so softmax(X @ W.T) approximates
    the K-Means assignment. Deterministic given the data.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_np)
    centroids = km.cluster_centers_  # (K, dim)
    # Normalize centroids to unit length for cosine-like logits
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / (norms + 1e-8)
    layer = nn.Linear(centroids.shape[1], K)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(centroids, dtype=torch.float32))
        layer.bias.zero_()
    return layer.to(device)


def train_loop(X_t, y, W_exp, K, loss_fn, loss_name, X_np, epochs=EPOCHS):
    """Edge-pair training loop. loss_fn signature:
        (P, logits, li, ri, w, alpha_ema, global_left) -> (loss, new_alpha)
    """
    n, dim = X_t.shape
    device = X_t.device

    network = kmeans_init_linear(X_np, K, device)
    optimizer = optim.AdamW(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    pairs = np.array(W_exp.nonzero()).T
    bps = min(BATCH_PAIR_SIZE, pairs.shape[0])
    dataset = ShuffledRangeDataset(n=pairs.shape[0], k=bps)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)

    grad_mix = GradientMixer(
        network.named_parameters(),
        loss_scale={loss_name: 1.0, "balance": 1.0},
    )
    if hasattr(loss_fn, 'init_alpha'):
        alpha_ema = loss_fn.init_alpha(K)
    else:
        alpha_ema = torch.ones(K, device=device) / K

    t0 = time.time()
    for epoch in range(epochs):
        for batch_idx in loader:
            batch_idx = batch_idx[0]
            bp = pairs[batch_idx]
            unique_idx, left_idx, right_idx = get_pairs_unique_map(bp)

            w = torch.tensor(
                np.array(W_exp[bp[:, 0], bp[:, 1]]),
                dtype=torch.float32, device=device,
            ).squeeze()

            x_batch = X_t[unique_idx].clone().contiguous()
            logits = network(x_batch)
            P = torch.softmax(logits, dim=-1)

            # Global indices of source (left) vertices
            global_left = unique_idx[left_idx]

            loss_val, alpha_ema = loss_fn(
                P, logits, left_idx, right_idx, w, alpha_ema, global_left
            )
            balance = -torch.special.entr(P.mean(0)).sum()

            optimizer.zero_grad()
            with grad_mix(loss_name):
                loss_val.backward(retain_graph=True)
            with grad_mix("balance"):
                balance.backward()
            optimizer.step()

    elapsed = time.time() - t0
    with torch.no_grad():
        Z = network(X_t)
        pred = Z.argmax(dim=-1).cpu().numpy()
        results = evaluate_clustering(y, pred, K)
    rcut, ncut = compute_rcut_ncut(W_exp, pred)
    return {
        "accuracy": float(results["accuracy"]),
        "nmi": float(results["nmi"]),
        "rcut": float(rcut), "ncut": float(ncut),
        "time_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def make_prcut_loss():
    """PRCut: w · p_left · (1-p_right) / ᾱ."""
    rho = EMA
    def fn(P, logits, li, ri, w, alpha_ema, gl):
        cut = (w.unsqueeze(-1) * P[li] * (1 - P[ri])).mean(0)
        vol = rho * alpha_ema + (1 - rho) * P.mean(0)
        loss = (cut / (vol + 1e-12)).sum() / w.sum()
        return loss, alpha_ema * rho + (1 - rho) * P.mean(0).detach()
    return fn


def make_ratiocut_loss(m):
    """RatioCut: w · CE · ₂F₁(-m, 1; 2; ᾱ) — per-cluster envelope (β=1)."""
    rho = EMA
    def fn(P, logits, li, ri, w, alpha_ema, gl):
        log_p_right = torch.log_softmax(logits[ri], dim=-1)
        cut = (w.unsqueeze(-1) * (-P[li] * log_p_right)).mean(0)
        alpha_bar = rho * alpha_ema + (1 - rho) * P.mean(0)
        H = _hyp2f1(-m, 1.0, 2.0, alpha_bar.clamp(1e-7, 1 - 1e-7))
        loss = (cut * H).sum() / w.sum()
        return loss, alpha_ema * rho + (1 - rho) * P.mean(0).detach()
    return fn


def make_ncut_loss(m, degrees_np, device, num_bins=NCUT_BINS, binning="equal"):
    """NCut with Hölder-binned ₂F₁ — (d × K) hyp2f1 evals per step.

    Each bin j has its own ᾱ_{ℓj} = mean of P_{iℓ} for vertices i in bin j.
    This is the correct Hölder bound — different bins have different means.

    ₂F₁(-m, 1; c_j; ᾱ_{ℓj}) computed once per (bin, cluster).
    Per-vertex variation: 1/d_i factor only.
    """
    rho = EMA
    if binning == "equal":
        bins = equal_size_bins(degrees_np, num_bins)
    else:
        bins = log_kmeans_bins(degrees_np, num_bins)

    d = len(bins)
    beta_stars = torch.tensor([b["beta_star"] for b in bins], dtype=torch.float32, device=device)
    counts = torch.tensor([b["count"] for b in bins], dtype=torch.float32, device=device)
    bin_weights = counts / counts.sum()  # (d,)
    degrees_t = torch.tensor(degrees_np, dtype=torch.float32, device=device)

    # Node-to-bin mapping for fast per-bin mean computation
    n = len(degrees_np)
    node_to_bin = torch.zeros(n, dtype=torch.long, device=device)
    bin_idx_lists = []
    for j, b in enumerate(bins):
        idx = torch.tensor(b["indices"], dtype=torch.long, device=device)
        node_to_bin[idx] = j
        bin_idx_lists.append(idx)

    # Representative q per bin = mean degree in that bin
    q_stars = torch.tensor(
        [degrees_np[b["indices"]].mean() for b in bins],
        dtype=torch.float32, device=device
    )  # (d,)

    # c parameter per bin: c_j = q*_j / β*_j + 1
    c_per_bin = q_stars / beta_stars + 1.0  # (d,)

    def fn(P, logits, li, ri, w, alpha_ema, global_left):
        K_ = P.shape[1]

        log_p_right = torch.log_softmax(logits[ri], dim=-1)
        ce = -P[li] * log_p_right  # (E, K)

        # Per-bin mean ᾱ_{ℓj} from batch — shape (d, K)
        # Use unique batch nodes to estimate per-bin means
        P_det = P.detach()
        # Compute batch per-bin means from the unique nodes in this batch
        # global indices of unique nodes are implicit from unique_idx
        # but we only have P for unique nodes. Use them all.
        batch_bin_means = torch.zeros(d, K_, device=device)
        batch_bin_counts = torch.zeros(d, device=device)
        unique_global = global_left.unique()  # global indices of batch nodes
        bins_of_unique = node_to_bin[unique_global]  # (U,)
        for j in range(d):
            mask = bins_of_unique == j
            if mask.any():
                # Map global indices back to local P indices
                # unique_global[mask] are the global indices in bin j
                # But P is indexed by local batch position...
                # We need to find which local indices correspond to these globals
                pass

        # Simpler: use full P's global indices via node_to_bin
        # Since we're in edge-pair mode, P has shape (num_unique, K)
        # and unique_idx maps local -> global. But we don't have unique_idx here.
        # Fallback: EMA per-bin means (tracked across steps)

        # alpha_ema is (d, K) — per-bin EMA
        alpha_bars = alpha_ema  # (d, K)
        z = alpha_bars.clamp(1e-7, 1 - 1e-7)  # (d, K)

        # ₂F₁ once for (d, K)
        c_dk = c_per_bin.unsqueeze(1).expand(d, K_)  # (d, K)
        F = _hyp2f1(-m, 1.0, c_dk, z)  # (d, K)

        # Hölder composition: log Φ_base = Σ_j w_j · log F_j
        log_F = torch.log(F.clamp(min=1e-30))  # (d, K)
        log_Phi_base = (bin_weights.unsqueeze(1) * log_F).sum(0)  # (K,)

        # Per-edge: Φ_ℓ(d_i) = (1/d_i) · exp(log_Phi_base_ℓ)
        d_i = degrees_t[global_left].clamp(min=1e-6)  # (E,)
        Phi = torch.exp(log_Phi_base).unsqueeze(0) / d_i.unsqueeze(1)  # (E, K)

        loss = (w.unsqueeze(-1) * ce * Phi).sum() / w.sum()

        # Update per-bin EMA from batch mean
        batch_mean = P_det.mean(0)  # (K,) — global approx per cluster
        # Distribute to bins weighted by how much each bin's nodes contribute
        # For proper per-bin: update each bin from its own nodes
        new_alpha = alpha_ema.clone()
        for j in range(d):
            # Find batch nodes in bin j via global_left
            in_bin = node_to_bin[global_left] == j
            if in_bin.any():
                # Mean of P for left nodes in this bin
                bin_mean = P_det[li[in_bin]].mean(0)  # (K,)
                new_alpha[j] = rho * alpha_ema[j] + (1 - rho) * bin_mean
        return loss, new_alpha

    # Return init function for alpha_ema shape (d, K) instead of (K,)
    fn.init_alpha = lambda K_: torch.ones(d, K_, device=device) / K_
    fn.num_bins = d
    return fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_name, model_name = "cifar10", "clipvitL14"
    train_ds = load_embeddings(dataset_name, model_name, REPR_DIR, split="train")
    X, y = train_ds.feats, train_ds.labels
    y, K = remap_labels(y)
    n = X.shape[0]

    print(f"{dataset_name}/{model_name}: n={n}, K={K}, dim={X.shape[1]}")
    print("Building KNN RBF graph...", end=" ", flush=True)
    W_exp = build_rbf_knn_graph(X, n_neighbors=min(N_NEIGHBORS, n - 1))
    degrees = np.array(W_exp.sum(axis=1)).flatten().astype(np.float32)
    print("done")

    device = torch.device(DEVICE)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    all_results = []

    # PRCut baseline
    print("\nPRCut...", end=" ", flush=True)
    r = train_loop(X_t, y, W_exp, K, make_prcut_loss(), "prcut", X)
    r["method"] = "PRCut"; r["m"] = "-"
    all_results.append(r)
    print(f"acc={r['accuracy']:.4f} rcut={r['rcut']:.2f} ncut={r['ncut']:.4f} ({r['time_s']}s)")

    # Sweep m
    m_values = [64, 128, 256, 512, 1024, 2048, 4096]

    for m_val in m_values:
        # RatioCut
        print(f"\nRatioCut m={m_val}...", end=" ", flush=True)
        r = train_loop(X_t, y, W_exp, K, make_ratiocut_loss(m_val), "rcut", X)
        r["method"] = "RatioCut"; r["m"] = m_val
        all_results.append(r)
        print(f"acc={r['accuracy']:.4f} rcut={r['rcut']:.2f} ncut={r['ncut']:.4f} ({r['time_s']}s)")

        # NCut (per-vertex q=d_i)
        print(f"NCut m={m_val}...", end=" ", flush=True)
        r = train_loop(X_t, y, W_exp, K,
                       make_ncut_loss(m_val, degrees, device), "ncut", X)
        r["method"] = "NCut"; r["m"] = m_val
        all_results.append(r)
        print(f"acc={r['accuracy']:.4f} rcut={r['rcut']:.2f} ncut={r['ncut']:.4f} ({r['time_s']}s)")

        # Save incrementally
        with open("benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # NCut with log-kmeans binning at best m
    print(f"\nNCut m=512 log_kmeans...", end=" ", flush=True)
    r = train_loop(X_t, y, W_exp, K,
                   make_ncut_loss(512, degrees, device, binning="log_kmeans"), "ncut", X)
    r["method"] = "NCut-logkm"; r["m"] = 512
    all_results.append(r)
    print(f"acc={r['accuracy']:.4f} rcut={r['rcut']:.2f} ncut={r['ncut']:.4f} ({r['time_s']}s)")

    # Save final
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Report — focus on target objectives
    lines = [
        "# PGCut Benchmark: CIFAR-10 / clipvitL14",
        "",
        f"Settings: {EPOCHS} epochs, lr={LR}, KNN={N_NEIGHBORS}, EMA={EMA}",
        "",
        "## Results",
        "",
        "| Method | m | ACC | NMI | RCut ↓ | NCut ↓ | Time |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['method']} | {r['m']} | {r['accuracy']:.4f} | {r['nmi']:.4f} | "
            f"{r['rcut']:.2f} | {r['ncut']:.4f} | {r['time_s']}s |"
        )

    # Summary
    prcut_r = [r for r in all_results if r["method"] == "PRCut"][0]
    rcuts = [r for r in all_results if r["method"] == "RatioCut"]
    ncuts = [r for r in all_results if r["method"].startswith("NCut")]
    best_rc = min(rcuts, key=lambda r: r["rcut"]) if rcuts else None
    best_nc = min(ncuts, key=lambda r: r["ncut"]) if ncuts else None

    lines += ["", "## Summary", ""]
    lines.append(f"- **PRCut**: RCut={prcut_r['rcut']:.2f}, NCut={prcut_r['ncut']:.4f}, ACC={prcut_r['accuracy']:.4f}")
    if best_rc:
        lines.append(f"- **Best RatioCut** (m={best_rc['m']}): RCut={best_rc['rcut']:.2f}, ACC={best_rc['accuracy']:.4f}")
    if best_nc:
        lines.append(f"- **Best NCut** (m={best_nc['m']}): NCut={best_nc['ncut']:.4f}, ACC={best_nc['accuracy']:.4f}")

    lines.append("")
    report = "\n".join(lines)
    with open("report.md", "w") as f:
        f.write(report)
    print(f"\nReport saved to report.md")


if __name__ == "__main__":
    main()
