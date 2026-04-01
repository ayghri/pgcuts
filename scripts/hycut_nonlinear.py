"""Batch-based HyCut nonlinear (2-layer SiLU) model — sweep over embeddings.

Objectives: PRCut, Hyp-RCut, Hyp-NCut (with binning).
Direct GPU edge sampling, temperature annealing, profiling.
"""

import sys, time, json, numpy as np, torch, scipy.sparse as sp
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from torch import nn
from embedata import load_embeddings
from pgcuts.hyp2f1.autograd import Hyp2F1
from pgcuts.losses.pncut import log_kmeans_bins, compute_ncut_bin_phi
from pgcuts.losses.flashcut import flashcut_rcut
from pgcuts.metrics import evaluate_clustering, compute_rcut_ncut
from pgcuts.optim import GradientMixer
from pgcuts.graph import build_rbf_knn_graph

_h = Hyp2F1.apply

# ── Config ────────────────────────────────────────────────────────────
REPR_DIR = "/buckets/representations"
DATASETS = ['aircraft']
MODELS = ['dinov2', 'clipvitL14']
OBJECTIVES = ['hyp_ncut']

batch_size = 1024 * 8
steps = 1500
lr = 1e-3
wd = 0.1
ema = 0.9
m = 512
tau_start = 10.0
tau_end = 1.0
knn = 50
num_bins = 16
eval_every = 500
device = torch.device("cuda:0")


def setup_data(dataset_name, model_name):
    """Load embeddings + build graph. Returns dict with all tensors."""
    ds = load_embeddings(dataset_name, model_name, REPR_DIR, split="train")
    X = ds.feats.astype(np.float32)
    y = ds.labels
    unique = np.unique(y)
    lmap = {o: n for n, o in enumerate(unique)}
    y = np.array([lmap[l] for l in y])
    K = len(unique)
    N, D = X.shape

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    W_exp = build_rbf_knn_graph(X, n_neighbors=min(knn, N - 1))
    degrees = np.array(W_exp.sum(1)).flatten().astype(np.float32)

    # Preload edges + weights on GPU
    pairs = np.array(W_exp.nonzero()).T
    n_edges = pairs.shape[0]
    edge_w = np.array(W_exp[pairs[:, 0], pairs[:, 1]]).flatten().astype(np.float32)

    pairs_gpu = torch.tensor(pairs, dtype=torch.long, device=device)
    edge_w_gpu = torch.tensor(edge_w, device=device)
    degrees_t = torch.tensor(degrees, dtype=torch.float32, device=device)

    # Binning for NCut
    bins = log_kmeans_bins(degrees, num_bins)
    n_bins = len(bins)
    bin_weights_t = torch.tensor(
        [b["count"] for b in bins], dtype=torch.float32, device=device
    )
    bin_weights_t = bin_weights_t / bin_weights_t.sum()
    bin_idx = [
        torch.tensor(b["indices"], dtype=torch.long, device=device)
        for b in bins
    ]
    beta_stars = torch.tensor(
        [b["beta_star"] for b in bins], dtype=torch.float32, device=device
    )
    q_stars = torch.tensor(
        [degrees[b["indices"]].mean() for b in bins],
        dtype=torch.float32,
        device=device,
    )

    # Node-to-bin mapping for edge lookup
    node_to_bin = np.zeros(N, dtype=np.int64)
    for j, b in enumerate(bins):
        node_to_bin[b["indices"]] = j
    node_to_bin_gpu = torch.tensor(node_to_bin, dtype=torch.long, device=device)

    return dict(
        X_t=X_t,
        y=y,
        K=K,
        N=N,
        D=D,
        W_exp=W_exp,
        pairs_gpu=pairs_gpu,
        edge_w_gpu=edge_w_gpu,
        n_edges=n_edges,
        degrees_t=degrees_t,
        n_bins=n_bins,
        bin_weights_t=bin_weights_t,
        bin_idx=bin_idx,
        beta_stars=beta_stars,
        q_stars=q_stars,
        node_to_bin_gpu=node_to_bin_gpu,
    )


def run(objective, data):
    K, D, N = data["K"], data["D"], data["N"]
    X_t, y = data["X_t"], data["y"]
    pairs_gpu, edge_w_gpu, n_edges = (
        data["pairs_gpu"],
        data["edge_w_gpu"],
        data["n_edges"],
    )
    degrees_t = data["degrees_t"]
    n_bins = data["n_bins"]
    bin_weights_t, bin_idx, beta_stars = (
        data["bin_weights_t"],
        data["bin_idx"],
        data["beta_stars"],
    )
    q_stars, node_to_bin_gpu = data["q_stars"], data["node_to_bin_gpu"]

    torch.manual_seed(42)
    hidden = 1024

    model = nn.Sequential(
        nn.Linear(D, hidden, bias=False),
        nn.SiLU(),
        nn.Linear(hidden, K, bias=False),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=1e-5
    )
    gm = GradientMixer(
        list(model.named_parameters()), loss_scale={"cut": 1.0, "balance": 1.0}
    )

    p_ema = torch.ones(K, device=device) / K
    alpha_ema = torch.ones(n_bins, K, device=device) / K

    timers = {
        "sample": 0.0,
        "forward": 0.0,
        "loss": 0.0,
        "backward": 0.0,
        "eval": 0.0,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        tau = tau_start + (tau_end - tau_start) * (step / steps)

        # ── Sample ────────────────────────────────────────────
        ts = time.time()
        idx = torch.randint(n_edges, (batch_size,), device=device)
        bp = pairs_gpu[idx]
        w = edge_w_gpu[idx]
        all_nodes = torch.cat([bp[:, 0], bp[:, 1]])
        unique_nodes, inv = torch.unique(all_nodes, return_inverse=True)
        left_idx = inv[:batch_size]
        right_idx = inv[batch_size:]
        timers["sample"] += time.time() - ts

        # ── Forward ───────────────────────────────────────────
        tf = time.time()
        logits = model(X_t[unique_nodes]) / tau
        P = torch.softmax(logits, dim=-1)
        P_left = P[left_idx]
        P_right = P[right_idx]
        timers["forward"] += time.time() - tf

        # ── Loss ──────────────────────────────────────────────
        tl = time.time()
        if objective == "prcut":
            cut_per_k = (w.unsqueeze(-1) * P_left * (1.0 - P_right)).mean(0)
            p_bar = ema * p_ema + (1 - ema) * P.mean(0).detach()
            cut_loss = (cut_per_k / (p_bar + 1e-12)).sum() / w.sum()

        elif objective == "hyp_rcut":
            log_P_right = torch.log_softmax(logits[right_idx], dim=-1)
            hycut_per_k = (w.unsqueeze(-1) * (-P_left * log_P_right)).mean(0)
            alpha = ema * p_ema + (1 - ema) * P.mean(0).detach()
            z = alpha.clamp(1e-7, 1 - 1e-7)
            H = _h(
                -m,
                torch.tensor(1.0, device=device),
                torch.tensor(2.0, device=device),
                z,
            )
            cut_loss = (hycut_per_k * H).sum() / w.sum()

        elif objective == "flash_rcut":
            log_P_right = torch.log_softmax(logits[right_idx], dim=-1)
            alpha = ema * p_ema + (1 - ema) * P.mean(0).detach()
            cut_per_k = flashcut_rcut(P_left, log_P_right, w, alpha, m)
            cut_loss = cut_per_k.sum() / w.sum()

        elif objective == "hyp_ncut":
            log_P_right = torch.log_softmax(logits[right_idx], dim=-1)
            hycut_per_edge = w.unsqueeze(-1) * (-P_left * log_P_right)  # (E, K)
            # Phi is (d, K) without 1/q — lookup by left node's bin
            Phi_bins = compute_ncut_bin_phi(
                q_stars, alpha_ema, beta_stars, bin_weights_t, m
            )  # (d, K)
            left_bins = node_to_bin_gpu[bp[:, 0]]  # (E,)
            Phi_edges = Phi_bins[left_bins]  # (E, K)
            # Apply 1/degree per edge (the factored-out 1/q)
            deg_left = degrees_t[bp[:, 0]].clamp(min=1e-6).unsqueeze(-1)  # (E, 1)
            cut_loss = (hycut_per_edge * Phi_edges / deg_left).sum() / w.sum()

        balance = -torch.special.entr(P.mean(0)).sum()
        timers["loss"] += time.time() - tl

        # ── Backward ──────────────────────────────────────────
        tb = time.time()
        with torch.no_grad():
            p_ema.mul_(ema).add_(P.mean(0).detach() * (1 - ema))
            for j, ix in enumerate(bin_idx):
                mask = torch.isin(unique_nodes, ix)
                if mask.any():
                    alpha_ema[j] = ema * alpha_ema[j] + (1 - ema) * P.detach()[
                        mask
                    ].mean(0)

        opt.zero_grad()
        with gm("cut"):
            cut_loss.backward(retain_graph=True)
        with gm("balance"):
            balance.backward()
        opt.step()
        sched.step()
        timers["backward"] += time.time() - tb

        # ── Eval ──────────────────────────────────────────────
        if step % eval_every == 0 or step == 1:
            te = time.time()
            with torch.no_grad():
                lo = model(X_t)
                pred = lo.argmax(dim=-1).cpu().numpy()
                res = evaluate_clustering(y, pred, K)
                sizes = np.bincount(pred, minlength=K)
                P_eval = torch.softmax(lo / tau, dim=-1)
                P_max = P_eval.max(dim=-1).values.mean().item()
                H_ps = -torch.sum(P_eval * torch.log(P_eval + 1e-12), dim=-1)
                sharp = (np.log(K) - H_ps.mean().item()) / np.log(K)
                print(
                    f"  [{objective}] {step:>5}/{steps}: "
                    f'acc={res["accuracy"]:.4f} nmi={res["nmi"]:.4f} '
                    f"empty={(sizes==0).sum():>2} sharp={sharp:.3f} tau={tau:.2f} "
                    f"({time.time()-t0:.0f}s)"
                )
            timers["eval"] += time.time() - te

    # Profile
    total = sum(timers.values())
    print(f"  Profile ({total:.1f}s):  ", end="")
    for k, v in sorted(timers.items(), key=lambda x: -x[1]):
        print(f"{k}={v:.1f}s({100*v/total:.0f}%) ", end="")
    print()

    # Final
    with torch.no_grad():
        lo = model(X_t)
        pred = lo.argmax(dim=-1).cpu().numpy()
        res = evaluate_clustering(y, pred, K)
        rcut, ncut = compute_rcut_ncut(data["W_exp"], pred)

    return {
        "accuracy": res["accuracy"],
        "nmi": res["nmi"],
        "rcut": rcut,
        "ncut": ncut,
    }


# ── Sweep ─────────────────────────────────────────────────────────────
all_results = {}

for ds_name in DATASETS:
    for model_name in MODELS:
        tag = f"{ds_name}/{model_name}"
        emb_path = Path(REPR_DIR) / ds_name / model_name / "feats_train.npy"
        if not emb_path.exists():
            print(f"\n--- SKIP {tag} (not found) ---")
            continue

        print(f'\n{"="*60}\n  {tag}\n{"="*60}')
        try:
            data = setup_data(ds_name, model_name)
            print(
                f'  N={data["N"]}, K={data["K"]}, D={data["D"]}, edges={data["n_edges"]}'
            )
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        all_results[tag] = {}
        for obj in OBJECTIVES:
            res = run(obj, data)
            all_results[tag][obj] = res
            print(
                f'  => {obj}: acc={res["accuracy"]:.4f} nmi={res["nmi"]:.4f} '
                f'rcut={res["rcut"]:.4f} ncut={res["ncut"]:.4f}'
            )

        # Free GPU memory between datasets
        del data
        torch.cuda.empty_cache()

# ── Summary table ─────────────────────────────────────────────────────
print(f'\n{"="*80}')
print(
    f'{"dataset/model":<30} {"objective":<12} {"acc":>8} {"nmi":>8} {"rcut":>8} {"ncut":>8}'
)
print(f'{"-"*80}')
for tag, objs in all_results.items():
    for obj, res in objs.items():
        print(
            f'{tag:<30} {obj:<12} {res["accuracy"]:>8.4f} {res["nmi"]:>8.4f} '
            f'{res["rcut"]:>8.4f} {res["ncut"]:>8.4f}'
        )

# Save results
with open("hycut_linear_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to hycut_linear_results.json")
