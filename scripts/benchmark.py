"""Reproducible PGCut benchmark via Hydra.

Usage:
    # Single run
    python scripts/benchmark.py dataset=cifar10 model=dinov2 objective=hyp_ncut

    # Full sweep (all datasets × all models × all objectives)
    python scripts/benchmark.py --multirun \
        dataset=cifar10,cifar100,stl10,aircraft,eurosat,dtd,flowers,pets,food101,gtsrb,fashionmnist,mnist,imagenette,cub,resisc45 \
        model=clipvitL14,dinov2,dinov3b \
        objective=prcut,hyp_rcut,hyp_ncut
"""
import sys, time, json, logging
import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch import nn
from embedata import load_embeddings
from pgcuts.algorithms import prcut_original_step, prcut_step, hyp_rcut_step, hyp_ncut_step
from pgcuts.losses.pncut import log_kmeans_bins
from pgcuts.metrics import evaluate_clustering, compute_rcut_ncut
from pgcuts.optim import GradientMixer
from pgcuts.graph import build_rbf_knn_graph

log = logging.getLogger(__name__)


def graph_quality(W, y):
    W = sp.csr_matrix(W)
    W.setdiag(0)
    W.eliminate_zeros()
    n = W.shape[0]; K = int(y.max()) + 1
    deg = np.maximum(np.array(W.sum(1)).flatten(), 1e-12)
    T = sp.diags(1.0 / deg) @ W
    oh = sp.csr_matrix((np.ones(n), (np.arange(n), y)), shape=(n, K))
    q = np.array((T @ oh)[np.arange(n), y]).flatten().mean()
    q_ch = np.sum((np.bincount(y, minlength=K) / n) ** 2)
    return (q - q_ch) / (1 - q_ch) if q_ch < 1 else 0.0


@hydra.main(config_path="configs", config_name="benchmark", version_base=None)
def main(cfg: DictConfig):
    tag = f"{cfg.dataset}/{cfg.model}/{cfg.objective}"
    log.info(f"Starting: {tag}")

    # Auto-assign GPU: hash the tag to distribute across available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        gpu_id = hash(tag) % n_gpus
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device(cfg.device)

    # ── Load data ─────────────────────────────────────────────
    try:
        ds = load_embeddings(cfg.dataset, cfg.model, cfg.repr_dir, split="train")
    except FileNotFoundError:
        log.warning(f"SKIP {tag} — not found")
        return {"status": "skipped"}

    X = ds.feats.astype(np.float32)
    y = ds.labels
    unique = np.unique(y)
    lmap = {o: n for n, o in enumerate(unique)}
    y = np.array([lmap[l] for l in y])
    K = len(unique)
    N, D = X.shape

    # Subsample large datasets
    if N > 80000:
        rng = np.random.RandomState(cfg.seed)
        idx = []
        for k in range(K):
            cls_idx = np.where(y == k)[0]
            n_take = max(1, int(80000 * len(cls_idx) / N))
            idx.append(rng.choice(cls_idx, n_take, replace=False))
        idx = np.concatenate(idx)
        X, y = X[idx], y[idx]
        N = len(y)
        log.info(f"Subsampled to N={N}")

    log.info(f"N={N}, K={K}, D={D}")

    # ── Build graph ───────────────────────────────────────────
    W_exp = build_rbf_knn_graph(X, n_neighbors=min(cfg.knn, N - 1))
    degrees = np.array(W_exp.sum(1)).flatten().astype(np.float32)
    quality = graph_quality(W_exp, y)
    log.info(f"Graph: edges={W_exp.nnz}, quality={quality:.4f}")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    # Edges on GPU
    pairs = np.array(W_exp.nonzero()).T
    n_edges = pairs.shape[0]
    edge_w = np.array(W_exp[pairs[:, 0], pairs[:, 1]]).flatten().astype(np.float32)
    pairs_gpu = torch.tensor(pairs, dtype=torch.long, device=device)
    edge_w_gpu = torch.tensor(edge_w, device=device)
    degrees_t = torch.tensor(degrees, dtype=torch.float32, device=device)

    # Binning
    bins = log_kmeans_bins(degrees, cfg.num_bins)
    n_bins = len(bins)
    bin_weights_t = torch.tensor([b["count"] for b in bins], dtype=torch.float32, device=device)
    bin_weights_t = bin_weights_t / bin_weights_t.sum()
    beta_stars = torch.tensor([b["beta_star"] for b in bins], dtype=torch.float32, device=device)
    q_stars = torch.tensor(
        [degrees[b["indices"]].mean() for b in bins], dtype=torch.float32, device=device
    )
    node_to_bin = torch.zeros(N, dtype=torch.long, device=device)
    for j, b in enumerate(bins):
        node_to_bin[torch.tensor(b["indices"], dtype=torch.long, device=device)] = j

    # ── Model ─────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    model = nn.Linear(D, K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=1e-5)

    gm = GradientMixer(list(model.named_parameters()), loss_scale={"cut": 1.0, "balance": 1.0})
    p_ema = torch.ones(K, device=device) / K
    alpha_ema = torch.ones(n_bins, K, device=device) / K

    # ── Train ─────────────────────────────────────────────────
    t0 = time.time()
    best_acc = 0.0
    best_res = {}

    for step in range(1, cfg.steps + 1):
        tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * (step / cfg.steps)

        # Sample edges
        idx = torch.randint(n_edges, (cfg.batch_size,), device=device)
        bp = pairs_gpu[idx]
        w = edge_w_gpu[idx]
        all_nodes = torch.cat([bp[:, 0], bp[:, 1]])
        unique_nodes, inv = torch.unique(all_nodes, return_inverse=True)
        left_idx = inv[: cfg.batch_size]
        right_idx = inv[cfg.batch_size :]

        logits = model(X_t[unique_nodes]) / tau

        # Compute loss
        if cfg.objective == "prcut_original":
            P = torch.softmax(logits, dim=-1)
            surrogate, p_ema = prcut_original_step(
                P, left_idx, right_idx, w, p_ema, N, cfg.ema
            )
            opt.zero_grad()
            surrogate.backward()
            opt.step()
            sched.step()
            cut_loss = surrogate  # for logging
        else:
            if cfg.objective == "prcut":
                P = torch.softmax(logits, dim=-1)
                cut_loss, balance, p_ema = prcut_step(
                    P, left_idx, right_idx, w, p_ema, cfg.ema
                )
            elif cfg.objective == "hyp_rcut":
                cut_loss, balance, p_ema = hyp_rcut_step(
                    logits, left_idx, right_idx, w, p_ema, cfg.m, cfg.ema
                )
            elif cfg.objective == "hyp_ncut":
                cut_loss, balance, alpha_ema = hyp_ncut_step(
                    logits, left_idx, right_idx, w, bp[:, 0],
                    alpha_ema, q_stars, beta_stars, bin_weights_t,
                    node_to_bin, degrees_t, cfg.m, cfg.ema,
                )
            else:
                raise ValueError(f"Unknown objective: {cfg.objective}")

            opt.zero_grad()
            with gm("cut"):
                cut_loss.backward(retain_graph=True)
            with gm("balance"):
                balance.backward()
            opt.step()
            sched.step()

        # Eval
        if step % cfg.eval_every == 0 or step == 1 or step == cfg.steps:
            with torch.no_grad():
                lo = model(X_t)
                pred = lo.argmax(dim=-1).cpu().numpy()
                res = evaluate_clustering(y, pred, K)
                sizes = np.bincount(pred, minlength=K)
                P_eval = torch.softmax(lo / tau, dim=-1)
                H_ps = -torch.sum(P_eval * torch.log(P_eval + 1e-12), dim=-1)
                sharp = (np.log(K) - H_ps.mean().item()) / np.log(K)

                log.info(
                    f"[{step:>5}/{cfg.steps}] acc={res['accuracy']:.4f} nmi={res['nmi']:.4f} "
                    f"empty={(sizes==0).sum()} sharp={sharp:.3f} tau={tau:.2f}"
                )

                if res["accuracy"] > best_acc:
                    best_acc = res["accuracy"]
                    rcut, ncut = compute_rcut_ncut(W_exp, pred)
                    best_res = {
                        "dataset": cfg.dataset,
                        "model": cfg.model,
                        "objective": cfg.objective,
                        "accuracy": float(res["accuracy"]),
                        "nmi": float(res["nmi"]),
                        "rcut": float(rcut),
                        "ncut": float(ncut),
                        "quality": float(quality),
                        "N": N,
                        "K": K,
                        "D": D,
                        "step": step,
                        "time": time.time() - t0,
                        "empty": int((sizes == 0).sum()),
                    }

    # ── Save ──────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{cfg.dataset}_{cfg.model}_{cfg.objective}.json"
    with open(result_file, "w") as f:
        json.dump(best_res, f, indent=2)

    log.info(f"DONE: acc={best_res['accuracy']:.4f} nmi={best_res['nmi']:.4f} → {result_file}")
    return best_res


if __name__ == "__main__":
    main()
