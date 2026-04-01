"""Online HyCut benchmark — batch-based similarity.

Instead of precomputing a global KNN graph, builds a similarity
graph within each batch on-the-fly. Scales to any dataset size.

Usage:
    python scripts/benchmark_online.py dataset=cifar10 model=dinov2
"""
import sys
import time
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedata import load_embeddings
from pgcuts.similarity import batch_knn_graph
from pgcuts.losses.pncut import log_kmeans_bins, compute_ncut_bin_phi
from pgcuts.metrics import evaluate_clustering
from pgcuts.optim import GradientMixer
from pgcuts.hyp2f1.autograd import Hyp2F1

log = logging.getLogger(__name__)
_h = Hyp2F1.apply


@hydra.main(
    config_path="configs",
    config_name="benchmark_online",
    version_base=None,
)
def main(cfg: DictConfig):
    """Run online HyCut benchmark."""
    tag = f"{cfg.dataset}/{cfg.model}/{cfg.objective}"
    log.info("Starting: %s", tag)
    device = torch.device(cfg.device)

    # ── Load data ─────────────────────────────────────────
    ds = load_embeddings(
        cfg.dataset, cfg.model, cfg.repr_dir, split="train"
    )
    # Keep on CPU — only move batches to GPU
    feats_cpu = ds.feats  # numpy array, may be mmap
    labels = ds.labels
    unique = np.unique(labels)
    label_map = {o: n for n, o in enumerate(unique)}
    labels = np.array([label_map[l] for l in labels])
    num_clusters = len(unique)
    num_samples, num_dims = feats_cpu.shape
    log.info("N=%d, K=%d, D=%d", num_samples, num_clusters, num_dims)

    # ── Model ─────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    model = nn.Linear(num_dims, num_clusters).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.steps, eta_min=1e-5
    )
    gm = GradientMixer(
        list(model.named_parameters()),
        loss_scale={"cut": 1.0, "balance": 1.0},
    )

    # ── EMA state ─────────────────────────────────────────
    p_ema = torch.ones(num_clusters, device=device) / num_clusters
    # For NCut: online degree/bin tracking (CPU to avoid OOM)
    degree_ema = torch.ones(num_samples)
    num_bins = cfg.num_bins
    # Initialize bins from uniform degrees
    bins_initialized = False
    q_stars = None
    beta_stars = None
    bin_weights = None
    node_to_bin = None
    alpha_ema = None

    # ── Quality tracking ──────────────────────────────────
    quality_ema = 0.0

    # ── Train ─────────────────────────────────────────────
    t_start = time.time()
    best_acc = 0.0
    best_res = {}

    for step in range(1, cfg.steps + 1):
        tau = cfg.tau_start + (
            cfg.tau_end - cfg.tau_start
        ) * (step / cfg.steps)

        # Sample batch of nodes — load from CPU to GPU
        batch_idx_np = np.random.randint(
            num_samples, size=cfg.batch_size
        )
        x_batch = torch.tensor(
            feats_cpu[batch_idx_np].astype(np.float32),
            device=device,
        )
        batch_idx = torch.tensor(
            batch_idx_np, dtype=torch.long, device=device
        )

        # Build batch KNN graph on-the-fly
        src, dst, w = batch_knn_graph(
            x_batch, k=cfg.knn, sigma=cfg.sigma
        )

        # Update degree EMA from batch graph
        with torch.no_grad():
            batch_degrees = torch.zeros(
                cfg.batch_size, device=device
            )
            batch_degrees.scatter_add_(0, src, w)
            # Map back to global node degrees (CPU)
            bd_cpu = batch_degrees.cpu()
            degree_ema[batch_idx_np] = (
                cfg.ema * degree_ema[batch_idx_np]
                + (1 - cfg.ema) * bd_cpu
            )

        # Batch quality tracking (same-class edge fraction)
        with torch.no_grad():
            y_batch = labels[batch_idx_np]
            same_class = (
                y_batch[src.cpu().numpy()]
                == y_batch[dst.cpu().numpy()]
            )
            q_raw = same_class.mean()
            counts = np.bincount(
                y_batch, minlength=num_clusters
            )
            q_chance = np.sum(
                (counts / len(y_batch)) ** 2
            )
            q_norm = (
                (q_raw - q_chance) / (1 - q_chance)
                if q_chance < 1
                else 0.0
            )
            quality_ema = (
                0.99 * quality_ema + 0.01 * q_norm
            )

        # Forward
        logits = model(x_batch) / tau
        probs = torch.softmax(logits, dim=-1)

        # ── Compute loss ──────────────────────────────────
        p_left = probs[src]

        if cfg.objective in ("hyp_rcut", "hyp_ncut"):
            import torch.nn.functional as F  # pylint: disable=import-outside-toplevel
            log_p_right = F.log_softmax(
                logits[dst], dim=-1
            )
            cut_per_edge = w.unsqueeze(-1) * (
                -p_left * log_p_right
            )
        else:  # prcut
            p_right = probs[dst]
            cut_per_edge = w.unsqueeze(-1) * (
                p_left * (1.0 - p_right)
            )

        if cfg.objective == "prcut":
            p_bar = (
                cfg.ema * p_ema
                + (1 - cfg.ema) * probs.mean(0).detach()
            )
            cut_per_k = cut_per_edge.mean(0)
            cut_loss = (
                (cut_per_k / (p_bar + 1e-12)).sum()
                / w.sum()
            )

        elif cfg.objective == "hyp_rcut":
            cut_per_k = cut_per_edge.mean(0)
            alpha = (
                cfg.ema * p_ema
                + (1 - cfg.ema) * probs.mean(0)
            )
            z = alpha.clamp(1e-7, 1 - 1e-7)
            h_val = _h(
                -cfg.m,
                torch.tensor(1.0, device=device),
                torch.tensor(2.0, device=device),
                z,
            )
            cut_loss = (
                (cut_per_k * h_val).sum() / w.sum()
            )

        elif cfg.objective == "hyp_ncut":
            # Update bins from degree EMA periodically
            if not bins_initialized or step % 100 == 0:
                deg_np = degree_ema.numpy()
                bins = log_kmeans_bins(deg_np, num_bins)
                n_bins = len(bins)
                q_stars = torch.tensor(
                    [deg_np[b["indices"]].mean() for b in bins],
                    dtype=torch.float32,
                    device=device,
                )
                beta_stars = torch.tensor(
                    [b["beta_star"] for b in bins],
                    dtype=torch.float32,
                    device=device,
                )
                bw = torch.tensor(
                    [b["count"] for b in bins],
                    dtype=torch.float32,
                    device=device,
                )
                bin_weights = bw / bw.sum()
                ntb = torch.zeros(
                    num_samples,
                    dtype=torch.long,
                    device=device,
                )
                for j, b in enumerate(bins):
                    ntb[
                        torch.tensor(
                            b["indices"],
                            dtype=torch.long,
                            device=device,
                        )
                    ] = j
                node_to_bin = ntb
                if alpha_ema is None or n_bins != alpha_ema.shape[0]:
                    alpha_ema = (
                        torch.ones(
                            n_bins,
                            num_clusters,
                            device=device,
                        )
                        / num_clusters
                    )
                bins_initialized = True

            # Live alpha
            alpha_live = (
                cfg.ema * alpha_ema
                + (1 - cfg.ema) * probs.mean(0).unsqueeze(0)
            )
            phi_bins = compute_ncut_bin_phi(
                q_stars,
                alpha_live,
                beta_stars,
                bin_weights,
                cfg.m,
            )
            left_bins = node_to_bin[batch_idx[src]]
            phi_edges = phi_bins[left_bins]
            deg_left = (
                degree_ema[batch_idx_np[src.cpu().numpy()]]
                .clamp(min=1e-6)
                .unsqueeze(-1)
                .to(device)
            )
            cut_loss = (
                (cut_per_edge * phi_edges / deg_left).sum()
                / w.sum()
            )

        balance = -torch.special.entr(  # pylint: disable=not-callable
            probs.mean(0)
        ).sum()

        # ── Update ────────────────────────────────────────
        opt.zero_grad()
        with gm("cut"):
            cut_loss.backward(retain_graph=True)
        with gm("balance"):
            balance.backward()
        opt.step()
        sched.step()

        # EMA updates
        with torch.no_grad():
            p_ema = (
                cfg.ema * p_ema
                + (1 - cfg.ema) * probs.mean(0).detach()
            ).detach()
            if cfg.objective == "hyp_ncut" and bins_initialized:
                alpha_ema = (
                    cfg.ema * alpha_ema
                    + (1 - cfg.ema)
                    * probs.detach().mean(0).unsqueeze(0)
                ).detach()

        # ── Eval ──────────────────────────────────────────
        if (
            step % cfg.eval_every == 0
            or step == 1
            or step == cfg.steps
        ):
            with torch.no_grad():
                # Batched eval for large datasets
                all_preds = []
                sharp_sum = 0.0
                eval_bs = 8192
                for start in range(
                    0, num_samples, eval_bs
                ):
                    end = min(start + eval_bs, num_samples)
                    x_eval = torch.tensor(
                        feats_cpu[start:end].astype(
                            np.float32
                        ),
                        device=device,
                    )
                    lo = model(x_eval)
                    all_preds.append(
                        lo.argmax(dim=-1).cpu().numpy()
                    )
                    p_eval = torch.softmax(
                        lo / tau, dim=-1
                    )
                    h_ps = -torch.sum(
                        p_eval
                        * torch.log(p_eval + 1e-12),
                        dim=-1,
                    )
                    sharp_sum += h_ps.sum().item()
                pred = np.concatenate(all_preds)
                res = evaluate_clustering(
                    labels, pred, num_clusters
                )
                sizes = np.bincount(
                    pred, minlength=num_clusters
                )
                sharp = (
                    np.log(num_clusters)
                    - sharp_sum / num_samples
                ) / np.log(num_clusters)

                log.info(
                    "[%5d/%d] acc=%.4f nmi=%.4f "
                    "empty=%d sharp=%.3f Q=%.3f",
                    step,
                    cfg.steps,
                    res["accuracy"],
                    res["nmi"],
                    (sizes == 0).sum(),
                    sharp,
                    quality_ema,
                )

                if res["accuracy"] > best_acc:
                    best_acc = res["accuracy"]
                    best_res = {
                        "dataset": cfg.dataset,
                        "model": cfg.model,
                        "objective": cfg.objective,
                        "accuracy": float(res["accuracy"]),
                        "nmi": float(res["nmi"]),
                        "quality_ema": float(quality_ema),
                        "N": num_samples,
                        "K": num_clusters,
                        "step": step,
                        "time": time.time() - t_start,
                        "empty": int((sizes == 0).sum()),
                    }

    # ── Save ──────────────────────────────────────────────
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = (
        out_dir
        / f"{cfg.dataset}_{cfg.model}_{cfg.objective}_online.json"
    )
    with open(result_file, "w") as f:
        json.dump(best_res, f, indent=2)

    log.info(
        "DONE: acc=%.4f nmi=%.4f Q=%.3f -> %s",
        best_res.get("accuracy", 0),
        best_res.get("nmi", 0),
        best_res.get("quality_ema", 0),
        result_file,
    )
    return best_res


if __name__ == "__main__":
    main()
