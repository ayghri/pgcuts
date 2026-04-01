"""Benchmark PRCut, H-RCut, H-NCut on all datasets in /buckets/representations/.

Non-parametric PGCut: free logits + GradientMixer + entropy balance.
Three cut objectives compared:
  - PRCut: 1/alpha_bar bound (simple ratio)
  - H-RCut: 2F1(-m,1;2;alpha_bar) bound (Theorem 1, beta=1)
  - H-NCut: Holder-binned 2F1 bound (Theorem 2, per-vertex degree)

Uses 2 GPUs in parallel via spawn multiprocessing.
"""

import sys
import os
import time
import json
import numpy as np
import torch
import scipy.sparse as sp
import multiprocessing as mp

sys.stdout.reconfigure(line_buffering=True)

REPR_DIR = "/buckets/representations/"
RESULTS_FILE = "outputs/bench_all_results.json"


def get_hparams(K, N):
    avg_class_size = N / K
    knn = min(100, max(5, int(avg_class_size / 2)))
    return {
        "knn": knn,
        "lr": 0.5,
        "wd": max(0.5 / K, 0.001),
        "bal": 1.0,  # GradientMixer balance weight (equal mixing)
        "epochs": 50000,
        "num_bins": 4,
        "m": 512,
    }


def run_pgcut(X, y, K, W_exp, degrees, hp, mode="hncut", device="cuda:0"):
    """Parametric PGCut: linear layer on embeddings + log-barrier.

    logits = X @ W_param + b_param
    mode: 'prcut' (1/alpha), 'hrcut' (Theorem 1), 'hncut' (Theorem 2)
    """
    from pgcuts.hyp2f1.autograd import Hyp2F1
    from pgcuts.losses.pncut import log_kmeans_bins
    from pgcuts.metrics import evaluate_clustering, compute_rcut_ncut
    from pgcuts.optim import GradientMixer
    _h = Hyp2F1.apply

    device = torch.device(device)
    N, D = X.shape

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    W_coo = sp.coo_matrix(W_exp)
    W_t = torch.sparse_coo_tensor(
        torch.tensor(np.vstack([W_coo.row, W_coo.col]), dtype=torch.long, device=device),
        torch.tensor(W_coo.data, dtype=torch.float32, device=device), W_coo.shape
    ).coalesce()
    W_sum = W_exp.sum()
    degrees_t = torch.tensor(degrees, dtype=torch.float32, device=device).clamp(min=1e-6)

    bins = log_kmeans_bins(degrees, hp["num_bins"])
    d = len(bins)
    counts = torch.tensor([b["count"] for b in bins], dtype=torch.float32, device=device)
    bin_weights = counts / counts.sum()
    bin_idx = [torch.tensor(b["indices"], dtype=torch.long, device=device) for b in bins]
    q_stars = torch.tensor([degrees[b["indices"]].mean() for b in bins],
                           dtype=torch.float32, device=device)
    beta_stars = torch.tensor([b["beta_star"] for b in bins],
                              dtype=torch.float32, device=device)
    c_per_bin = q_stars / beta_stars + 1.0
    m = hp["m"]

    # Linear parameterization
    W_param = torch.nn.Parameter(torch.randn(D, K, device=device) * 0.01)
    b_param = torch.nn.Parameter(torch.zeros(K, device=device))
    alpha_ema = torch.ones(d, K, device=device) / K

    optimizer = torch.optim.AdamW([W_param, b_param], lr=hp["lr"], weight_decay=hp["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp["epochs"], eta_min=hp["lr"] * 0.01)
    grad_mix = GradientMixer(
        [("W", W_param), ("b", b_param)], loss_scale={"cut": 1.0, "balance": hp["bal"]})

    t0 = time.time()
    for epoch in range(hp["epochs"]):
        logits = X_t @ W_param + b_param
        P = torch.softmax(logits, dim=-1)
        M = P * torch.sparse.mm(W_t, 1.0 - P)

        with torch.no_grad():
            for j, ix in enumerate(bin_idx):
                alpha_ema[j] = 0.9 * alpha_ema[j] + 0.1 * P[ix].mean(0)

        if mode == "prcut":
            alpha_bar = P.detach().mean(0).clamp(min=1e-7)
            U = (M.sum(0) / alpha_bar).sum() / W_sum
        elif mode == "hrcut":
            alpha_bar = alpha_ema[0].clamp(1e-7, 1 - 1e-7)
            H = _h(-m, 1.0, 2.0, alpha_bar)
            U = (M.sum(0) * H).sum() / W_sum
        elif mode == "hncut":
            z = alpha_ema.clamp(1e-7, 1 - 1e-7)
            F = _h(-m, 1.0, c_per_bin.unsqueeze(1).expand(d, K), z)
            log_Phi = (bin_weights.unsqueeze(1) * torch.log(F.clamp(min=1e-30))).sum(0)
            Phi = torch.exp(log_Phi).unsqueeze(0) / degrees_t.unsqueeze(1)
            U = (M * Phi).sum() / W_sum

        bal = -torch.special.entr(P.mean(0)).sum()

        optimizer.zero_grad()
        with grad_mix("cut"):
            U.backward(retain_graph=True)
        with grad_mix("balance"):
            bal.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10000 == 0 or epoch == hp["epochs"] - 1:
            with torch.no_grad():
                pred = (X_t @ W_param + b_param).argmax(dim=-1).cpu().numpy()
                res = evaluate_clustering(y, pred, K)
                rc, nc = compute_rcut_ncut(W_exp, pred)
                sizes = np.bincount(pred, minlength=K)
                print(f"      [{mode}] ep {epoch:>5}: ncut={nc:.4f} acc={res['accuracy']:.4f} "
                      f"empty={(sizes==0).sum()} ({time.time()-t0:.0f}s)", flush=True)

    pred = (X_t @ W_param + b_param).detach().argmax(dim=-1).cpu().numpy()
    return pred


def run_dataset(dataset, model, gpu_id, result_queue):
    """Run full benchmark for one dataset+model on a specific GPU."""
    from pgcuts.metrics import evaluate_clustering, compute_rcut_ncut
    from pgcuts.graph import build_rbf_knn_graph

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    feats_path = os.path.join(REPR_DIR, dataset, model, "feats_train.npy")
    labels_path = os.path.join(REPR_DIR, dataset, model, "y_train.npy")

    X = np.load(feats_path).astype(np.float32)
    y = np.load(labels_path)
    unique = np.unique(y)
    lmap = {old: new for new, old in enumerate(unique)}
    y = np.array([lmap[l] for l in y])
    K = len(unique)
    N = X.shape[0]

    hp = get_hparams(K, N)
    tag = f"{dataset}/{model}"
    print(f"\n[GPU{gpu_id}] === {tag}: N={N}, K={K}, knn={hp['knn']}, wd={hp['wd']:.4f} ===",
          flush=True)

    result = {"dataset": dataset, "model": model, "N": N, "K": K, "hparams": hp}

    # Build graph
    t0 = time.time()
    W_exp = build_rbf_knn_graph(X, n_neighbors=hp["knn"])
    degrees = np.array(W_exp.sum(axis=1)).flatten().astype(np.float32)
    result["graph_time"] = round(time.time() - t0, 1)
    print(f"[GPU{gpu_id}] {tag}: graph built ({result['graph_time']}s)", flush=True)

    # Run PRCut, H-RCut, H-NCut
    for mode in ["prcut", "hrcut", "hncut"]:
        t0 = time.time()
        try:
            pred = run_pgcut(X, y, K, W_exp, degrees, hp, mode=mode, device=device)
            res = evaluate_clustering(y, pred, K)
            rc, nc = compute_rcut_ncut(W_exp, pred)
            sizes = np.bincount(pred, minlength=K)
            result[mode] = {
                "accuracy": round(float(res["accuracy"]), 4),
                "nmi": round(float(res["nmi"]), 4),
                "ncut": round(float(nc), 4),
                "rcut": round(float(rc), 2),
                "empty": int((sizes == 0).sum()),
                "time": round(time.time() - t0, 1),
            }
            print(f"[GPU{gpu_id}] {tag}: {mode} acc={res['accuracy']:.4f} ncut={nc:.4f}",
                  flush=True)
        except Exception as e:
            import traceback
            result[mode] = {"error": str(e)}
            print(f"[GPU{gpu_id}] {tag}: {mode} FAILED: {e}", flush=True)
            traceback.print_exc()

    result_queue.put(result)


def gpu_worker(tasks, gpu_id, result_queue):
    """Process tasks sequentially on one GPU."""
    for dataset, model in tasks:
        try:
            run_dataset(dataset, model, gpu_id, result_queue)
        except Exception as e:
            print(f"[GPU{gpu_id}] {dataset}/{model}: CRASHED: {e}", flush=True)
            result_queue.put({"dataset": dataset, "model": model, "error": str(e)})


def main():
    os.makedirs("outputs", exist_ok=True)

    # Discover all dataset/model combos
    all_tasks = []
    for dataset in sorted(os.listdir(REPR_DIR)):
        ds_path = os.path.join(REPR_DIR, dataset)
        if not os.path.isdir(ds_path):
            continue
        for model in sorted(os.listdir(ds_path)):
            feats = os.path.join(ds_path, model, "feats_train.npy")
            if os.path.exists(feats):
                all_tasks.append((dataset, model))

    print(f"Found {len(all_tasks)} dataset/model combinations across 2 GPUs", flush=True)
    print(f"Methods: PRCut, H-RCut, H-NCut", flush=True)

    # Split across GPUs
    gpu0_tasks = all_tasks[0::2]
    gpu1_tasks = all_tasks[1::2]

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    p0 = ctx.Process(target=gpu_worker, args=(gpu0_tasks, 0, result_queue))
    p1 = ctx.Process(target=gpu_worker, args=(gpu1_tasks, 1, result_queue))
    p0.start()
    p1.start()

    all_results = []
    total = len(all_tasks)
    while len(all_results) < total:
        result = result_queue.get()
        all_results.append(result)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        ds = result.get("dataset", "?")
        model = result.get("model", "?")
        print(f"  [{len(all_results)}/{total}] {ds}/{model} saved", flush=True)

    p0.join()
    p1.join()

    # Summary
    print(f"\n{'='*100}", flush=True)
    hdr = (f"{'Dataset':15s} {'Model':12s} {'K':>4} {'N':>6} | "
           f"{'PR_acc':>7} {'HR_acc':>7} {'HN_acc':>7} | "
           f"{'PR_nmi':>7} {'HR_nmi':>7} {'HN_nmi':>7}")
    print(hdr, flush=True)
    print(f"{'-'*90}", flush=True)

    for r in sorted(all_results, key=lambda x: (x.get("dataset", ""), x.get("model", ""))):
        def fmt(d, key):
            v = d.get(key, "-") if isinstance(d, dict) else "-"
            return f"{v:.4f}" if isinstance(v, float) else "  err"

        pr = r.get("prcut", {})
        hr = r.get("hrcut", {})
        hn = r.get("hncut", {})

        print(f"{r.get('dataset','?'):15s} {r.get('model','?'):12s} "
              f"{r.get('K','?'):>4} {r.get('N','?'):>6} | "
              f"{fmt(pr,'accuracy'):>7} {fmt(hr,'accuracy'):>7} {fmt(hn,'accuracy'):>7} | "
              f"{fmt(pr,'nmi'):>7} {fmt(hr,'nmi'):>7} {fmt(hn,'nmi'):>7}",
              flush=True)

    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
