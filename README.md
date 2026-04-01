# PGCuts: Probabilistic Graph Cuts

Differentiable graph cut objectives for unsupervised clustering on pre-computed embeddings. PGCuts provides tight hypergeometric upper bounds on the expected Normalized Cut, enabling gradient-based optimization without spectral decomposition.

**Paper:** [Beyond Spectral Clustering: Probabilistic Cuts for Differentiable Graph Partitioning](https://github.com/ayghri/pgcuts/blob/main/aistats2026_paper.pdf) (AISTATS 2026)

**Docs:** [pgcuts.readthedocs.io](https://pgcuts.readthedocs.io)

## Quick Start

```python
from pgcuts import HyCut

# Like sklearn
labels = HyCut(n_clusters=10).fit_predict(X)
```

Works on any pre-computed embeddings (DINOv2, CLIP, etc.):

```python
import numpy as np
from pgcuts import HyCut

X = np.load("dinov2_embeddings.npy")  # (50000, 1536)
model = HyCut(n_clusters=100, objective="hyp_ncut")
labels = model.fit_predict(X)

print(model.ncut_)  # normalized cut value
print(model.rcut_)  # ratio cut value
```

## Installation

```bash
pip install pgcuts
```

Or from source:

```bash
git clone https://github.com/ayghri/pgcuts.git
cd pgcuts && pip install -e .
```

Requires Python >= 3.11, PyTorch >= 2.8 with CUDA, and Triton.

## Algorithms

Three probabilistic graph cut objectives:

| Objective | Envelope | What it optimizes |
|-----------|----------|-------------------|
| **PRCut** | `1/p_bar` | Expected Ratio Cut / cluster size |
| **H-RCut** | `2F1(-m, 1; 2; alpha)` | Hypergeometric bound on expected Ratio Cut |
| **H-NCut** | Holder-binned `2F1` per degree bin | Hypergeometric bound on expected Normalized Cut |

For custom training loops, use the step functions directly:

```python
from pgcuts.algorithms import prcut_step, hyp_rcut_step, hyp_ncut_step
```

Each takes a batch of edges and returns `(cut_loss, balance_loss, updated_ema)`.

## Key Components

**Graph Construction**

```python
from pgcuts.graph import build_rbf_knn_graph
W = build_rbf_knn_graph(X, n_neighbors=50)
```

**Hypergeometric Function** (GPU-accelerated via Triton/CUDA)

```python
from pgcuts import Hyp2F1
H = Hyp2F1.apply(-512, 1.0, 2.0, z)  # differentiable w.r.t. z
```

**Gradient Mixer** (prevents cluster collapse)

```python
from pgcuts import GradientMixer

gm = GradientMixer(model.named_parameters(), loss_scale={"cut": 1.0, "balance": 1.0})
with gm("cut"):
    cut_loss.backward(retain_graph=True)
with gm("balance"):
    balance.backward()
```

**Evaluation**

```python
from pgcuts import evaluate_clustering, compute_rcut_ncut

results = evaluate_clustering(y_true, y_pred, K)
rcut, ncut = compute_rcut_ncut(W, y_pred)
```

## Reproducing Results

```bash
pip install "pgcuts[experiments]"

python scripts/benchmark.py --multirun \
    dataset=cifar10,cifar100,stl10,eurosat,imagenette,fashionmnist,mnist,pets,flowers,food101,resisc45,dtd,gtsrb,cub,aircraft \
    model=dinov2,dinov3b,clipvitL14 \
    objective=prcut,hyp_rcut,hyp_ncut
```

Results saved to `results/` as JSON. Embeddings loaded via [embedata](https://github.com/ayghri/embedata).

## Package Structure

```
pgcuts/
    cluster.py      # HyCut (sklearn-compatible API)
    algorithms/     # prcut_step, hyp_rcut_step, hyp_ncut_step
    hyp2f1/         # GPU 2F1 kernels (Triton + CUDA)
    losses/         # PRCut, H-RCut, H-NCut, FlashCut
    graph.py        # KNN graph construction
    metrics.py      # ACC, NMI, ARI, RCut, NCut
    optim.py        # GradientMixer
    utils/          # Edge sampling, data utilities
```

## Citation

```bibtex
@inproceedings{ghriss2026pgcuts,
    title={Beyond Spectral Clustering: Probabilistic Cuts for Differentiable Graph Partitioning},
    author={Ghriss, Ayoub},
    booktitle={The 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
    year={2026}
}
```

## License

MIT
