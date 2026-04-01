# PGCuts: Probabilistic Graph Cuts

Differentiable graph cut objectives for unsupervised clustering on pre-computed embeddings. PGCuts provides tight hypergeometric upper bounds on the expected Normalized Cut, enabling gradient-based optimization without spectral decomposition.

## Quick Start

```python
from pgcuts import HyCut

# Like sklearn — just pass features and number of clusters
model = HyCut(n_clusters=10)
labels = model.fit_predict(X)

print(model.ncut_)  # normalized cut value
print(model.rcut_)  # ratio cut value
```

Works on any pre-computed embeddings (CLIP, DINOv2, etc.):

```python
import numpy as np
from pgcuts import HyCut

X = np.load("dinov2_embeddings.npy")  # (50000, 1536)
labels = HyCut(n_clusters=100, objective="hyp_ncut").fit_predict(X)
```

For full control over the optimization, see `scripts/benchmark.py` or use the
step functions directly:

```python
from pgcuts.algorithms import prcut_step, hyp_rcut_step, hyp_ncut_step
```

## Installation

```bash
pip install pgcuts
```

Or from source:

```bash
git clone https://github.com/aghriss/pgcuts.git
cd pgcuts && pip install -e .
```

Requires PyTorch >= 2.0 with CUDA and Triton.

## Algorithms

PGCuts implements three probabilistic graph cut objectives:

| Objective | Envelope | What it optimizes |
|-----------|----------|-------------------|
| **PRCut** | `1/p_bar` | Expected Ratio Cut / cluster size |
| **H-RCut** | `2F1(-m, 1; 2; alpha)` | Hypergeometric bound on expected Ratio Cut |
| **H-NCut** | Holder-binned `2F1` per degree bin | Hypergeometric bound on expected Normalized Cut |

```python
from pgcuts.algorithms import prcut_step, hyp_rcut_step, hyp_ncut_step
```

Each function takes a batch of edges and returns `(cut_loss, balance_loss, updated_ema)`.

## Key Components

### Graph Construction

```python
from pgcuts.graph import build_rbf_knn_graph

W = build_rbf_knn_graph(X, n_neighbors=50)  # sparse (N, N) similarity matrix
```

### Hypergeometric Function

GPU-accelerated `2F1(-m, b; c; z)` via Triton associative scan:

```python
from pgcuts.hyp2f1 import Hyp2F1

H = Hyp2F1.apply(-512, 1.0, 2.0, z)  # differentiable w.r.t. z
```

### Gradient Mixer

Normalizes gradients from multiple losses to prevent one from dominating:

```python
from pgcuts.optim import GradientMixer

gm = GradientMixer(model.named_parameters(), loss_scale={"cut": 1.0, "balance": 1.0})

with gm("cut"):
    cut_loss.backward(retain_graph=True)
with gm("balance"):
    balance.backward()
```

### Evaluation

```python
from pgcuts.metrics import evaluate_clustering, compute_rcut_ncut

results = evaluate_clustering(y_true, y_pred, K)  # ACC, NMI
rcut, ncut = compute_rcut_ncut(W, y_pred)          # graph cut values
```

## Reproducing Results

Run the full benchmark (15 datasets x 3 embeddings x 3 objectives = 135 experiments):

```bash
pip install embedata hydra-core hydra-joblib-launcher

python scripts/benchmark.py --multirun \
    dataset=cifar10,cifar100,stl10,aircraft,eurosat,dtd,flowers,pets,food101,gtsrb,fashionmnist,mnist,imagenette,cub,resisc45 \
    model=clipvitL14,dinov2,dinov3b \
    objective=prcut,hyp_rcut,hyp_ncut
```

Results are saved to `results/` as JSON files. Embeddings are loaded via [embedata](https://pypi.org/project/embedata/).

## Graph Quality

We introduce a metric to assess how well a KNN graph captures class structure:

```python
Q = (q - q_chance) / (1 - q_chance)
```

where `q` = probability of landing in the same class after one random walk step, and `q_chance` accounts for class imbalance. `Q = 0` means random; `Q = 1` means perfect class separation.

Strong correlation between Q and downstream accuracy: when Q > 0.8, H-Cut achieves high accuracy; when Q < 0.4, no graph method works well.

## Package Structure

```
pgcuts/
    algorithms/     # prcut_step, hyp_rcut_step, hyp_ncut_step
    hyp2f1/         # GPU 2F1 kernel (Triton + CUDA backends)
    losses/         # Loss modules (PRCut, H-RCut, H-NCut, FlashCut)
    graph.py        # KNN graph construction
    metrics.py      # ACC, NMI, ARI, RCut, NCut evaluation
    optim.py        # GradientMixer
    functional.py   # Quadrature-based PRCut
    layers.py       # PerFeatLinear, DecoupledLinear
    utils/          # Edge sampling, data utilities
```

## Citation

```bibtex
@inproceedings{pgcuts2026,
    title={Probabilistic Graph Cuts},
    author={Ayoub Ghriss},
    booktitle={AISTATS},
    year={2026}
}
```
