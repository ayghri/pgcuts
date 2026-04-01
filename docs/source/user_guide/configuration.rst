Configuration
=============

H-Cut parameters
-----------------

All parameters can be set via the constructor:

.. code-block:: python

   from pgcuts import HyCut

   model = HyCut(
       n_clusters=10,          # number of clusters
       objective="hyp_ncut",   # "hyp_ncut", "hyp_rcut", "prcut"
       n_neighbors=50,         # KNN graph neighbors
       steps=3000,             # optimization steps
       batch_size=8192,        # edges per step
       lr=1e-3,                # learning rate
       weight_decay=0.1,       # AdamW weight decay
       m=512,                  # polynomial degree for 2F1
       num_bins=16,            # degree bins (hyp_ncut only)
       tau_start=10.0,         # initial softmax temperature
       tau_end=1.0,            # final softmax temperature
       distance="ce",          # "ce" (cross-entropy) or "xor"
       ema=0.9,                # EMA decay for cluster proportions
       device="cuda",          # "cuda" or "cpu"
       seed=42,                # random seed
   )

Parameter guide
---------------

**Objective** (``objective``)
    - ``"hyp_ncut"``: Default. Hypergeometric NCut with Holder binning.
    - ``"hyp_rcut"``: Simpler variant without degree binning.
    - ``"prcut"``: Original PRCut baseline.

**Graph density** (``n_neighbors``)
    Controls the KNN graph sparsity. Typical: 20--100.
    Denser graphs are more robust but slower.

**Training steps** (``steps``)
    Default 3000 is sufficient for most datasets.
    Increase for very large K or low graph quality.

**Temperature schedule** (``tau_start``, ``tau_end``)
    Annealed linearly from ``tau_start`` to ``tau_end``.
    Higher start = softer assignments early; lower end = sharper final clusters.

**Polynomial degree** (``m``)
    Controls tightness of the hypergeometric bound.
    Higher = tighter but more computation. Default 512 works well.

Benchmark scripts
-----------------

For reproducible experiments, use the Hydra-based benchmark:

.. code-block:: bash

   python scripts/benchmark.py \
       dataset=cifar10 model=dinov2 objective=hyp_ncut

Multi-run sweep:

.. code-block:: bash

   python scripts/benchmark.py --multirun \
       dataset=cifar10,cifar100,stl10 \
       model=dinov2 \
       objective=prcut,hyp_rcut,hyp_ncut

Config files are in ``scripts/configs/``.
