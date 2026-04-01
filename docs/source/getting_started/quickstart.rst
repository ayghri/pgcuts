Quickstart
==========

PGCuts provides an sklearn-compatible API. If you've used
``KMeans`` or ``SpectralClustering``, you already know how to use PGCuts.

Basic usage
-----------

.. code-block:: python

   from pgcuts import HyCut

   # X is an (N, D) numpy array of features (e.g., from DINOv2)
   labels = HyCut(n_clusters=10).fit_predict(X)

That's it. Under the hood, H-Cut:

1. Builds a KNN similarity graph on ``X``
2. Trains a linear model to minimize the hypergeometric NCut bound
3. Returns hard cluster assignments via argmax

Comparison with sklearn
-----------------------

PGCuts is a drop-in alternative to sklearn's clustering:

.. code-block:: python

   from sklearn.cluster import KMeans, SpectralClustering
   from pgcuts import HyCut

   X = ...  # (N, D) feature matrix

   # K-Means
   labels_km = KMeans(n_clusters=10).fit_predict(X)

   # Spectral Clustering
   labels_sc = SpectralClustering(n_clusters=10).fit_predict(X)

   # PGCuts (Hypergeometric NCut)
   labels_hycut = HyCut(n_clusters=10).fit_predict(X)

Choosing the objective
----------------------

PGCuts supports three graph cut objectives:

.. code-block:: python

   # Hypergeometric NCut (default) — best for most cases
   HyCut(n_clusters=K, objective="hyp_ncut")

   # Hypergeometric RatioCut — simpler, no degree binning
   HyCut(n_clusters=K, objective="hyp_rcut")

   # Probabilistic RatioCut — PRCut baseline
   HyCut(n_clusters=K, objective="prcut")

**When to use which:**

- ``hyp_ncut``: Default. Best when cluster sizes are unbalanced or
  the graph has heterogeneous degree distribution.
- ``hyp_rcut``: Simpler variant. Works well when clusters are
  roughly equal-sized.
- ``prcut``: Original PRCut objective. Useful as a baseline.

Common options
--------------

.. code-block:: python

   model = HyCut(
       n_clusters=10,
       objective="hyp_ncut",    # "hyp_ncut", "hyp_rcut", or "prcut"
       n_neighbors=50,          # KNN graph neighbors (default: 50)
       steps=3000,              # optimization steps (default: 3000)
       lr=1e-3,                 # learning rate (default: 1e-3)
       m=512,                   # polynomial degree for 2F1 bound
       device="cuda",           # "cuda" or "cpu"
   )
   labels = model.fit_predict(X)

   # Access cut values after fitting
   print(f"NCut: {model.ncut_:.4f}, RCut: {model.rcut_:.4f}")

Evaluation
----------

.. code-block:: python

   from pgcuts import evaluate_clustering

   results = evaluate_clustering(y_true, labels, n_clusters=10)
   print(f"ACC: {results['accuracy']:.4f}")
   print(f"NMI: {results['nmi']:.4f}")

Working with embeddings
-----------------------

PGCuts works best with pre-extracted embeddings from foundation models
(DINOv2, CLIP, etc.):

.. code-block:: python

   import numpy as np
   from pgcuts import HyCut

   # Load pre-extracted features
   X = np.load("features.npy")       # (N, D) float32
   y = np.load("labels.npy")         # (N,) int, for evaluation only

   model = HyCut(n_clusters=len(np.unique(y)))
   labels = model.fit_predict(X)

Graph quality
-------------

Before clustering, check if the KNN graph captures class structure:

.. code-block:: python

   from pgcuts.graph import build_rbf_knn_graph
   from pgcuts.metrics import compute_rcut_ncut
   import numpy as np

   W = build_rbf_knn_graph(X, n_neighbors=50)

   # Graph quality Q: 1.0 = perfect, 0.0 = random
   T = W.toarray() / W.sum(axis=1)
   q = np.mean([T[i] @ (y == y[i]) for i in range(len(y))])
   q_chance = np.sum((np.bincount(y) / len(y)) ** 2)
   Q = (q - q_chance) / (1 - q_chance)
   print(f"Graph quality Q = {Q:.3f}")

If ``Q < 0.3``, the embeddings don't separate classes well enough
for any graph-cut method to work. Try a better embedding model.
