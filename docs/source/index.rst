PGCuts Documentation
====================

PGCuts is a GPU-accelerated implementation of Probabilistic Graph Cuts (PGCuts) for graph clustering with a scikit-learn-compatible API.

For reasonable feature matrices, simply use as follows:

.. code-block:: python
    
   from pgcuts import HyCut

   # Just like KMeans or SpectralClustering
   labels = HyCut(n_clusters=10).fit_predict(X)

PGCuts replace the spectral eigendecomposition with a differentiable probabilistic upper bound that scales to large
datasets, as introduced in:

- `Beyond Spectral Clustering: Probabilistic Cuts for Differentiable Graph Partitioning (AISTATS 2026) <https://openreview.net/forum?id=FN6QAT5Tmc>`_
- `Deep clustering via probabilistic ratio-cut optimization (AISTATS 2025) <https://arxiv.org/abs/2502.03405>`_

PGCuts include:

- **PRCut** -- Probabilistic RatioCut :math:`(\frac{1}{\bar{p}_\ell})`
- **H-RCut** -- Hypergeometric RatioCut :math:`({}_{2}F_{1}(-m, 1; 2; \alpha_\ell))`
- **H-NCut** -- Hypergeometric NCut with Holder-binned envelopes (default)

**Key features:**

- Drop-in replacement for ``sklearn.cluster.SpectralClustering``
- GPU-accelerated :math:`{}_{2}F_{1}` kernels (Triton + CUDA)
- Gradient mixing for collapse-free training
- Works on pre-extracted embeddings (DINOv2, CLIP, etc.)


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/feature_extraction
   user_guide/graph_construction
   user_guide/prcut
   user_guide/hrcut
   user_guide/hncut
   user_guide/evaluation
   user_guide/configuration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/losses
   api/hyp2f1
   api/graph
   api/metrics
   api/data
   api/utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
