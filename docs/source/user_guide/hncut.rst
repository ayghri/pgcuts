H-NCut -- Hypergeometric NCut
==============================

H-NCut is the default and recommended objective. It provides a per-vertex
hypergeometric envelope using Holder-binned degree weights (Theorem 2).

Quick usage
-----------

.. code-block:: python

   from pgcuts import HyCut

   # hyp_ncut is the default
   labels = HyCut(n_clusters=10).fit_predict(X)

Objective
---------

.. math::

   \text{H-NCut} = \sum_\ell \sum_i \frac{M_{i\ell}(P)}{d_i}
   \cdot \Phi_\ell(d_i)

where the per-vertex envelope :math:`\Phi_\ell(d_i)` is a Holder product
over degree bins:

.. math::

   \Phi_\ell(d_i) = \prod_j \left[
   {}_{2}F_{1}\!\left(-m, 1;\, \frac{d_i}{\beta^*_j}+1;\,
   \bar{\alpha}_{\ell j}\right)
   \right]^{w_j}

Here :math:`\beta^*_j = \min_{v \in \text{bin } j} d_v` and
:math:`w_j` is the bin weight.

Binning strategies
------------------

.. code-block:: python

   from pgcuts import equal_size_bins, log_kmeans_bins

   degrees = W.sum(axis=1).A1  # from the KNN graph
   bins = log_kmeans_bins(degrees, num_bins=16)

- ``equal_size_bins``: Simple, robust. Bins have equal vertex count.
- ``log_kmeans_bins``: K-Means on log-degrees. Better for heavy-tailed
  distributions (recommended).

Key hyperparameters
-------------------

``m`` (int, default: 512)
    Polynomial degree for :math:`{}_{2}F_{1}`. Higher = tighter bound.

``num_bins`` (int, default: 16)
    Number of degree bins. More bins = finer envelope approximation.

``n_neighbors`` (int, default: 50)
    KNN graph density. 20--100 typical range.

Low-level API
-------------

.. code-block:: python

   from pgcuts.algorithms import hyp_ncut_step

   cut_loss, balance, alpha_ema = hyp_ncut_step(
       logits, left_idx, right_idx, w,
       alpha_ema, degrees, node_to_bin,
       q_stars, beta_stars, bin_weights,
       m=512, ema=0.9,
   )

.. list-table:: H-RCut vs H-NCut
   :header-rows: 1

   * - Property
     - H-RCut
     - H-NCut
   * - Normalization
     - RatioCut (:math:`1/n`)
     - NCut (:math:`1/\text{vol}`)
   * - Vertex weights
     - Homogeneous (:math:`\beta = 1`)
     - Per-vertex (:math:`\beta = d_i`)
   * - Envelope
     - Per-cluster
     - Per-vertex (Holder-binned)
   * - Best for
     - Equal-sized clusters
     - Unbalanced clusters, varied degrees
