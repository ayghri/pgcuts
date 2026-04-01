H-RCut -- Hypergeometric RatioCut
==================================

H-RCut extends PRCut with a hypergeometric envelope
:math:`{}_{2}F_{1}(-m, 1; 2; \bar{\alpha}_\ell)` that provides a
tighter bound on the expected RatioCut (Theorem 1).

Quick usage
-----------

.. code-block:: python

   from pgcuts import HyCut

   labels = HyCut(n_clusters=10, objective="hyp_rcut").fit_predict(X)

Objective
---------

.. math::

   \text{H-RCut} = \sum_\ell {}_{2}F_{1}(-m, 1; 2; \bar{\alpha}_\ell)
   \cdot \sum_i M_{i\ell}(P)

where :math:`M_{i\ell}(P) = \sum_j w_{ij} p_{i\ell}(1 - p_{j\ell})` and
:math:`\bar{\alpha}_\ell` is the EMA cluster proportion.

The envelope is the same for every vertex (:math:`\beta = 1`), making
H-RCut simpler than H-NCut but less expressive for graphs with
heterogeneous degree distributions.

When to use H-RCut
-------------------

- Clusters are roughly equal-sized
- Graph degrees are relatively uniform
- You want a simpler objective with fewer hyperparameters (no binning)

For unbalanced clusters or heavy-tailed degree distributions, use
:doc:`H-NCut <hncut>` instead.

Low-level API
-------------

.. code-block:: python

   from pgcuts.algorithms import hyp_rcut_step

   cut_loss, balance, p_ema = hyp_rcut_step(
       logits, left_idx, right_idx, w, p_ema,
       m=512, ema=0.9,
   )

.. list-table:: Comparison with PRCut
   :header-rows: 1

   * - Property
     - PRCut
     - H-RCut
   * - Envelope
     - :math:`1/\bar{\alpha}_\ell`
     - :math:`{}_{2}F_{1}(-m, 1; 2; \bar{\alpha}_\ell)`
   * - Bound tightness
     - Looser
     - Tighter
   * - Per-vertex weights
     - No
     - No (see :doc:`hncut` for per-vertex)
