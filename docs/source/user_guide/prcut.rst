PRCut -- Probabilistic RatioCut
================================

PRCut is the baseline probabilistic graph cut objective from
`Yu et al. (2023) <https://arxiv.org/abs/2310.00969>`_.

Quick usage
-----------

.. code-block:: python

   from pgcuts import HyCut

   labels = HyCut(n_clusters=10, objective="prcut").fit_predict(X)

Objective
---------

PRCut minimizes the expected RatioCut normalized by cluster proportions:

.. math::

   \text{PRCut} = \sum_\ell \frac{1}{\bar{\alpha}_\ell}
   \cdot \frac{1}{|E|} \sum_{(i,j) \in E} w_{ij} \, p_{i\ell} (1 - p_{j\ell})

where :math:`p_{i\ell} = \text{softmax}(f_\theta(x_i))_\ell` and
:math:`\bar{\alpha}_\ell` is the EMA cluster proportion.

A balance loss prevents cluster collapse:

.. math::

   \mathcal{L}_\text{balance} = -\sum_\ell \bar{p}_\ell \log \bar{p}_\ell

Both losses are combined via gradient mixing (each gradient is
unit-normalized before summing).

Low-level API
-------------

For custom training loops, use the step functions directly:

.. code-block:: python

   from pgcuts.algorithms import prcut_step

   cut_loss, balance, p_ema = prcut_step(
       probs, left_idx, right_idx, w, p_ema,
       ema=0.9,
   )

Or the loss classes:

.. code-block:: python

   from pgcuts.losses import PRCutBatchLoss

   loss_fn = PRCutBatchLoss(num_clusters=K, gamma=0.95)
   loss_fn.update_cluster_p(P)
   loss = loss_fn(W_batch, P_left, P_right)
