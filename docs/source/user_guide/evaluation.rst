Evaluation
==========

PGCuts provides metrics for evaluating clustering quality, including both
standard classification metrics (via Hungarian matching) and graph-based
objectives.

Clustering metrics
------------------

The main entry point is :func:`~pgcuts.metrics.evaluate_clustering`:

.. code-block:: python

   from pgcuts.metrics import evaluate_clustering

   results = evaluate_clustering(y_true, y_pred, num_classes=K)
   print(results["accuracy"])  # Hungarian-matched accuracy
   print(results["nmi"])       # Normalized Mutual Information

This function:

1. Computes a confusion matrix between true and predicted labels
2. Applies the Hungarian algorithm to find the optimal label matching
3. Returns accuracy, NMI, and the confusion matrix

Individual metrics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pgcuts.metrics import nmi_score, ari_score, cluster_acc_score

   nmi = nmi_score(y_true, y_pred)    # geometric average NMI
   ari = ari_score(y_true, y_pred)    # Adjusted Rand Index
   acc = cluster_acc_score(y_true, y_pred)  # Hungarian accuracy

Graph-based objectives
-----------------------

Compute the actual RatioCut and NCut values on the graph:

.. code-block:: python

   from pgcuts.metrics import compute_rcut_ncut

   rcut, ncut = compute_rcut_ncut(W_sparse, y_pred)
   # W_sparse: scipy sparse adjacency matrix
   # y_pred: integer cluster labels

This computes:

.. math::

   \text{RCut} = \sum_\ell \frac{\text{Cut}(\ell)}{|\text{cluster}_\ell|}
   \qquad
   \text{NCut} = \sum_\ell \frac{\text{Cut}(\ell)}{\text{Vol}(\text{cluster}_\ell)}

Soft objectives
~~~~~~~~~~~~~~~

For soft (probabilistic) assignments, use :func:`~pgcuts.metrics.soft_ncut`
and :func:`~pgcuts.metrics.soft_rcut`:

.. code-block:: python

   from pgcuts.metrics import soft_ncut, soft_rcut

   P = softmax(network(X_tensor)).detach().cpu().numpy()

   soft_ncut_val = soft_ncut(W_sparse, P)
   soft_rcut_val = soft_rcut(W_sparse, P)

These evaluate the expected cut under the soft assignment matrix, which is the
actual quantity being optimized during training.

Ratio cut per cluster
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pgcuts.metrics import ratio_cut_score

   rcut = ratio_cut_score(W_dense, y_pred, num_clusters=K)

Full evaluation example
-----------------------

.. code-block:: python

   import numpy as np
   import torch
   from pgcuts.metrics import evaluate_clustering, compute_rcut_ncut

   # After training
   with torch.no_grad():
       logits = network(X_tensor)
       pred = logits.argmax(dim=-1).cpu().numpy()

   # Classification metrics
   results = evaluate_clustering(y_true, pred, K)
   print(f"Accuracy: {results['accuracy']:.4f}")
   print(f"NMI:      {results['nmi']:.4f}")

   # Graph objectives
   rcut, ncut = compute_rcut_ncut(W_sparse, pred)
   print(f"RCut:     {rcut:.4f}")
   print(f"NCut:     {ncut:.4f}")
