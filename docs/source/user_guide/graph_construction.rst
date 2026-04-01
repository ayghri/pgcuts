Graph Construction
==================

.. note::

   If you use the :class:`~pgcuts.HyCut` class, graph construction is
   handled automatically. This page is for advanced users who want to build
   graphs manually or customize the pipeline.

All PGCuts algorithms operate on a weighted graph where nodes are data points
and edge weights encode similarity. The ``pgcuts.graph`` module provides the
graph construction pipeline.

Standard pipeline
-----------------

The recommended function is :func:`~pgcuts.graph.build_rbf_knn_graph`, which
performs the full pipeline:

1. **KNN distances** -- compute k-nearest neighbor distances
2. **Symmetrize** -- ``(W + W^T) / 2``
3. **Row-normalize** -- ``W / W.sum(axis=1)``
4. **Scale by degree** -- multiply by the number of neighbors per node
5. **Gaussian RBF** -- ``exp(-d^2 / 2)``
6. **Final symmetrization**

.. code-block:: python

   from pgcuts.graph import build_rbf_knn_graph

   # X: numpy array, shape (n, d)
   W = build_rbf_knn_graph(X, n_neighbors=100)
   # Returns: scipy sparse CSR matrix, shape (n, n)

The result is a sparse symmetric similarity matrix suitable for all PGCuts losses.

Choosing ``n_neighbors``
~~~~~~~~~~~~~~~~~~~~~~~~

- Typical values: 20--200
- Larger values produce denser graphs with smoother cuts
- Smaller values are more sensitive to local structure
- A good starting point is ``n_neighbors=100``

Step-by-step construction
--------------------------

For more control, use the individual functions:

.. code-block:: python

   from pgcuts.graph import knn_graph, gaussian_rbf_kernel

   # Step 1: build symmetric KNN distance graph
   W_dist = knn_graph(X, n_neighbors=50, mode="distance")

   # Step 2: convert distances to RBF similarities
   W_sim = gaussian_rbf_kernel(W_dist, sigma=None)  # auto bandwidth

The ``sigma`` parameter controls the RBF bandwidth. When ``None``, it uses the
median of per-row maximum distances.

Edge pairs for mini-batch training
-----------------------------------

Training requires sampling edges from the graph. Extract the edge list and
use :class:`~pgcuts.utils.data.ShuffledRangeDataset`:

.. code-block:: python

   import numpy as np
   from torch.utils.data import DataLoader
   from pgcuts.utils.data import ShuffledRangeDataset

   pairs = np.array(W.nonzero()).T  # shape (E, 2)
   dataset = ShuffledRangeDataset(n=pairs.shape[0], k=8192)
   loader = DataLoader(dataset, batch_size=1, num_workers=4)

   for batch_idx in loader:
       batch_pairs = pairs[batch_idx[0]]  # (k, 2)
       # ... training step ...

Extracting edge weights and node maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each batch of edge pairs, use :func:`~pgcuts.utils.get_pairs_unique_map` to
deduplicate nodes (since many edges share endpoints):

.. code-block:: python

   from pgcuts.utils import get_pairs_unique_map

   unique_idx, left_idx, right_idx = get_pairs_unique_map(batch_pairs)
   # unique_idx: indices of unique nodes in X
   # left_idx: indices into unique_idx for left endpoints
   # right_idx: indices into unique_idx for right endpoints

   x_batch = X_tensor[unique_idx]   # only forward unique nodes
   p = softmax(network(x_batch))
   p_left = p[left_idx]
   p_right = p[right_idx]

Vertex degrees
--------------

For NCut, you need per-vertex degrees:

.. code-block:: python

   import numpy as np

   degrees = np.array(W.sum(axis=1)).flatten()  # shape (n,)
