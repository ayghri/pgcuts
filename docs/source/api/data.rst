pgcuts.data
===========

Data loading utilities for edge-pair mini-batch training.

.. autoclass:: pgcuts.utils.data.ShuffledRangeDataset
   :members:
   :undoc-members:

Usage:

.. code-block:: python

   from torch.utils.data import DataLoader
   from pgcuts.data import ShuffledRangeDataset

   # n: total number of edges, k: batch size
   dataset = ShuffledRangeDataset(n=num_edges, k=8192)
   loader = DataLoader(dataset, batch_size=1, num_workers=4)

   for batch_idx in loader:
       edge_batch = pairs[batch_idx[0]]
