pgcuts.losses
=============

Loss functions for probabilistic graph cut optimization.

PRCut losses
------------

.. autoclass:: pgcuts.losses.PRCutGradLoss
   :members:
   :undoc-members:

.. autoclass:: pgcuts.losses.PRCutBatchLoss
   :members:
   :undoc-members:

.. autoclass:: pgcuts.losses.SimplexL2Loss
   :members:
   :undoc-members:

H-RCut loss
-----------

.. autoclass:: pgcuts.losses.HyCutLoss
   :members:
   :undoc-members:

RatioCut / NCut losses
-----------------------

.. autoclass:: pgcuts.losses.pncut.RatioCutLoss
   :members:
   :undoc-members:

.. autoclass:: pgcuts.losses.pncut.NCutLoss
   :members:
   :undoc-members:

Binning utilities
~~~~~~~~~~~~~~~~~

.. autofunction:: pgcuts.losses.pncut.equal_size_bins

.. autofunction:: pgcuts.losses.pncut.log_kmeans_bins

.. autofunction:: pgcuts.losses.pncut.compute_ncut_bin_phi
