pgcuts.hyp2f1
=============

GPU-accelerated Gauss hypergeometric function :math:`{}_{2}F_{1}(-m, b; c; z)`
specialized for non-positive integer :math:`a = -m` with :math:`z \in [0, 1]`.

.. autoclass:: pgcuts.hyp2f1.Hyp2F1
   :members:
   :undoc-members:

Usage:

.. code-block:: python

   from pgcuts.hyp2f1 import Hyp2F1

   # Forward: compute 2F1(-512, 1, 2, z) on GPU
   result = Hyp2F1.apply(-512, 1.0, 2.0, z)

   # Backward: gradient w.r.t. z via derivative identity
   result.sum().backward()  # computes z.grad

The backend (Triton or CUDA) is selected automatically. Set
``HYP2F1_BACKEND=cuda`` to force the CUDA backend.
