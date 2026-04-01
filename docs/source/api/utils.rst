pgcuts.utils
============

Gradient mixing
---------------

.. autoclass:: pgcuts.optim.GradientMixer
   :members:
   :undoc-members:

Usage:

.. code-block:: python

   from pgcuts.optim import GradientMixer

   grad_mix = GradientMixer(
       network.named_parameters(),
       loss_scale={"cut": 1.0, "balance": 1.0},
   )

   optimizer.zero_grad()
   with grad_mix("cut"):
       cut_loss.backward(retain_graph=True)
   with grad_mix("balance"):
       balance_loss.backward()
   optimizer.step()

Pair utilities
--------------

.. autofunction:: pgcuts.utils.pairs.get_pairs_unique_map
