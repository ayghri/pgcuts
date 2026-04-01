Installation
============

Requirements
------------

- Python >= 3.11, < 3.14
- PyTorch >= 2.8.0 (with CUDA support recommended)
- A CUDA-capable GPU (for the Triton/CUDA hypergeometric kernels)

Install from PyPI
-----------------

.. code-block:: bash

   pip install pgcuts

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/ayghri/pgcuts.git
   cd pgcuts
   pip install .

For development:

.. code-block:: bash

   pip install -e ".[experiments]"

Verify installation
-------------------

.. code-block:: python

   from pgcuts import HyCut
   print("PGCuts installed successfully!")

   # Quick GPU check
   import torch
   from pgcuts import Hyp2F1
   z = torch.rand(10, device="cuda")
   out = Hyp2F1.apply(-10, 1.0, 2.0, z)
   print(f"2F1 output: {out}")
