Feature Extraction
==================

PGCuts operates on pre-computed feature embeddings. You bring the
features, PGCuts handles the clustering.

Any embedding model works — DINOv2, CLIP, MAE, etc. The key requirement
is that similar samples should have similar feature vectors.

Recommended models
------------------

.. list-table::
   :header-rows: 1

   * - Model
     - Dimension
     - Best for
   * - DINOv2 ViT-L/14
     - 1536
     - Fine-grained datasets (best overall)
   * - DINOv3-B
     - 1536
     - Coarse-grained datasets
   * - CLIP ViT-L/14
     - 768
     - Text-aligned, zero-shot tasks

Using ``embedata``
------------------

The easiest way to get embeddings is via the
`embedata <https://github.com/ayghri/embedata>`_ package
(``pip install embedata``), which provides pre-extracted embeddings
for common datasets:

.. code-block:: python

   from embedata import load_embeddings

   ds = load_embeddings("cifar10", "dinov2", "/path/to/representations/")
   X, y = ds.feats, ds.labels   # numpy arrays

Supported datasets include: CIFAR-10/100, STL-10, EuroSAT, ImageNet,
Flowers-102, CUB-200, Food-101, Pets, DTD, GTSRB, RESISC-45,
FGVC-Aircraft, and more.

Extracting your own features
-----------------------------

Use any standard feature extraction pipeline. For example, with
``torchvision``:

.. code-block:: python

   import torch
   import numpy as np
   from torchvision import datasets, transforms

   # Load model
   model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
   model.eval().cuda()

   # Load dataset
   transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225],
       ),
   ])
   dataset = datasets.CIFAR10(
       root="./data", train=True,
       download=True, transform=transform,
   )
   loader = torch.utils.data.DataLoader(
       dataset, batch_size=256, num_workers=4,
   )

   # Extract
   features, labels = [], []
   with torch.no_grad():
       for images, targets in loader:
           feat = model(images.cuda())
           features.append(feat.cpu().numpy())
           labels.append(targets.numpy())

   X = np.concatenate(features)  # (N, 1536)
   y = np.concatenate(labels)    # (N,)

Using with PGCuts
-----------------

Once you have features, clustering is one line:

.. code-block:: python

   from pgcuts import HyCut

   labels = HyCut(n_clusters=10).fit_predict(X)

The ``embedata`` package (``pip install embedata``) provides a convenient
loader for pre-extracted embeddings:

.. code-block:: python

   from embedata import load_embeddings

   ds = load_embeddings("cifar10", "dinov2", "/path/to/repr/")
   X, y = ds.feats, ds.labels
