pgcuts.graph
============

Graph construction and similarity computation.

Main pipeline
-------------

.. autofunction:: pgcuts.graph.build_rbf_knn_graph

Building blocks
----------------

.. autofunction:: pgcuts.graph.knn_graph

.. autofunction:: pgcuts.graph.gaussian_rbf_kernel

.. autofunction:: pgcuts.graph.symmetrize

Cross-set similarities
-----------------------

.. autofunction:: pgcuts.graph.get_knn_distances

.. autofunction:: pgcuts.graph.sp_knn_similarity

.. autofunction:: pgcuts.graph.torch_knn_similarity

.. autofunction:: pgcuts.graph.torch_pairwise_similarities

.. autofunction:: pgcuts.graph.compute_sp_similarities

Laplacian
---------

.. autofunction:: pgcuts.graph.sparse_laplacian
