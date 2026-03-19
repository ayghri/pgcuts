# TURTLE Algorithm

-   Number of spaces F, F features Z_in, number of clusters K
-   Create outer classifier (task encoder) E with weight normalization, which
    takes Z_in and outputs Z_outer (logits_per_space) of shape (N, F, C)
-   A function init_inner that returns a new instance of Inner classifier I
-   For a number of iterations T:
    -   sample a large batch B (b, F, \[D\])
    -   Use outer classifier to compute label_per_space and labels (average of
        spaces of the former)
    -   If we use cold start, we re-instantiate the inner classifier for each
        iteration (it can be done much faster via init_weights)
    - for a number of iteration M:
        - backward the cross_entropy between each space inner logits and outer labels.detach()
    
