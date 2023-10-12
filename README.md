# Silly Dimensional Reduction
This is repo for Silly Dimensional Reduction where we just perform regression on the data, select the most prominent features, and then randomly generate the data from the pruned regression model. Use that with other algorithms and see what happens.

## Assumption

At this point, assume that data is "linear" enough that we can just naively fit $$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$ or have like $r^2 > 0.8$ or something like that. 
