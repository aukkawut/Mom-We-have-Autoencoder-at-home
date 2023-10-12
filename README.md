# Silly Dimensional Reduction
This is repo for Silly Dimensional Reduction where we just perform regression on the data, select the most prominent features, and then randomly generate the data from the pruned regression model. Use that with other algorithms and see what happens.

## Assumption

At this point, assume that data is "linear" enough that we can just naively fit $$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$ or have like $r^2 > 0.8$ or something like that. 

## Idea

So, first we want to summarize the data down into the linear function that we can then use the traditional variable selection method like LASSO or something to reduce the dimension on the coefficient down. Then, we generate the random points (constraints by the domain of the features), use that as our new dimensional reduced data, and see what will go wrong.

This idea can expand by using more sophisicated method. 