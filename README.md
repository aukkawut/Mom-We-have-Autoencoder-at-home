# Mom: We have Autoencoder at home
This is repo for Silly Dimensional Reduction where we use PCA to reduce the dimension of the data, then later on use the multiple multivariate linear regression to recreate the initial image like what Autoencoder do.

## Rationale

~~I hate when people say autoencoder is a neural network.~~ Given the data $x \in \mathbb{R}^{m}$, we can potentially summarize that data down into an embedding $q \in \mathbb{R}^{k}$ for $k << m$. This is essentially an encoder part of an autoencoder. Now, we just need the decoder part. The naive model to do this is a regression model $f:\mathbb{R}^{k} \rightarrow \mathbb{R}^m$.

## Result

Here is an empirical result, the MSE is around 0.01 on 25% test set of the linear regression model so it is quite working well.

![LinearModel](/result/linearencoder.png)

![KernelModel](/result/rbfkernelencoder.png)
