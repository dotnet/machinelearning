This class uses [empirical risk minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization) (i.e., ERM)
to formulate the optimization problem built upon collected data.
Note that empirical risk is usually measured by applying a loss function on the model's predictions on collected data points.
If the training data does not contain enough data points
(for example, to train a linear model in $n$-dimensional space, we need at least $n$ data points),
[overfitting](https://en.wikipedia.org/wiki/Overfitting) may happen so that
the model produced by ERM is good at describing training data but may fail to predict correct results in unseen events.
[Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) is a common technique to alleviate
such a phenomenon by penalizing the magnitude (usually measured by the
[norm function](https://en.wikipedia.org/wiki/Norm_(mathematics))) of model parameters.
This trainer supports [elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization),
which penalizes a linear combination of L1-norm (LASSO), $|| \textbf{w}_c ||_1$, and L2-norm (ridge), $|| \textbf{w}_c ||_2^2$ regularizations for $c=1,\dots,m$.
L1-norm and L2-norm regularizations have different effects and uses that are complementary in certain respects.

Together with the implemented optimization algorithm, L1-norm regularization can increase the sparsity of the model weights, $\textbf{w}_1,\dots,\textbf{w}_m$.
For high-dimensional and sparse data sets, if users carefully select the coefficient of L1-norm,
it is possible to achieve a good prediction quality with a model that has only a few non-zero weights
(e.g., 1% of total model weights) without affecting its prediction power.
In contrast, L2-norm cannot increase the sparsity of the trained model but can still prevent overfitting by avoiding large parameter values.
Sometimes, using L2-norm leads to a better prediction quality, so users may still want to try it and fine tune the coefficients of L1-norm and L2-norm.
Note that conceptually, using L1-norm implies that the distribution of all model parameters is a
[Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution) while
L2-norm implies a [Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution) for them.

An aggressive regularization (that is, assigning large coefficients to L1-norm or L2-norm regularization terms)
can harm predictive capacity by excluding important variables from the model.
For example, a very large L1-norm coefficient may force all parameters to be zeros and lead to a trivial model.
Therefore, choosing the right regularization coefficients is important in practice.
