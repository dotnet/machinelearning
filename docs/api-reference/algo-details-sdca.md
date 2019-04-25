### Training Algorithm Details
This trainer is based on the Stochastic Dual Coordinate Ascent (SDCA) method, a
state-of-the-art optimization technique for convex objective functions. The
algorithm can be scaled because it's a streaming training algorithm as described
in a [KDD best
paper.](https://www.csie.ntu.edu.tw/~cjlin/papers/disk_decomposition/tkdd_disk_decomposition.pdf)
        
Convergence is underwritten by periodically enforcing synchronization between
primal and dual variables in a separate thread. Several choices of loss
functions are also provided such as
[hinge-loss](https://en.wikipedia.org/wiki/Hinge_loss) and [logistic
loss](http://www.hongliangjie.com/wp-content/uploads/2011/10/logistic.pdf).
Depending on the loss used, the trained model can be, for example, [support
vector machine](https://en.wikipedia.org/wiki/Support-vector_machine) or
[logistic regression](https://en.wikipedia.org/wiki/Logistic_regression). The
SDCA method combines several of the best properties such the ability to do
streaming learning (without fitting the entire data set into your memory),
reaching a reasonable result with a few scans of the whole data set (for
example, see experiments in [this
paper](https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf)), and spending no
computation on zeros in sparse data sets.
          
Note that SDCA is a stochastic and streaming optimization algorithm. The result
depends on the order of training data because the stopping tolerance is not
tight enough. In strongly-convex optimization, the optimal solution is unique
and therefore everyone eventually reaches the same place. Even in
non-strongly-convex cases, you will get equally-good solutions from run to run.
For reproducible results, it is recommended that one sets 'Shuffle' to False and
'NumThreads' to 1.

This class use [empricial risk minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization) to formulate the optimized problem built upon collected data.
If the training data does not contain enough data points (for example, to train a linear model in $n$-dimensional space, we at least need $n$ data points),
(overfitting)(https://en.wikipedia.org/wiki/Overfitting) may happen so the trained model is good at describing training data but may fail to predict correct results in unseen events.
[Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) is a common technique to alleviate such a phenomenon by penalizing the magnitude (usually measureed by [norm function](https://en.wikipedia.org/wiki/Norm_(mathematics))) of model parameters.
This trainer supports [elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization) which penalizing a linear combination of L1-norm (LASSO), $|| \textbf{w}_c ||_1$, and L2-norm (ridge), $|| \textbf{w}_c ||_2^2$ regularizations.
L1-norm and L2-norm regularizations have different effects and uses that are complementary in certain respects.
Togehter with the implemented optimization algorithm, L1-norm regularization can increase the sparsity of the $\textbf{w}_1,\dots,\textbf{w}_m$.
For high-dimention and sparse data set, if user carefully select the coefficient of L1-norm, it is possible to achieve a good prediction quality with a model with a few of non-zeros (e.g., 1% values) in $\textbf{w}_1,\dots,\textbf{w}_m$ without affecting its .
In contrast, L2-norm can not increase the sparsity of the trained model but can still prevernt overfitting by avoiding large parameter values.
Sometimes, using L2-norm leads to a better prediction quality, so user may still want try it and fine tune the coefficints of L1-norm and L2-norm.
Note that conceptually, using L1-norm implies that the distribution of all model parameters is a [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution) while
L2-norm means that a [Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution) for them.

An aggressive regularization (that is, assigning large coefficients to L1-norm or L2-norm regularization terms) can harm predictive capacity by excluding important variables out of the model.
Therefore, choosing the right regularization coefficients is important in practice.
For example, a very large L1-norm coefficient may force all parameters to be zeros and lead to a trivial model.

For more information, see:
* [Scaling Up Stochastic Dual Coordinate
  Ascent.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/main-3.pdf)
* [Stochastic Dual Coordinate Ascent Methods for Regularized Loss
  Minimization.](http://www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf)

Check the See Also section for links to examples of the usage.
