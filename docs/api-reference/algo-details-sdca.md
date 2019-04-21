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
'NumThreads' to 1. Elastic net regularization can be specified by the 'L2Const'
and 'L1Threshold' parameters. Note that the 'L2Const' has an effect on the rate
of convergence. In general, the larger the 'L2Const', the faster SDCA converges.
Regularization is a method that can render an ill-posed problem more tractable
by imposing constraints that provide information to supplement the data and that
prevents overfitting by penalizing model's magnitude usually measured by some
norm functions. This can improve the generalization of the model learned by
selecting the optimal complexity in the bias-variance tradeoff. Regularization
works by adding the penalty that is associated with coefficient values to the
error of the hypothesis. An accurate model with extreme coefficient values would
be penalized more, but a less accurate model with more conservative values would
be penalized less. This learner supports [elastic net
regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization): a
linear combination of L1-norm (LASSO), $|| \boldsymbol{w} ||_1$, and L2-norm
(ridge), $|| \boldsymbol{w} ||_2^2$ regularizations. L1-nrom and L2-norm
regularizations have different effects and uses that are complementary in
certain respects. Using L1-norm can increase sparsity of the trained
$\boldsymbol{w}$. When working with high-dimensional data, it shrinks small
weights of irrevalent features to 0 and therefore no reource will be spent on
those bad features when making prediction. L2-norm regularization is preferable
for data that is not sparse and it largely penalizes the existence of large
weights.

For more information, see:
* [Scaling Up Stochastic Dual Coordinate
  Ascent.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/main-3.pdf)
* [Stochastic Dual Coordinate Ascent Methods for Regularized Loss
  Minimization.](http://www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf)