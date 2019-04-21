### Training Algorithm Details
This trainer is based on the Stochastic Dual Coordinate Ascent (SDCA) method, a
state-of-the-art optimization technique for convex objective functions. The
algorithm can be scaled for use on large out-of-memory data sets due to a
semi-asynchronized implementation that supports multi-threading.
        
Convergence is underwritten by periodically enforcing synchronization between
primal and dual variables in a separate thread. Several choices of loss
functions are also provided.
          
Note that SDCA is a stochastic and streaming optimization algorithm. The result
depends on the order of training data because the stopping tolerance is not
tight enough. In strongly-convex optimization, the optimal solution is unique
and therefore everyone eventually reaches the same place. Even in
non-strongly-convex cases, you will get equally-good solutions from run to run.
For reproducible results, it is recommended that one sets 'Shuffle' to False and
'NumThreads' to 1. Elastic net regularization can be specified by the 'L2Const'
and 'L1Threshold' parameters. Note that the 'L2Const' has an effect on the rate
of convergence. In general, the larger the 'L2Const', the faster SDCA converges.

For more information, see:
* [Scaling Up Stochastic Dual Coordinate
  Ascent.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/main-3.pdf)
* [Stochastic Dual Coordinate Ascent Methods for Regularized Loss
  Minimization.](http://www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf)