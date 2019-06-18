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
