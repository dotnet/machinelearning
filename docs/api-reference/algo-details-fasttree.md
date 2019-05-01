### Training Algorithm Details
FastTree is an efficient implementation of the
[MART](https://arxiv.org/abs/1505.01866) gradient boosting algorithm. Gradient
boosting is a machine learning technique for regression problems. It builds each
regression tree in a step-wise fashion, using a predefined loss function to
measure the error for each step and corrects for it in the next. So this
prediction model is actually an ensemble of weaker prediction models. In
regression problems, boosting builds a series of such trees in a step-wise
fashion and then selects the optimal tree using an arbitrary differentiable loss
function.

MART learns an ensemble of regression trees, which is a decision tree with
scalar values in its leaves. A decision (or regression) tree is a binary
tree-like flow chart, where at each interior node one decides which of the two
child nodes to continue to based on one of the feature values from the input. At
each leaf node, a value is returned. In the interior nodes, the decision is
based on the test x <= v where x is the value of the feature in the input
sample and v is one of the possible values of this feature. The functions that
can be produced by a regression tree are all the piece-wise constant functions.
          
The ensemble of trees is produced by computing, in each step, a regression tree
that approximates the gradient of the loss function, and adding it to the
previous tree with coefficients that minimize the loss of the new tree. The
output of the ensemble produced by MART on a given instance is the sum of the
tree outputs.

* In case of a binary classification problem, the output is converted to a
  probability by using some form of calibration.
* In case of a regression problem, the output is the predicted value of the
  function.
* In case of a ranking problem, the instances are ordered by the output value of
  the ensemble.

For more information see:
* [Wikipedia: Gradient boosting (Gradient tree
boosting).](https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting)
* [Greedy function approximation: A gradient boosting
machine.](https://projecteuclid.org/DPubS?service=UI&amp;version=1.0&amp;verb=Display&amp;handle=euclid.aos/1013203451)

Check the See Also section for links to examples of the usage.