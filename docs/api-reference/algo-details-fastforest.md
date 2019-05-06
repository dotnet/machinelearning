### Training Algorithm Details
Decision trees are non-parametric models that perform a sequence of simple tests
on inputs. This decision procedure maps them to outputs found in the training
dataset whose inputs were similar to the instance being processed. A decision is
made at each node of the binary tree data structure based on a measure of
similarity that maps each instance recursively through the branches of the tree
until the appropriate leaf node is reached and the output decision returned.

Decision trees have several advantages:
* They are efficient in both computation and memory usage during training and
  prediction.
* They can represent non-linear decision boundaries.
* They perform integrated feature selection and classification.
* They are resilient in the presence of noisy features.

Fast forest is a random forest implementation. The model consists of an ensemble
of decision trees. Each tree in a decision forest outputs a Gaussian
distribution by way of prediction. An aggregation is performed over the ensemble
of trees to find a Gaussian distribution closest to the combined distribution
for all trees in the model. This decision forest classifier consists of an
ensemble of decision trees.

Generally, ensemble models provide better coverage and accuracy than single
decision trees. Each tree in a decision forest outputs a Gaussian distribution.

For more see:
* [Wikipedia: Random forest](https://en.wikipedia.org/wiki/Random_forest)
* [Quantile regression
  forest](http://jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf)
* [From Stumps to Trees to
  Forests](https://blogs.technet.microsoft.com/machinelearning/2014/09/10/from-stumps-to-trees-to-forests/)

Check the See Also section for links to examples of the usage.