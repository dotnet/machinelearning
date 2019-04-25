### Training Algorithm Details
The Stochastic Gradient Descent (SGD) is one of the popular stochastic
optimization procedures that can be integrated into several machine learning
tasks to achieve state-of-the-art performance. This trainer implements the
Hogwild Stochastic Gradient Descent for binary classification that supports
multi-threading without any locking. If the associated optimization problem is
sparse, Hogwild Stochastic Gradient Descent achieves a nearly optimal rate of
convergence. For more details about Hogwild Stochastic Gradient Descent can be
found [here](http://arxiv.org/pdf/1106.5730v2.pdf).

Check the See Also section for links to examples of the usage.