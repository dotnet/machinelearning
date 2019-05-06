### Training Algorithm Details
Generalized Additive Models, or GAMs, model the data as a set of linearly
independent features similar to a linear model. For each feature, the GAM
trainer learns a non-linear function, called a "shape function", that computes
the response as a function of the feature's value. (In contrast, a linear model
fits a linear response (e.g. a line) to each feature.) To score an input, the
outputs of all the shape functions are summed and the score is the total value.

This GAM trainer is implemented using shallow gradient boosted trees (e.g. tree
stumps) to learn nonparametric shape functions, and is based on the method
described in Lou, Caruana, and Gehrke. ["Intelligible Models for Classification
and Regression."](http://www.cs.cornell.edu/~yinlou/papers/lou-kdd12.pdf)
KDD'12, Beijing, China. 2012. After training, an intercept is added to represent
the average prediction over the training set, and the shape functions are
normalized to represent the deviation from the average prediction. This results
in models that are easily interpreted simply by inspecting the intercept and the
shape functions. See the sample below for an example of how to train a GAM model
and inspect and interpret the results.

Check the See Also section for links to examples of the usage.