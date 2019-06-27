### Prediction Details
This estimator produces several output columns from a tree ensemble model. Assume that the model contains only one decision tree:

                   Node 0
                   /    \
                 /        \
               /            \
             /                \
           Node 1            Node 2
           /    \            /    \
         /        \        /        \
       /            \     Leaf -3  Node 3
      Leaf -1      Leaf -2         /    \
                                 /        \
                                Leaf -4  Leaf -5

Assume that the input feature vector falls into `Leaf -1`. The output `Trees` may be a 1-element vector where
the only value is the decision value carried by `Leaf -1`. The output `Leaves` is a 0-1 vector. If the reached
leaf is the $i$-th (indexed by $-(i+1)$ so the first leaf is `Leaf -1`) leaf in the tree, the $i$-th value in `Leaves`
would be 1 and all other values would be 0. The output `Paths` is a 0-1 representation of the nodes passed
through before reaching the leaf. The $i$-th element in `Paths` indicates if the $i$-th node (indexed by $i$) is touched.
For example, reaching `Leaf -1` lead to $[1, 1, 0, 0]$ as the `Paths`. If there are multiple trees, this estimator
just concatenates `Trees`'s, `Leaves`'s, `Paths`'s from all trees (first tree's information comes first in the concatenated vectors).

Check the See Also section for links to usage examples.