### Input and Output Columns
The input label column data must be <xref:System.Boolean>.
The input features column data must be a known-sized vector of <xref:System.Single>.

This estimator outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Trees` | Known-sized vector of <xref:System.Single> | The output values of all trees. Its size is identical to the total number of trees in the tree ensemble model. |
| `Leaves` | Known-sized vector of <xref:System.Single> | 0-1 vector representation to the IDs of all leaves where the input feature vector falls into. Its size is the number of total leaves in the tree ensemble model. |
| `Paths` | Known-sized vector of <xref:System.Single> | 0-1 vector representation to the paths the input feature vector passed through to reach the leaves. Its size is the number of non-leaf nodes in the tree ensemble model. |

Those output columns are all optional and user can change their names.
Please set the names of skipped columns to null so that they would not be produced.