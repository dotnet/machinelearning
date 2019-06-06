### Input and Output Columns
The input label column data must be <xref:System.Boolean>.
The input features column data must be a known-sized vector of <xref:System.Single>.

This estimator outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Trees` | Vector of <xref:System.Single> | The output values of all trees. |
| `Leaves` | Vector of <xref:System.Single> | The IDs of all leaves where the input feature vector falls into. |
| `Paths` | Vector of <xref:System.Single> | The paths the input feature vector passed through to reach the leaves. |

Those output columns are all optional and user can change their names.
Please set the names of skipped columns to null so that they would not be produced.