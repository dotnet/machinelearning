### Input and Output Columns
The input label data type must be [key](xref:Microsoft.ML.Data.KeyDataViewType)
type or <xref:System.Single>. The value of the label determines relevance, where
higher values indicate higher relevance. If the label is a
[key](xref:Microsoft.ML.Data.KeyDataViewType) type, then the key index is the
relevance value, where the smallest index is the least relevant. If the label is a
<xref:System.Single>, larger values indicate higher relevance. The feature
column must be a known-sized vector of <xref:System.Single> and input row group
column must be [key](xref:Microsoft.ML.Data.KeyDataViewType) type.

This estimator outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Trees` | Vector of <xref:System.Single> | The output values of all trees. |
| `Leaves` | Vector of <xref:System.Single> | The IDs of all leaves where the input feature vector falls into. |
| `Paths` | Vector of <xref:System.Single> | The paths the input feature vector passed through to reach the leaves. |

Those output columns are all optional and user can change their names.
Please set the names of skipped columns to null so that they would not be produced.