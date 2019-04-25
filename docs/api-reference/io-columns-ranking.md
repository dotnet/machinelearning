### Input and Output Columns
The input label data type must be [key](xref:Microsoft.ML.Data.KeyDataViewType)
type or <xref:System.Single>. The value of the label determines relevance, where
higher values indicate higher relevance. If the label is a
[key](xref:Microsoft.ML.Data.KeyDataViewType) type, then the key index is the
relevance value, where the smallest index is the least relevant. If the label is a
<xref:System.Single>, larger values indicate higher relevance. The feature
column must be a known-sized vector of <xref:System.Single> and input row group
column must be [key](xref:Microsoft.ML.Data.KeyDataViewType) type.

This trainer outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | <xref:System.Single> | The unbounded score that was calculated by the model to determine the prediction.|