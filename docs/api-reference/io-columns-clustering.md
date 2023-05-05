### Input and Output Columns
The input features column data must be <xref:System.Single>. No label column needed. This trainer outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | vector of <xref:System.Single> | The distances of the given data point to all clusters' centroids. |
| `PredictedLabel` | [key](xref:Microsoft.ML.Data.KeyDataViewType) type | The closest cluster's index predicted by the model. |
