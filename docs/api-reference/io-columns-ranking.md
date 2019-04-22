### Input and Output Columns
The input label column data must be <xref:System.Single> or [key](xref:Microsoft.ML.Data.KeyDataViewType) type, the feature column must be a known-sized vector of <xref:System.Single> and input row group column must be [key](xref:Microsoft.ML.Data.KeyDataViewType) type. This trainer outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | <xref:System.Single> | The unbounded score that was calculated by the model to determine the prediction.|