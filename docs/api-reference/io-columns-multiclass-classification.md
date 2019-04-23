### Input and Output Columns
The input label column data must be [key](xref:Microsoft.ML.Data.KeyDataViewType) type and the feature column must be a known-sized vector of <xref:System.Single>.

This trainer outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | Vector of <xref:System.Single> | The scores of all classes. Higher value means higher probability to fall into the associated class. If the i-th element has the largest value, the predicted label index would be i. Note that i is zero-based index. |
| `PredictedLabel` | [key](xref:Microsoft.ML.Data.KeyDataViewType) type | The predicted label's index. If its value is i, the actual label would be the i-th category in the key-valued input label type. |
