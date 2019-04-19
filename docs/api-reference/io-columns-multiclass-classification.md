### Input and Output Columns
The input label column data must be [key-typed](xref:Microsoft.ML.Data.KeyDataViewType) and the feature column must be a known-sized vector of <xref:System.Single>.

This trainer outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | vector of <xref:System.Single> | The scores of all classes. Higher value means higher probability to fall into the associated class. If the i-th element has the largest value, the predicted label index would be i. Note that i is zero-based index. |
| `PredictedLabel` | <xref:System.UInt32> | The predicted label's index. If it's value is i, the actual label would be the i-th category in the key-valued input label type. Note that i is zero-based index. |
