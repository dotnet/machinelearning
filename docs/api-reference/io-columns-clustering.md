### Input and Output Columns
The input features column data must be <xref:System.Single>. No label column needed. This trainer outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | <xref:System.Single> | The unbounded score that was calculated by the trainer to determine the prediction.|
| `PredictedLabel` | <xref:System.Int32> | The cluster id predicted by the trainer.|