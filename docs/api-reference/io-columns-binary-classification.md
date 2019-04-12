### Input/Output Columns
The label column data must be [System.Boolean](xref:System.Boolean). This trainer outputs the following columns:

| Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | <xref:System.Single> | The unbounded score that was calculated by the trainer to determine the prediction.|
| `PredictedLabel` | <xref:System.Boolean> | The label predicted by the trainer. `false` maps to negative score and   `true` maps to positive score.|
| `Probability` | <xref:System.Single> | The probability of the score in range [0, 1].|