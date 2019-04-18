### Input and Output Columns
The input label column data must be <xref:System.Boolean>.
The input features column data must be a known-sized vector of <xref:System.Single>. This trainer outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | <xref:System.Single> | The unbounded score that was calculated by the predictor to determine the prediction.|
| `PredictedLabel` | <xref:System.Boolean> | The label predicted by the predictor. `false` maps to negative score and `true` maps to positive score.|
| `Probability` | <xref:System.Single> | The probability calculated by the predictor of having true as the label. Probability value is in range [0, 1].||