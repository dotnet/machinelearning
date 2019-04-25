### Input and Output Columns
The input label column data must be <xref:System.Boolean>.
The input features column data must be a known-sized vector of <xref:System.Single>. This trainer outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | <xref:System.Single> | The unbounded score that was calculated by the model.|
| `PredictedLabel` | <xref:System.Boolean> | The predicted label, based on the sign of the score. A negative score maps to `false` and a positive score maps to `true`.|
