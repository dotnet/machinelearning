### Input and Output Columns
The input features column data must be a known-sized vector of <xref:System.Single>. This trainer outputs the following columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Score` | <xref:System.Single> | The non-negative, unbounded score that was calculated by the anomaly detection model.|
| `PredictedLabel` | <xref:System.Boolean> | The predicted label, based on the threshold. A score higher than the threshold maps to `true` and a score lower than the threshold maps to `false`. The default threshold is `0.5`.Use <xref:AnomalyDetectionCatalog.ChangeModelThreshold> to change the default value.|