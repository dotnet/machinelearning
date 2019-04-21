### Input and Output Columns
There is only one input column and its type is <xref:System.Single>.
This estimator adds the following output columns:

| Output Column Name | Column Type | Description|
| -- | -- | -- |
| `Prediction` | 3-element vector of <xref:System.Double> | It sequentially contains alert level (non-zero value means a change point), score, and p-value. |
