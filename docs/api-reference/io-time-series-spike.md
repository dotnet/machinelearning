### Input and Output Columns
There is only one input column.
The input column must be <xref:System.Single> where a <xref:System.Single> value indicates a value at a timestamp in the time series.

It produces a column that is a vector with 3 elements.
The output vector sequentially contains alert level (non-zero value means a change point), score, and p-value.
