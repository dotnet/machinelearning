### Training Algorithm Details
This trainer assumes that data points collected in the time series are independently sampled from the same distribution (independent identically distributed).
Thus, the value at the current timestamp can be viewed as the value at the next timestamp in expectation.
If the observed value at timestamp $t-1$ is $p$, the predicted value at $t$ timestamp would be $p$ as well.
