### Anomaly Scorer
Once the raw score at a timestamp is computed, it is fed to the anomaly scorer component to calculate the final anomaly score at that timestamp.

#### Spike detection based on p-value
The p-value score indicates whether the current point is an outlier (also known as a spike).
The lower its value, the more likely it is a spike. The p-value score is always in $[0, 1]$.

This score is the p-value of the current computed raw score according to a distribution of raw scores.
Here, the distribution is estimated based on the most recent raw score values up to certain depth back in the history.
More specifically, this distribution is estimated using [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) with the Gaussian [kernels](https://en.wikipedia.org/wiki/Kernel_(statistics)#In_non-parametric_statistics) of adaptive bandwidth.

If the p-value score exceeds $1 - \frac{\text{confidence}}{100}$, the associated timestamp may get a non-zero alert value in spike detection, which means a spike point is detected.
Note that $\text{confidence}$ is defined in the signatures of [DetectIidSpike](xref:Microsoft.ML.TimeSeriesCatalog.DetectIidSpike(Microsoft.ML.TransformsCatalog,System.String,System.String,System.Int32,System.Int32,Microsoft.ML.Transforms.TimeSeries.AnomalySide))
and [DetectSpikeBySsa](xref:Microsoft.ML.TimeSeriesCatalog.DetectSpikeBySsa(Microsoft.ML.TransformsCatalog,System.String,System.String,System.Int32,System.Int32,System.Int32,System.Int32,Microsoft.ML.Transforms.TimeSeries.AnomalySide,Microsoft.ML.Transforms.TimeSeries.ErrorFunction)).