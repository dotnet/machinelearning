### Anomaly Scorer
Once the raw score at a timestamp is computed, it is fed to the anomaly scorer component to calculate the final anomaly score at that timestamp.
There are two statistics involved in this scorer, p-value and martingale score.

#### P-value score
The p-value score indicates the p-value of the current computed raw score according to a distribution of raw scores.
Here, the distribution is estimated based on the most recent raw score values up to certain depth back in the history.
More specifically, this distribution is estimated using [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation)
with the Gaussian [kernels](https://en.wikipedia.org/wiki/Kernel_(statistics)#In_non-parametric_statistics) of adaptive bandwidth.
The p-value score is always in $[0, 1]$, and the lower its value, the more likely the current point is an outlier (also known as a spike).


#### Change point detection based on martingale score
The martingale score is an extra level of scoring that is built upon the p-value scores.
The idea is based on the [Exchangeability Martingales](https://arxiv.org/pdf/1204.3251.pdf) that detect a change of distribution over a stream of i.i.d. values.
In short, the value of the martingale score starts increasing significantly when a sequence of small p-values detected in a row; this indicates the change of the distribution of the underlying data generation process.
Thus, the martingale score is used for change point detection.
Given a sequence of most recently observed p-values, $p1, \dots, p_n$, the martingale score is computed as:? $s(p1, \dots, p_n) = \prod_{i=1}^n  \beta(p_i)$.
There are two choices of $\beta$: $\beta(p) = e p^{\epsilon - 1}$ for $0 < \epsilon < 1$ or $\beta(p) = \int_{0}^1 \epsilon p^{\epsilon - 1} d\epsilon$.

If the martingle score exceeds $s(q_1, \dots, q_n)$ where $q_i=1 - \frac{\text{confidence}}{100}$, the associated timestamp may get a non-zero alert value for change point detection.
Note that $\text{confidence}$ is defined in the signatures of [DetectChangePointBySsa](xref:Microsoft.ML.TimeSeriesCatalog.DetectChangePointBySsa(Microsoft.ML.TransformsCatalog,System.String,System.String,System.Int32,System.Int32,System.Int32,System.Int32,Microsoft.ML.Transforms.TimeSeries.ErrorFunction,Microsoft.ML.Transforms.TimeSeries.MartingaleType,System.Double)) or
[DetectIidChangePoint](xref:Microsoft.ML.TimeSeriesCatalog.DetectIidChangePoint(Microsoft.ML.TransformsCatalog,System.String,System.String,System.Int32,System.Int32,Microsoft.ML.Transforms.TimeSeries.MartingaleType,System.Double)).
