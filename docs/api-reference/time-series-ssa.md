### Training Algorithm Details
This class implements the general anomaly detection transform based on [Singular Spectrum Analysis (SSA)](https://en.wikipedia.org/wiki/Singular_spectrum_analysis).
SSA is a powerful framework for decomposing the time-series into trend, seasonality and noise components as well as forecasting the future values of the time-series.
In principle, SSA performs spectral analysis on the input time-series where each component in the spectrum corresponds to a trend, seasonal or noise component in the time-series.
For details of the Singular Spectrum Analysis (SSA), refer to [this document](http://arxiv.org/pdf/1206.6910.pdf).
