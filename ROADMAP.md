# The ML.NET Roadmap

The goal of the ML.NET project is to make .NET developers great at machine learning. This document describes the plan for the project.

ML.NET is a community effort and we welcome community feedback on our plans. The best way to give feedback is to open an issue in this repo.

We also invite contributions.  The [up-for-grabs issues](https://github.com/dotnet/machinelearning/issues?q=is%3Aopen+is%3Aissue+label%3Aup-for-grabs) on GitHub are a good place to start.

## Goals through June 30, 2020
### Test stability
Continuous integration builds currently have a 30% pass rate. We aim to get this pass rate up to at least 80%.

### Streaming metrics
Currently, the way ML.NET computes [metrics](https://docs.microsoft.com/dotnet/machine-learning/resources/metrics) is memory-intensive. We will compute metrics in a streaming fashion instead, thereby reducing memory consumption.

### Multivariate anomaly detection
ML.NET already supports [univariate anomaly detection](https://docs.microsoft.com/dotnet/api/microsoft.ml.timeseriescatalog.detectanomalybysrcnn?view=ml-dotnet), but we will add the ability to detect anomalies in multiple variables over time.

### ONNX Runtime exportability

We will expand the number of ML.NET transforms and estimators that are exportable to the [ONNX Runtime](https://github.com/Microsoft/onnxruntime).
