# ML.NET 0.11 Release Notes

[ML.NET](https://aka.ms/mlnet) 0.11 will be the last preview release before we reach `Release Candidate` for v1. We continue to push for creating a coherent and clean API surface for [ML.NET](https://aka.ms/mlnet) users. This release includes several bug fixes as well as extensive work on reducing the public API surface. The work on the API is tracked via [this project](https://github.com/dotnet/machinelearning/projects/13). In the upcoming 0.12 (RC1) releases before we reach 1.0, we will continue on refining the API and improving documentation.

### Installation

ML.NET supports Windows, MacOS, and Linux. See [supported OS versions of .NET Core 2.0](https://github.com/dotnet/core/blob/master/release-notes/2.0/2.0-supported-os.md) for more details.

You can install ML.NET NuGet from the CLI using:
```
dotnet add package Microsoft.ML
```

From package manager:
```
Install-Package Microsoft.ML
```

### Release Notes

Below are a few of the highlights from this release. There are many other improvements in the API.

* Creation of components through MLContext: advanced options and other feedback. ([#1798](https://github.coalsom/dotnet/machinelearning/issues/1798))
* Several issues closed on internalizing public surface. ([19 issues](https://github.com/dotnet/machinelearning/issues?q=is%3Aissue+lockdown+is%3Aclosed))
* Stop using MEF as part of the public API. ([#2422](https://github.com/dotnet/machinelearning/issues/2422))
* Several `NameSpace` changes. ([#2326](https://github.com/dotnet/machinelearning/issues/2326))
* `ONNX` is now `ONNXConverter`. ([#2625](https://github.com/dotnet/machinelearning/pull/2625))
* `ONNXTransform` is now `ONNXTransformer`. ([#2544](https://github.com/dotnet/machinelearning/pull/2544))
* `FastTree` has it's own package now. ([#2752](https://github.com/dotnet/machinelearning/issues/2752))
* `Ensemble` has been moved out of `Microsoft.ML` package. ([#2717](https://github.com/dotnet/machinelearning/issues/2717))
* Add support for string types in TensorFlowTransformer. ([#2545](https://github.com/dotnet/machinelearning/issues/2545))
* Make FastTree/LightGBM learned model suitable for public consumption. ([#1960](https://github.com/dotnet/machinelearning/issues/1960))


### Acknowledgements

Shoutout to [PaulTFreedman](https://github.com/PaulTFreedman),
[kant2002](https://github.com/kant2002),
[jwood803](https://github.com/jwood803), [mareklinka](https://github.com/mareklinka), [elbruno](https://github.com/elbruno) and the [ML.NET](https://aka.ms/mlnet) team for their
contributions as part of this release!
