# ML.NET 0.9 Release Notes

Welcome to 2019! For the past 9 months we have been working on adding features and improving [ML.NET](https://aka.ms/mlnet). In the remaining releases before we reach 1.0, we are focusing more on the overall stability of the package. To acheive that we will continue to work on refining the API, increasing test coverage and improving documentation. 0.9 release packs multiple fixes as well as significant clean up to the internal code of the package.

### Installation

ML.NET supports Windows, MacOS, and Linux. See [supported OS versions of .NET
Core
2.0](https://github.com/dotnet/core/blob/master/release-notes/2.0/2.0-supported-os.md)
for more details.

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

* Added Feature Contribution Calculation
  ([#1847](https://github.com/dotnet/machinelearning/pull/1847))

* Removed Legacy namespace that was marked obsolete.
  ([#2043](https://github.com/dotnet/machinelearning/pull/2043))

* GPU support for ONNX Transform.
  ([#1922](https://github.com/dotnet/machinelearning/pull/1922))

    * Cross platform support is still being developed for this feature.

* `Permutation Feature Importance` now supports confidence intervals.
  ([#1844](https://github.com/dotnet/machinelearning/pull/1844))

* Introducing `PredictionEngine` instead of `PredictionFunction`.
  ([#1920](https://github.com/dotnet/machinelearning/pull/1920))
  
### Acknowledgements

Shoutout to [dhilmathy](https://github.com/dhilmathy),
[mnboos](https://github.com/mnboos),
[robosek](https://github.com/robosek), and the [ML.NET](https://aka.ms/mlnet) team for their
contributions as part of this release!