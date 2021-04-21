# ML.NET 0.10 Release Notes

[ML.NET](https://aka.ms/mlnet) 0.10 brings us one step closer to the stable v1 release. We understand that the API surface has been changing rapidly and we deeply appreciate the amazing support from [ML.NET](https://aka.ms/mlnet) community. These changes are necessary for the support of the stable API for many years to come. In the upcoming 0.11, and 0.12 releases before we reach 1.0, we will continue on refining the API and improving documentation.

We have also instrumented [code coverage](https://codecov.io/gh/dotnet/machinelearning) tools as part of our CI systems and will continue to push for stability and quality in the code.

One of the milestones that we have achieved in this release is moving `IDataView` into a new and separate assembly under `Microsoft.Data.DataView` namespace. For detailed documentation on `IDataView` please take a look at [IDataView design principles](https://github.com/dotnet/machinelearning/blob/main/docs/code/IDataViewDesignPrinciples.md).

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

* DataView moved into a separate assembly and NuGet package
([#2220](https://github.com/dotnet/machinelearning/pull/2220))

* Improvements in the API for prediction engine
([#2250](https://github.com/dotnet/machinelearning/pull/2250))

* Introducing Microsoft.ML.Recommender NuGet name instead of Microsoft.ML.MatrixFactorization name
([#2081](https://github.com/dotnet/machinelearning/pull/2081))
  - Better naming for NuGet packages based on the scenario (Recommendations) instead of the trainer's name

* Support multiple 'feature columns' in FFM (Field-aware Factorization Machines)
([#2205](https://github.com/dotnet/machinelearning/pull/2205))
  - Allows multiple feature column names in advanced trainer arguments so certain FFM trainers can support multiple multiple feature columns as explained in [#2179](https://github.com/dotnet/machinelearning/issues/2179) issue

* Added support for loading map from file through dataview by using ValueMapperTransformer
([#2232](https://github.com/dotnet/machinelearning/pull/2232))
  - This provides support for additional scenarios like a Text/NLP scenario ([#747](https://github.com/dotnet/machinelearning/issues/747)) in TensorFlowTransform where model's expected input is vector of integers

* Added support for running benchmarks on .NET Framework in addition to .NET Core.
([#2157](https://github.com/dotnet/machinelearning/pull/2157))
  - Benchmarks can be based on [Microsoft.ML.Benchmarks](https://github.com/dotnet/machinelearning/tree/main/test/Microsoft.ML.Benchmarks)
  - This fixes issues like [#1945](https://github.com/dotnet/machinelearning/issues/1945)

* Added Tensorflow unfrozen models support in GetModelSchema
([#2112](https://github.com/dotnet/machinelearning/pull/2112))
  - Fixes issue [#2102](https://github.com/dotnet/machinelearning/issues/2102)

* Providing API for properly inspecting trees ([#2243](https://github.com/dotnet/machinelearning/pull/2243))

### Acknowledgements

Shoutout to [endintiers](https://github.com/endintiers),
[hvitved](https://github.com/hvitved),
[mareklinka](https://github.com/mareklinka), [kilick](https://github.com/kilick), and the [ML.NET](https://aka.ms/mlnet) team for their
contributions as part of this release!