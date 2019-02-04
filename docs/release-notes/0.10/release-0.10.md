# ML.NET 0.10 Release Notes

For the past 10 months we have been adding features and improving [ML.NET](https://aka.ms/mlnet). 

In this 0.10 release and the forthcoming 0.11, and 0.12 releases before we reach 1.0, we are focusing on the overall stability of the framework, continuing to refine the API, increase test coverage and improve documentation. 

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

Below are a few of the highlights from 0.10 release. There are many other improvements and bug fixes in the API.

* Added support for returning multiple ranked results when scoring a multi-class classification model. 
([#2250](https://github.com/dotnet/machinelearning/pull/2250))
  - This allows to classify something into more than one category, for instance, assign a product to multiple categories, not just one
  - The code to access this metadata will be simplified in upcoming releases, though.  


* DataView segregated into a single assembly and NuGet package 
([#2220](https://github.com/dotnet/machinelearning/pull/2220))


* Introducing Microsoft.ML.Recommender NuGet name instead of Microsoft.ML.MatrixFactorization name
([#2081](https://github.com/dotnet/machinelearning/pull/2081)) 
  - Better naming for NuGet packages based on the scenario (Recommendations) instead of the trainer's name

* Support multiple 'feature columns' in FFM (Field Factorization Machine)
([#2205](https://github.com/dotnet/machinelearning/pull/2205)) 
   - Allows multiple feature column names in advanced trainer arguments so certain FFM trainers can support multiple multiple feature columns as explained in [#2179](https://github.com/dotnet/machinelearning/issues/2179) issue.

* Added support for loading map from file through dataview by using ValueMapperTransformer.
([#2232](https://github.com/dotnet/machinelearning/pull/2232)) 
   - This provides support for additional scenarios like a Text/NLP scenario ([#747](https://github.com/dotnet/machinelearning/issues/747)) in TensorFlowTransform where model's expected input is vector of integers.

* Added support for running benchmarks on .NET Framework in addition to .NET Core.
([#2157](https://github.com/dotnet/machinelearning/pull/2157)) 
   - Benchmarks can be based on [Microsoft.ML.Benchmarks](https://github.com/dotnet/machinelearning/tree/master/test/Microsoft.ML.Benchmarks).
   - This fixes issues like [#1945](https://github.com/dotnet/machinelearning/issues/1945)

* Added Tensorflow unfrozen models support in GetModelSchema 
([#2112](https://github.com/dotnet/machinelearning/pull/2112)) 
   - Fixes issues like [#2102](https://github.com/dotnet/machinelearning/issues/2102)

  
### Acknowledgements

Kudos to [Ivanidzo4ka](https://github.com/Ivanidzo4ka), [eerhardt](https://github.com/eerhardt), [wschin](https://github.com/wschin), [zeahmed](https://github.com/zeahmed), [Anipik](https://github.com/Anipik) and the [ML.NET](https://aka.ms/mlnet) team for their
contributions as part of this release!