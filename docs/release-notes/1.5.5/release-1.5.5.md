# [ML.NET](http://dot.net/ml) 1.5.5

## **New Features**
- **New API allowing confidence parameter to be a double**.([#5623](https://github.com/dotnet/machinelearning/pull/5623))
. A new API has been added to accept double type for the confidence level. This helps when you need to have higher precision than an int will allow for. (**Thank you @esso23**)
- **Support to export ValueMapping estimator to ONNX was added** ([#5577](https://github.com/dotnet/machinelearning/pull/5577))
- **New API to treat TensorFlow output as batched/not-batched** ([#5634](https://github.com/dotnet/machinelearning/pull/5634)) A new API has been added so you can specify if the output from TensorFlow is batched or not.


## **Enhancements**
- Make ColumnInference serializable ([#5611](https://github.com/dotnet/machinelearning/pull/5611))


## **Bug Fixes**
- **AutoML.NET specific fixes**.
  - Fixed an AutoML aggregate timeout exception ([#5631](https://github.com/dotnet/machinelearning/pull/5631))
  - Offer suggestions for possibly mistyped label column names in AutoML ([#5624](https://github.com/dotnet/machinelearning/pull/5624)) (**Thank you @Crabzmatic**)
- Update some ToString conversions ([#5627](https://github.com/dotnet/machinelearning/pull/5627)) (**Thanks @4201104140**)
- Fixed an issue in SRCnnEntireAnomalyDetector ([#5579](https://github.com/dotnet/machinelearning/pull/5579))
- Fixed nuget.config multi-feed issue ([#5614](https://github.com/dotnet/machinelearning/pull/5614))
- Remove references to Microsoft.ML.Scoring ([#5602](https://github.com/dotnet/machinelearning/pull/5602))
- Fixed Averaged Perceptron default value ([#5586](https://github.com/dotnet/machinelearning/pull/5586))


## **Build / Test updates**
- Fixing official build by adding homebrew bug workaround ([#5596](https://github.com/dotnet/machinelearning/pull/5596))
- Nuget.config url fix for roslyn compilers ([#5584](https://github.com/dotnet/machinelearning/pull/5584))
- Add SymSgdNative reference to AutoML.Tests.csproj ([#5559](https://github.com/dotnet/machinelearning/pull/5559))


## **Documentation Updates**
- Updated documentation for the correct version of CUDA for TensorFlow. ([#5635](https://github.com/dotnet/machinelearning/pull/5635))
- Updates documentation for an issue with brew and installing libomp. ([#5635](https://github.com/dotnet/machinelearning/pull/5635))
- Updated an ONNX url to the correct url.  ([#5635](https://github.com/dotnet/machinelearning/pull/5635))
- Added a note in the documentation that the PredictionEngine is not thread safe. ([#5583](https://github.com/dotnet/machinelearning/pull/5583))


## **Breaking Changes**
- None
