# [ML.NET](http://dot.net/ml) 1.4.0-preview2

## **New Features**
- **Deep Neural Networks Training (0.16.0-preview2)**

  Improves the in-preview `ImageClassification` API further:
  - Early stopping feature stops the training when optimal accuracy is reached ([#4237](https://github.com/dotnet/machinelearning/pull/4237))
  - Enables inferencing on in-memory images ([#4242](https://github.com/dotnet/machinelearning/pull/4242))
  - `PredictedLabel` output column now contains actual class labels instead of `uint32` class index values ([#4228](https://github.com/dotnet/machinelearning/pull/4228))
  - GPU support on Windows and Linux ([#4270](https://github.com/dotnet/machinelearning/pull/4270), [#4277](https://github.com/dotnet/machinelearning/pull/4277))
  - Upgraded [TensorFlow .NET](https://github.com/SciSharp/TensorFlow.NET) version to 0.11.3 ([#4205](https://github.com/dotnet/machinelearning/pull/4205))

  [In-memory image inferencing sample](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/ImageClassification/ResnetV2101TransferLearningTrainTestSplit.cs)
  [Early stopping sample](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/ImageClassification/ResnetV2101TransferLearningEarlyStopping.cs)
  [GPU samples](https://github.com/dotnet/machinelearning/tree/main/docs/samples/Microsoft.ML.Samples.GPU)

- **New ONNX Exporters (1.4.0-preview2)**
  - LpNormNormalizing transformer ([#4161](https://github.com/dotnet/machinelearning/pull/4161))
  - PCA transformer ([4188](https://github.com/dotnet/machinelearning/pull/4188))
  - TypeConverting transformer ([#4155](https://github.com/dotnet/machinelearning/pull/4155))
  - MissingValueIndicator transformer ([#4194](https://github.com/dotnet/machinelearning/pull/4194))

## **Bug Fixes**
- OnnxSequenceType and ColumnName attributes together doesn't work ([#4187](https://github.com/dotnet/machinelearning/pull/4187))
- Fix memory leak in TensorflowTransformer ([#4223](https://github.com/dotnet/machinelearning/pull/4223))
- Enable permutation feature importance to be used with model loaded from disk ([#4262](https://github.com/dotnet/machinelearning/pull/4262))
- `IsSavedModel` returns true when loaded TensorFlow model is a frozen model ([#4262](https://github.com/dotnet/machinelearning/pull/4197))
- Exception when using `OnnxSequenceType` attribute directly without specify sequence type ([#4272](https://github.com/dotnet/machinelearning/pull/4272), [#4297](https://github.com/dotnet/machinelearning/pull/4297))

## **Samples**
- TensorFlow full model retrain sample ([#4127](https://github.com/dotnet/machinelearning/pull/4127))

## **Breaking Changes**
None.

## **Obsolete API**
- `OnnxSequenceType` attribute that doesn't take a type ([#4272](https://github.com/dotnet/machinelearning/pull/4272), [#4297](https://github.com/dotnet/machinelearning/pull/4297))

## **Enhancements**
- Improve exception message in LightGBM ([#4214](https://github.com/dotnet/machinelearning/pull/4214))
- FeaturizeText should allow only outputColumnName to be defined ([#4211](https://github.com/dotnet/machinelearning/pull/4211))
- Fix NgramExtractingTransformer GetSlotNames to not allocate a new delegate on every invoke ([#4247](https://github.com/dotnet/machinelearning/pull/4247))
- Resurrect broken code coverage build and re-enable code coverage for pull request ([#4261](https://github.com/dotnet/machinelearning/pull/4261))
- NimbusML entrypoint for permutation feature importance ([#4232](https://github.com/dotnet/machinelearning/pull/4232))
- Reuse memory when copying outputs from TensorFlow graph ([#4260](https://github.com/dotnet/machinelearning/pull/4260))
- DateTime to DateTime standard conversion ([#4273](https://github.com/dotnet/machinelearning/pull/4273))
- CodeCov version upgraded to 1.7.2 ([#4291](https://github.com/dotnet/machinelearning/pull/4291))

## **CLI and AutoML API**
None.

## **Remarks**
None.





