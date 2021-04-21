# [ML.NET](http://dot.net/ml) 1.5.0-preview

## **New Features (IN-PREVIEW, please provide feedback)**
- **Export-to-ONNX for below components:**
    - WordTokenizingTransformer ([#4451](https://github.com/dotnet/machinelearning/pull/4451))
    - NgramExtractingTransformer ([#4451](https://github.com/dotnet/machinelearning/pull/4451))
    - OptionalColumnTransform ([#4454](https://github.com/dotnet/machinelearning/pull/4454))
    - KeyToValueMappingTransformer ([#4455](https://github.com/dotnet/machinelearning/pull/4455))
    - LbfgsMaximumEntropyMulticlassTrainer ([4462](https://github.com/dotnet/machinelearning/pull/4462))
    - LightGbmMulticlassTrainer ([4462](https://github.com/dotnet/machinelearning/pull/4462))
    - LightGbmMulticlassTrainer with SoftMax ([4462](https://github.com/dotnet/machinelearning/pull/4462))
    - OneVersusAllTrainer ([4462](https://github.com/dotnet/machinelearning/pull/4462))
    - SdcaMaximumEntropyMulticlassTrainer ([4462](https://github.com/dotnet/machinelearning/pull/4462))
    - SdcaNonCalibratedMulticlassTrainer ([4462](https://github.com/dotnet/machinelearning/pull/4462))
    - CopyColumn Transform ([#4486](https://github.com/dotnet/machinelearning/pull/4486))
    - PriorTrainer ([#4515](https://github.com/dotnet/machinelearning/pull/4515))

- **DateTime Transformer** ([#4521](https://github.com/dotnet/machinelearning/pull/4521))
- **Loader and Saver for [SVMLight file format](http://svmlight.joachims.org/)** ([#4190](https://github.com/dotnet/machinelearning/pull/4190))

  [Sample](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/LoadingSvmLight.cs)
- **Expression transformer** ([#4548](https://github.com/dotnet/machinelearning/pull/4548))
  The expression transformer takes the expression in the form of text using syntax of a simple expression language, and performs the operation defined in the expression on the input columns in each row of the data. The transformer supports having a vector input column, in which case it applies the expression to each slot of the vector independently. The expression language is extendable to user defined operations.

  [Sample](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Expression.cs)

## **Bug Fixes**
- Fix using permutation feature importance with Binary Prediction Transformer and CalibratedModelParametersBase loaded from disk. ([#4306](https://github.com/dotnet/machinelearning/pull/4306))
- Fixed model saving and loading of OneVersusAllTrainer to include SoftMax. ([#4472](https://github.com/dotnet/machinelearning/pull/4472))
- Ignore hidden columns in AutoML schema checks of validation data. ([#4490](https://github.com/dotnet/machinelearning/pull/4490))
- Ensure BufferBlocks are completed and empty in RowShufflingTransformer. ([#4479](https://github.com/dotnet/machinelearning/pull/4479))
- Create methods not being called when loading models from disk. ([#4485](https://github.com/dotnet/machinelearning/pull/4485))
- Fixes onnx exports for binary classification trainers. ([#4463](https://github.com/dotnet/machinelearning/pull/4463))
- Make PredictionEnginePool.GetPredictionEngine thread safe. ([#4570](https://github.com/dotnet/machinelearning/pull/4570))
- Memory leak when using FeaturizeText transform. ([#4576](https://github.com/dotnet/machinelearning/pull/4576))
- System.ArgumentOutOfRangeException issue in CustomStopWordsRemovingTransformer. ([#4592](https://github.com/dotnet/machinelearning/pull/4592))
- Image Classification low accuracy on EuroSAT Dataset. ([4522](https://github.com/dotnet/machinelearning/pull/4522))

## **Stability fixes by [Sam Harwell](https://github.com/sharwell)**
- Prevent exceptions from escaping FileSystemWatcher events. ([#4535](https://github.com/dotnet/machinelearning/pull/4535))
- Make local functions static where applicable. ([#4530](https://github.com/dotnet/machinelearning/pull/4530))
- Disable CS0649 in OnnxConversionTest. ([#4531](https://github.com/dotnet/machinelearning/pull/4531))
- Make test methods public. ([#4532](https://github.com/dotnet/machinelearning/pull/4532))
- Conditionally compile helper code. ([#4534](https://github.com/dotnet/machinelearning/pull/4534))
- Avoid running API Compat for design time builds. ([#4529](https://github.com/dotnet/machinelearning/pull/4529))
- Pass by reference when null is not expected. ([#4546](https://github.com/dotnet/machinelearning/pull/4546))
- Add Xunit.Combinatorial for test projects. ([#4545](https://github.com/dotnet/machinelearning/pull/4545))
- Use Theory to break up tests in OnnxConversionTest. ([#4533](https://github.com/dotnet/machinelearning/pull/4533))
- Update code coverage integration. ([#4543](https://github.com/dotnet/machinelearning/pull/4543))
- Use std::unique_ptr for objects in LdaEngine. ([#4547](https://github.com/dotnet/machinelearning/pull/4547))
- Enable VSTestBlame to show details for crashes. ([#4537](https://github.com/dotnet/machinelearning/pull/4537))
- Use std::unique_ptr for samplers_ and likelihood_in_iter_. ([#4551](https://github.com/dotnet/machinelearning/pull/4551))
- Add tests for IParameterValue implementations. ([#4549](https://github.com/dotnet/machinelearning/pull/4549))
- Convert LdaEngine to a SafeHandle. ([#4538](https://github.com/dotnet/machinelearning/pull/4538))
- Create SafeBoosterHandle and SafeDataSetHandle. ([#4539](https://github.com/dotnet/machinelearning/pull/4539))
- Add IterationDataAttribute. ([#4561](https://github.com/dotnet/machinelearning/pull/4561))
- Add tests for ParameterSet equality. ([#4550](https://github.com/dotnet/machinelearning/pull/4550))
- Add a test handler for AppDomain.UnhandledException. ([#4557](https://github.com/dotnet/machinelearning/commit/f1f8942a8272a9c87373d11bc89467461c8ecad1))

## **Breaking Changes**
None

## **Enhancements**
- Hash Transform API that takes in advanced options. ([#4443](https://github.com/dotnet/machinelearning/pull/4443))
- Image classification performance improvements and option to create validation set from train set. ([#4522](https://github.com/dotnet/machinelearning/pull/4522))
- Upgraded OnnxRuntime to v1.0 and Google Protobuf to 3.10.1. ([#4416](https://github.com/dotnet/machinelearning/pull/4416))

## **CLI and AutoML API**
  - None.

## **Remarks**
- Thank you, [Sam Harwell](https://github.com/sharwell) for making a series of stability fixes that has substantially increased the stability of our Build CI.





