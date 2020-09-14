# [ML.NET](http://dot.net/ml) 1.5.2

## **New Features**
- **New API and algorithms for time series data**. In this release ML.NET introduces new capabilities for working with time series data.
  - Detecting seasonality in time series  ([#5231](https://github.com/dotnet/machinelearning/pull/5231))
  - Removing seasonality from time series prior to anomaly detection ([#5202](https://github.com/dotnet/machinelearning/pull/5202))
  - Threshold for root cause analysis ([#5218](https://github.com/dotnet/machinelearning/pull/5218))
  - RCA for anomaly detection can now return multiple dimensions([#5236](https://github.com/dotnet/machinelearning/pull/5236))
- **Ranking experiments in AutoML.NET API**. ML.NET now adds support for automating ranking experiments. ([#5150](https://github.com/dotnet/machinelearning/pull/5150), [#5246](https://github.com/dotnet/machinelearning/pull/5246)) Corresponding support will soon be added to [Model Builder](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet/model-builder) in Visual Studio.
- **Cross validation support in ranking** ([#5263](https://github.com/dotnet/machinelearning/pull/5263))
- **CountTargetEncodingEstimator**. This transforms a categorical column into a set of features that includes the count of each label class, the log-odds for each label class and the back-off indicator ([#4514](https://github.com/dotnet/machinelearning/pull/4514))

## **Enhancements**
- **Onnx Enhancements**
  - Support more types for ONNX export of HashEstimator ([#5104](https://github.com/dotnet/machinelearning/pull/5104))
  - Added ONNX export support for NaiveCalibrator ([#5289](https://github.com/dotnet/machinelearning/pull/5289))
  - Added ONNX export support for  StopWordsRemovingEstimator and CustomStopWordsRemovingEstimator ([#5279](https://github.com/dotnet/machinelearning/pull/5279))
  - Support onnx export with previous OpSet version ([#5176](https://github.com/dotnet/machinelearning/pull/5176))
  - Added a sample for Onnx conversion ([#5195](https://github.com/dotnet/machinelearning/pull/5195))
- **New features in old transformers**
  - Robust Scaler now added to the Normalizer catalog ([#5166](https://github.com/dotnet/machinelearning/pull/5166))
  - ReplaceMissingValues now supports `Mode` as a replacement method. ([#5205](https://github.com/dotnet/machinelearning/pull/5205))
  - Added in standard conversions to convert types to string ([#5106](https://github.com/dotnet/machinelearning/pull/5106))
- Output topic summary to model file for LDATransformer ([#5260](https://github.com/dotnet/machinelearning/pull/5260))
- Use Channel Instead of BufferBlock ([#5123](https://github.com/dotnet/machinelearning/pull/5123), [#5313](https://github.com/dotnet/machinelearning/pull/5313)). (Thanks [**@jwood803**](https://github.com/jwood803))
- Support specifying command timeout while using the database loader ([#5288](https://github.com/dotnet/machinelearning/pull/5288))
- Added cross entropy support to validation training, edited metric reporting ([#5255](https://github.com/dotnet/machinelearning/pull/5255))
- Allow TextLoader to load empty float/double fields as NaN instead of 0 ([#5198](https://github.com/dotnet/machinelearning/pull/5198))


## **Bug Fixes**
- Changed default value of RowGroupColumnName from null to GroupId ([#5290](https://github.com/dotnet/machinelearning/pull/5290))
- Updated AveragedPerceptron default iterations from 1 to 10 ([#5258](https://github.com/dotnet/machinelearning/pull/5258))
- Properly normalize column names in Utils.GetSampleData() for duplicate cases ([#5280](https://github.com/dotnet/machinelearning/pull/5280))
- Add two-variable scenario in Tensor shape inference for TensorflowTransform ([#5257](https://github.com/dotnet/machinelearning/pull/5257))
- Fixed score column name and order bugs in CalibratorTransformer ([#5261](https://github.com/dotnet/machinelearning/pull/5261))
- Fix for conditional error in root cause analysis additions ([#5269](https://github.com/dotnet/machinelearning/pull/5269))
- Ensured Sanitized Column Names are Unique in AutoML CLI ([#5177](https://github.com/dotnet/machinelearning/pull/5177))
- Ensure that the graph is set to be the current graph when scoring with multiple models ([#5149](https://github.com/dotnet/machinelearning/pull/5149))
- Uniform onnx conversion method when using non-default column names ([#5146](https://github.com/dotnet/machinelearning/pull/5146))
- Fixed multiple issues related to splitting data. ([#5227](https://github.com/dotnet/machinelearning/pull/5227))
- Changed default NGram length from 1 to 2. ([#5248](https://github.com/dotnet/machinelearning/pull/5248))
- Improve exception msg by adding column name ([#5232](https://github.com/dotnet/machinelearning/pull/5232))
- Use model schema type instead of class definition schema ([#5228](https://github.com/dotnet/machinelearning/pull/5228))
- Use GetRandomFileName when creating random temp folder to avoid conflict ([#5229](https://github.com/dotnet/machinelearning/pull/5229))
- Filter anomalies according to boundaries under AnomalyAndMargin mode ([#5212](https://github.com/dotnet/machinelearning/pull/5212))
- Improve error message when defining custom type for variables ([#5114](https://github.com/dotnet/machinelearning/pull/5114))
- Fixed OnnxTransformer output column mapping. ([#5192](https://github.com/dotnet/machinelearning/pull/5192))
- Fixed version format of built packages ([#5197](https://github.com/dotnet/machinelearning/pull/5197))
- Improvements to "Invalid TValue" error message ([#5189](https://github.com/dotnet/machinelearning/pull/5189))
- Added IDisposable to OnnxTransformer and fixed memory leaks ([#5348](https://github.com/dotnet/machinelearning/pull/5348))
- Fixes [#4392](https://github.com/dotnet/machinelearning/issues/4392). Added AddPredictionEnginePool overload for implementation factory ([#4393](https://github.com/dotnet/machinelearning/pull/4393))
- Updated codegen to make it work with mlnet 1.5  ([#5173](https://github.com/dotnet/machinelearning/pull/5173))
- Updated codegen to support object detection scenario. ([#5216](https://github.com/dotnet/machinelearning/pull/5216))
- Fix issue [#5350](https://github.com/dotnet/machinelearning/issues/5350), check file lock before reload model ([#5351](https://github.com/dotnet/machinelearning/pull/5351))
- Improve handling of infinity values in AutoML.NET when calculating average CV metrics ([#5345](https://github.com/dotnet/machinelearning/pull/5345))
- Throw when PCA generates invalid eigenvectors ([#5349](https://github.com/dotnet/machinelearning/pull/5349))
- RobustScalingNormalizer entrypoint added ([#5310](https://github.com/dotnet/machinelearning/pull/5310))
- Replace whitelist terminology to allow list ([#5328](https://github.com/dotnet/machinelearning/pull/5328)) (Thanks [**@LetticiaNicoli**](https://github.com/LetticiaNicoli))
- Fixes ([#5352](https://github.com/dotnet/machinelearning/issues/5352)) issues caused by equality with non-string values for root cause localization  ([#5354](https://github.com/dotnet/machinelearning/pull/5354))
- Added catch in R^2 calculation for case with few samples ([#5319](https://github.com/dotnet/machinelearning/pull/5319))
- Added support for RankingMetrics with CrossValSummaryRunner ([#5386](https://github.com/dotnet/machinelearning/pull/5386))


## **Test updates**
- Refactor of OnnxConversionTests.cs ([#5185](https://github.com/dotnet/machinelearning/pull/5185))
- New code coverage ([#5169](https://github.com/dotnet/machinelearning/pull/5169))
- Test fix using breastcancel dataset and test cleanup ([#5292](https://github.com/dotnet/machinelearning/pull/5292))

## **Documentation Updates**
- Updated ORT version info for OnnxScoringEstimator ([#5175](https://github.com/dotnet/machinelearning/pull/5175))
- Updated OnnxTransformer docs ([#5296](https://github.com/dotnet/machinelearning/pull/5296))
- Improve VectorTypeAttribute(dims) docs ([#5301](https://github.com/dotnet/machinelearning/pull/5301))

## **Breaking Changes**
- None
