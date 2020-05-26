# [ML.NET](http://dot.net/ml) 1.5.0

## **New Features**
- **New anomaly detection algorithm** ([#5135](https://github.com/dotnet/machinelearning/pull/5135)). ML.NET has previously supported anomaly detection through [DetectAnomalyBySrCnn](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.timeseriescatalog.detectanomalybysrcnn?view=ml-dotnet). This function operates in a streaming manner by computing anomalies around each arriving point and examining a window around it. Now we introduce a new function `DetectEntireAnomalyBySrCnn` that computes anomalies by considering the entire dataset and also supports the ability to set sensitivity and output margin. 
- **Root Cause Detection** ([#4925](https://github.com/dotnet/machinelearning/pull/4925)) ML.NET now also supports root cause detection for anomalies detected in time series data. 

## **Enhancements**
- **Updates to TextLoader**
  - Enable TextLoader to accept new lines in quoted fields  ([#5125](https://github.com/dotnet/machinelearning/pull/5125))
  - Add escapeChar support to TextLoader ([#5147](https://github.com/dotnet/machinelearning/pull/5147))
  - Add public generic methods to TextLoader catalog that accept Options objects ([#5134](https://github.com/dotnet/machinelearning/pull/5134))
  - Added decimal marker option in TextLoader ([#5145](https://github.com/dotnet/machinelearning/pull/5145), [#5154](https://github.com/dotnet/machinelearning/pull/5154))
- Onnxruntime updated to v1.3 ([#5104](https://github.com/dotnet/machinelearning/pull/5104)). This brings support for additional data types for the HashingEstimator.
- Onnx export for OneHotHashEncodingTransformer and HashingTransormer ([#5013](https://github.com/dotnet/machinelearning/pull/5013), [#5152](https://github.com/dotnet/machinelearning/pull/5152), [#5138](https://github.com/dotnet/machinelearning/pull/5138))
- Support for Categorical features in CalculateFeatureContribution of LightGBM ([#5018](https://github.com/dotnet/machinelearning/pull/5018))
  

## **Bug Fixes**
In this release we have traced down every bug that would occur randomly and sporadically and fixed many subtle bugs. As a result, we have also re-enabled a lot of tests listed in the **Test Updates** section below. 
- Fixed race condition for test MulticlassTreeFeaturizedLRTest ([#4950](https://github.com/dotnet/machinelearning/pull/4950))
- Fix SsaForecast bug ([#5023](https://github.com/dotnet/machinelearning/pull/5023))
- Fixed x86 crash ([#5081](https://github.com/dotnet/machinelearning/pull/5081))
- Fixed and added unit tests for EnsureResourceAsync hanging issue ([#4943](https://github.com/dotnet/machinelearning/pull/4943))
- Added IDisposable support for several classes ([#4939](https://github.com/dotnet/machinelearning/pull/4939))
- Updated libmf and corresponding MatrixFactorizationSimpleTrainAndPredict() baselines per build ([#5121](https://github.com/dotnet/machinelearning/pull/5121))
- Fix MatrixFactorization trainer's warning ([#5071](https://github.com/dotnet/machinelearning/pull/5071))
- Update CodeGenerator's console project to netcoreapp3.1 ([#5066](https://github.com/dotnet/machinelearning/pull/5066))
- Let ImageLoadingTransformer dispose the last image it loads ([#5056](https://github.com/dotnet/machinelearning/pull/5056))
- [LightGBM] Fixed bug for empty categorical values ([#5048](https://github.com/dotnet/machinelearning/pull/5048))
- Converted potentially large variables to type long ([#5041](https://github.com/dotnet/machinelearning/pull/5041))
- Made resource downloading more robust ([#4997](https://github.com/dotnet/machinelearning/pull/4997))
- Updated MultiFileSource.Load to fix inconsistent behavior with multiple files ([#5003](https://github.com/dotnet/machinelearning/pull/5003))
- Removed WeakReference<IHosts> already cleaned up by GC ([#4995](https://github.com/dotnet/machinelearning/pull/4995))
- Fixed Bitmap(file) locking the file. ([#4994](https://github.com/dotnet/machinelearning/pull/4994))
- Remove WeakReference list in PredictionEnginePoolPolicy. ([#4992](https://github.com/dotnet/machinelearning/pull/4992))
- Added the assembly name of the custom transform to the model file ([#4989](https://github.com/dotnet/machinelearning/pull/4989))
- Updated constructor of ImageLoadingTransformer to accept empty imageFolder paths ([#4976](https://github.com/dotnet/machinelearning/pull/4976))

**Onnx bug fixes**
- ColumnSelectingTransformer now infers ONNX shape ([#5079](https://github.com/dotnet/machinelearning/pull/5079))
- Fixed KMeans scoring differences between ORT and OnnxRunner ([#4942](https://github.com/dotnet/machinelearning/pull/4942))
- CountFeatureSelectingEstimator no selection support ([#5000](https://github.com/dotnet/machinelearning/pull/5000))
- Fixes OneHotEncoding Issue ([#4974](https://github.com/dotnet/machinelearning/pull/4974))
- Fixes multiclass logistic regression ([#4963](https://github.com/dotnet/machinelearning/pull/4963))
- Adding vector tests for KeyToValue and ValueToKey ([#5090](https://github.com/dotnet/machinelearning/pull/5090))

**AutoML fixes**
- Handle NaN optimization metric in AutoML ([#5031](https://github.com/dotnet/machinelearning/pull/5031))
- Add projects capability in CodeGenerator ([#5002](https://github.com/dotnet/machinelearning/pull/5002))
- Simplify CodeGen - phase 2 ([#4972](https://github.com/dotnet/machinelearning/pull/4972))
- Support sweeping multiline option in AutoML ([#5148](https://github.com/dotnet/machinelearning/pull/5148))


## **Test updates**
- Fix libomp installation for MacOS Builds([#5143](https://github.com/dotnet/machinelearning/pull/5143), [#5141](https://github.com/dotnet/machinelearning/pull/5141)) 
- address TF test download fail, use resource manager with retry download ([#5102](https://github.com/dotnet/machinelearning/pull/5102))
- Adding OneHotHashEncoding Test ([#5098](https://github.com/dotnet/machinelearning/pull/5098))
- Changed Dictionary to ConcurrentDictionary ([#5097](https://github.com/dotnet/machinelearning/pull/5097))
- Added SQLite database to test loading of datasets in non-Windows builds ([#5080](https://github.com/dotnet/machinelearning/pull/5080))
- Added ability to compare configuration specific baselines, updated baslines for many tests and re-enabled disabled tests ([#5045](https://github.com/dotnet/machinelearning/pull/5045), [#5059](https://github.com/dotnet/machinelearning/pull/5059), [#5068](https://github.com/dotnet/machinelearning/pull/5068), [#5057](https://github.com/dotnet/machinelearning/pull/5057), [#5047](https://github.com/dotnet/machinelearning/pull/5047), [#5029](https://github.com/dotnet/machinelearning/pull/5029), [#5094](https://github.com/dotnet/machinelearning/pull/5094), [#5060](https://github.com/dotnet/machinelearning/pull/5060))
- Fixed TestCancellation hanging ([#4999](https://github.com/dotnet/machinelearning/pull/4999))
- fix benchmark test hanging issue ([#4985](https://github.com/dotnet/machinelearning/pull/4985))
- Added working version of checking whether file is available for access ([#4938](https://github.com/dotnet/machinelearning/pull/4938))

## **Documentation Updates**
- Update OnnxTransformer Doc XML ([#5085](https://github.com/dotnet/machinelearning/pull/5085))
- Updated build docs for .NET Core 3.1 ([#4967](https://github.com/dotnet/machinelearning/pull/4967))
- Updated OnnxScoringEstimator's documentation ([#4966](https://github.com/dotnet/machinelearning/pull/4966))
- Fix xrefs in the LDSVM trainer docs ([#4940](https://github.com/dotnet/machinelearning/pull/4940))
- Clarified parameters on time series ([#5038](https://github.com/dotnet/machinelearning/pull/5038))
- Update ForecastBySsa function specifications and add seealso ([#5027](https://github.com/dotnet/machinelearning/pull/5027))
- Add see also section to TensorFlowEstimator docs ([#4941](https://github.com/dotnet/machinelearning/pull/4941))


## **Breaking Changes**
- None



