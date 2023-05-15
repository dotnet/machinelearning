# [ML.NET](http://dot.net/ml) 3.0.0 Preview

## **New Features**
- **Add the ability to use Object Detection using TorchSharp** ([#6605](https://github.com/dotnet/machinelearning/pull/6605)) - We have added a new deep learning model back by TorchSharp that lets you fine tune your own Object Detection model!
- **Add SamplingKeyColumnName to AutoMLExperiment API** ([#6649](https://github.com/dotnet/machinelearning/pull/6649)) - You can now set the SamplingKeyColumnName when you are using AutoML. Thanks @torronen!
- **Add Object Detection to AutoML Sweeper** ([#6633](https://github.com/dotnet/machinelearning/pull/6633)) - Added Object Detection to the AutoML Sweeper so now they can be used together.
- **Add String Vector support to DataFrame** ([#6628](https://github.com/dotnet/machinelearning/pull/6628)) - Adds support for String Vectors in DataFrame. This also allows for Better IDataView <-> DataFrame conversions.
- **Add AutoZero tuner to BinaryClassification** ([#6615](https://github.com/dotnet/machinelearning/pull/6615)) - Can now user AutoZero tuner in AutoML Binary Classification experiments.
- **Added in fairness assessment and mitigation** ([#6539](https://github.com/dotnet/machinelearning/pull/6539)) - Support for fairness assessment and mitigation tool
- **Added in Support for some Intel OneDal Algorithms** ([#6521](https://github.com/dotnet/machinelearning/pull/6521)) - You can now use Intel's OneDal for some algorithms. This gives you access to some accelerated versions of these algorithms. The models are fully interoperable between ML.NET's normal models and these, so you can train with OneDal and then still run on machines where OneDal is not supported. Thanks @rgesteve!
- **Add in ability to have pre-defined weights for ngrams** ([#6458](https://github.com/dotnet/machinelearning/pull/6458)) - If you know the weights of your NGrams already you can now directly provide that.
- **Add SentenceSimilarity sweepable estimator in AutoML** ([#6445](https://github.com/dotnet/machinelearning/pull/6445)) - Can now use SentenceSimilarity with the sweepable estimator.
- **Add VBufferDataFrameCoumn to DataFrame** ([#6445](https://github.com/dotnet/machinelearning/pull/6445)) - Now DataFrame can support the VBuffer from ML.NET so the IDataView <-> DataFrame conversion can work with those types.
- **Added ADO.NET importing/exporting functionality to DataFrame** ([#5975](https://github.com/dotnet/machinelearning/pull/5975)) - Can now use ADO.NET import/export with DataFrames. Thanks @andrei-faber!

## **Enhancements**
- **Expose ExperimentSettings.MaxModel as public** ([#6663](https://github.com/dotnet/machinelearning/pull/6663)) - Exposes ExperimentSettings.MaxModel as public so now you can set the number of Max Models you want for an AutoML experiment.
- **Update to latest version of TorchSharp** ([#6636](https://github.com/dotnet/machinelearning/pull/6636)) - Updated to the latest version of TorchSharp and fixed any breaking changes so we can take advantage of their new features and bug fixes.
- **Update to latest version of Onnx Runtime** ([#6624](https://github.com/dotnet/machinelearning/pull/6624)) - Updated to the latest version of Onnx Runtime and fixed any breaking changes so we can take advantage of their new features and bug fixes.
- **Update ML.NET to compile with .NET8** ([#6641](https://github.com/dotnet/machinelearning/pull/6641)) - Removed some deprecated code now throws errors on .NET8 as well as other minor fixes to allow working/building with .NET8.
- **Added more logging to Object Detection** ([#6646](https://github.com/dotnet/machinelearning/pull/6646)) - Added more logging while Object Detection is training so even if epochs take a long time you can be sure things are still moving.
- **Update timeout error message in AutoMLExperiment** ([#6613](https://github.com/dotnet/machinelearning/pull/6613)) - Updated the error message so it is more clear what happened.
- **Add batchsize and arch to imageClassification SweepableTrainer** ([#6597](https://github.com/dotnet/machinelearning/pull/6597)) - Added batchsize and arch to the ImageClassification SweepableTrainer so those can now be trained on.
- **Update max_model when trial fails** ([#6596](https://github.com/dotnet/machinelearning/pull/6596))
- **Add default search space for standard trainers** ([#6576](https://github.com/dotnet/machinelearning/pull/6576)) - Added a default search space for all standard trainers so users have reasonable default values.
- **Adding more metrics to BinaryClassification Experiment** ([#6571](https://github.com/dotnet/machinelearning/pull/6571))
- **Add checkAlive in NasBertTrainer** ([#6546](https://github.com/dotnet/machinelearning/pull/6546)) - Now we check between batches if cancellation was requested and stop processing if so.
- **OneDAL - Fallback to default implementation** ([#6538](https://github.com/dotnet/machinelearning/pull/6538)) - If you specify you want to use OneDal but something happens that prevents you from using it, like it can't find the binaries/etc, it will auto default back to the normal implementation instead of crashing.
- **Add addKeyValueAnnotationsAsText flag in AutoML** ([#6535](https://github.com/dotnet/machinelearning/pull/6535))
- **Add continuous resource monitoring to AutoML.IMonitor** ([#6520](https://github.com/dotnet/machinelearning/pull/6520)) - Thanks @andrasfuchs!
- **Update WebClient to HttpClient implementations** ([#6476](https://github.com/dotnet/machinelearning/pull/6476)) - Update a usage of WebClient to HttpClient since WebClient is now deprecated. Thanks @rgesteve!
- **Set AutoML trial to unsuccess if trial loss is nan/inf** ([#6430](https://github.com/dotnet/machinelearning/pull/6430)) - Now trial will be marked as unsuccesssful if the loss is an invalid number.
- **Add diskConvert option in fast tree search space** ([#6316](https://github.com/dotnet/machinelearning/pull/6316))

## **Bug Fixes**
- **Fix DataFrame ToString** ([#6673](https://github.com/dotnet/machinelearning/pull/6673)) - Use correct alignment for columns to produce readable output when columns have longer names. Thanks @asmirnov82!
- **Fix DataFrame null math** ([#6661](https://github.com/dotnet/machinelearning/pull/6661)) - Fixes max in DataFrame columns when there are null values to match what Pandas does.
- **Clean up PrimitiveColumnContainer** ([#6656](https://github.com/dotnet/machinelearning/pull/6656)) - Cleaned up the code in PrimitiveColumnContainer so its more correct and easier to use.
- **Fix Apply in PrimitiveColumnContainer** ([#6642](https://github.com/dotnet/machinelearning/pull/6642)) - Fixes the Apply method so it no longer changes the source column. Thanks @janholo!
- **Fix datetime null error** ([#6627](https://github.com/dotnet/machinelearning/pull/6627)) - Fixes loading a null datetime from a database so it now returns correctly instead of throwing an error.
- **Fix AggregateTrainingStopManager is trying to cancel disposed tokens** ([#6612](https://github.com/dotnet/machinelearning/pull/6612)) - Will no longer try and cancel already disposed tokens.
- **Fix tostring bug for sweepable pipeline** ([#6610](https://github.com/dotnet/machinelearning/pull/6610))
- **Change Test to Validate in Dataset manager** ([#6599](https://github.com/dotnet/machinelearning/pull/6599))
- **Fixed System.OperationCanceledException when calling experimentResult.BestRun.Estimator.Fit** ([#6572](https://github.com/dotnet/machinelearning/pull/6572))
- **Fixed cancellation bug in SweepablePipelineRunner && Fixed object null exception in AutoML v1.0 regression API** ([#6560](https://github.com/dotnet/machinelearning/pull/6560))
- **Fixed one dal dispatching issues** ([#6547](https://github.com/dotnet/machinelearning/pull/6547)) - OneDal now dispatches correctly.
- **Fixed Multi-threaded access issue** ([#6537](https://github.com/dotnet/machinelearning/pull/6537)) - Fixed a multi-threaded access issue for variable length string arrays in ONNX models.
- **Fixed AutoML experiments in non declarative style not working** ([#6447](https://github.com/dotnet/machinelearning/pull/6447))

## **Build / Test updates**
- **Remove MSIL Check for TorchSharp** ([#6658](https://github.com/dotnet/machinelearning/pull/6658)) - Removes the MSIL check for TorchSharp while we figure out how we want to correctly handle this.
- **Change code coverage build pool** ([#6647](https://github.com/dotnet/machinelearning/pull/6647)) - Changed codecoverage build pool so the builds are faster and more stable.
- **Update AutoMLExperimentTests.cs to fix timeout error** ([#6638](https://github.com/dotnet/machinelearning/pull/6638))
- **Update FabricBot config** ([#6619](https://github.com/dotnet/machinelearning/pull/6619))
- **Libraries area pod updates March 2023** ([#6607](https://github.com/dotnet/machinelearning/pull/6607))
- **Update dependencies from dotnet/arcade** ([#6566 & #6518 & #6451 & #6439](https://github.com/dotnet/machinelearning/pull/6566))
- **Mac python fix** ([#6549](https://github.com/dotnet/machinelearning/pull/6549))
- **Moving onedal nuget download from onedal to native where its needed for building** ([#6527](https://github.com/dotnet/machinelearning/pull/6527))
- **New os image for official builds** ([#6467](https://github.com/dotnet/machinelearning/pull/6467))

## **Documentation Updates**
- **Add doc for CreateSweepableEstimator, Parameter and SearchSpace** ([#6611](https://github.com/dotnet/machinelearning/pull/6611))
- **Add AutoMLExperiment example doc** ([#6594](https://github.com/dotnet/machinelearning/pull/6594))
- **Fix minor doc typos** ([#6557](https://github.com/dotnet/machinelearning/pull/6557))
- **Fix minor roadmap nits** ([#6480](https://github.com/dotnet/machinelearning/pull/6480))
- **2023 roadmap outline** ([#6444](https://github.com/dotnet/machinelearning/pull/6444))
- **Fixed typo for calibrators** ([#6438](https://github.com/dotnet/machinelearning/pull/6438)) - Thanks @KKghub!

## **Breaking changes**
- None