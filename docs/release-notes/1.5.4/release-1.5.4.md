# [ML.NET](http://dot.net/ml) 1.5.4

## **New Features**
- **New API for exporting models to Onnx**. ([#5544](https://github.com/dotnet/machinelearning/pull/5544)). A new API has been added to Onnx converter to specify the output columns you care about. This will export a smaller and more performant model in many cases.

## **Enhancements**
- Perf improvement for TopK Accuracy and return all topK in Classification Evaluator ([#5395](https://github.com/dotnet/machinelearning/pull/5395)) (**Thank you @jasallen**)
- Update OnnxRuntime to 1.6 ([#5529](https://github.com/dotnet/machinelearning/pull/5529))
- Updated tensorflow.net to 0.20.0 ([#5404](https://github.com/dotnet/machinelearning/pull/5404))
- Added in DcgTruncationLevel to AutoML api and increased default level to 10 ([#5433](https://github.com/dotnet/machinelearning/pull/5433))

## **Bug Fixes**
- **AutoML.NET specific fixes**.
  - Fixed AutoFitMaxExperimentTimeTest ([#5506](https://github.com/dotnet/machinelearning/pull/5506))
  - Fixed code generator tests failure ([#5520](https://github.com/dotnet/machinelearning/pull/5520))
  - Use Timer and ctx.CancelExecution() to fix AutoML max-time experiment bug ([#5445](https://github.com/dotnet/machinelearning/pull/5445))
  - Handled exception during GetNextPipeline for AutoML ([#5455](https://github.com/dotnet/machinelearning/pull/5455))
  - Fixed internationalization bug([#5162](https://github.com/dotnet/machinelearning/pull/5163)) in AutoML parameter sweeping caused by culture dependent float parsing. ([#5163](https://github.com/dotnet/machinelearning/pull/5163))
  - Fixed MaxModels exit criteria for AutoML unit test ([#5471](https://github.com/dotnet/machinelearning/pull/5471))
  - Fixed AutoML CrossValSummaryRunner for TopKAccuracyForAllK ([#5548](https://github.com/dotnet/machinelearning/pull/5548))
- Fixed bug in Tensorflow Transforer with handling primitive types ([#5547](https://github.com/dotnet/machinelearning/pull/5547))
- Fixed MLNet.CLI build error ([#5546](https://github.com/dotnet/machinelearning/pull/5546))
- Fixed memory leaks from OnnxTransformer ([#5518](https://github.com/dotnet/machinelearning/pull/5518))
- Fixed memory leak in object pool ([#5521](https://github.com/dotnet/machinelearning/pull/5521))
- Fixed Onnx Export for ProduceWordBags ([#5435](https://github.com/dotnet/machinelearning/pull/5435))
- Upgraded boundary calculation and expected value calculation in SrCnnEntireAnomalyDetector ([#5436](https://github.com/dotnet/machinelearning/pull/5436))
- Fixed SR anomaly score calculation at beginning ([#5502](https://github.com/dotnet/machinelearning/pull/5502))
- Improved error message in ColumnConcatenatingEstimator ([#5444](https://github.com/dotnet/machinelearning/pull/5444))
- Fixed issue 5020, allow ML.NET to load tf model with primitive input and output column ([#5468](https://github.com/dotnet/machinelearning/pull/5468))
- Fixed issue 4322, enable lda summary output ([#5260](https://github.com/dotnet/machinelearning/pull/5260))
- Fixed perf regression in ShuffleRows ([#5417](https://github.com/dotnet/machinelearning/pull/5417))
- Change the _maxCalibrationExamples default on CalibratorUtils ([#5415](https://github.com/dotnet/machinelearning/pull/5415))


## **Build / Test updates**
- Migrated to [Arcade](https://github.com/dotnet/arcade/) build system that is used my multiple dotnet projects. This will give increased build/CI efficiencies going forward. Updated build instructions can be found in the docs/building folder
- Fixed MacOS builds ([#5467](https://github.com/dotnet/machinelearning/pull/5467) and [#5457](https://github.com/dotnet/machinelearning/pull/5457))

## **Documentation Updates**
- Fixed Spelling on stopwords ([#5524](https://github.com/dotnet/machinelearning/pull/5524))(**Thank you @LeoGaunt**)
- Changed LoadRawImages Sample ([#5460](https://github.com/dotnet/machinelearning/pull/5460))


## **Breaking Changes**
- None
