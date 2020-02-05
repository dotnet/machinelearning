# [ML.NET](http://dot.net/ml) 1.5.0-preview2

## **New Features (IN-PREVIEW, please provide feedback)**
- **TimeSeriesImputer** ([#4623](https://github.com/dotnet/machinelearning/pull/4623))
- **LDSVM Trainer** ([#4060](https://github.com/dotnet/machinelearning/pull/4060))
- **Export-to-ONNX for below components:**
    - SlotsDroppingTransformer ([#4562](https://github.com/dotnet/machinelearning/pull/4562))
    - ColumnSelectingTransformer ([#4590](https://github.com/dotnet/machinelearning/pull/4590))
    - VectorWhiteningTransformer ([#4577](https://github.com/dotnet/machinelearning/pull/4577))
    - NaiveBayesMulticlassTrainer ([#4636](https://github.com/dotnet/machinelearning/pull/4636))
  	- PlattCalibratorTransformer ([#4699](https://github.com/dotnet/machinelearning/pull/4699))

  
## **Bug Fixes**
- Fix for OneVersusAllTrainer([#4698](https://github.com/dotnet/machinelearning/pull/4698))
- Added support to run PFI on uncalibrated binary classification models ([#4587](https://github.com/dotnet/machinelearning/pull/4587))
- Fix bug in WordBagEstimator when training on empty data ([#4696](https://github.com/dotnet/machinelearning/pull/4696))
- Added Cancellation mechanism to Image Classification (through the experimental nuget) (fixes #4632) ([#4650](https://github.com/dotnet/machinelearning/pull/4650))
-	Changed F1 score to return 0 instead of NaN when Precision + Recall is 0 ([#4674](https://github.com/dotnet/machinelearning/pull/4674))
- TextLoader, BinaryLoader and SvmLightLoader now check the existence of the input file before training ([#4665](https://github.com/dotnet/machinelearning/pull/4665))
- Changed onnx export to append a .onnx string to column names to support better mapping to ML.NET output column names ([#4734](https://github.com/dotnet/machinelearning/pull/4734))
-	TextLoader, BinaryLoader and SvmLightLoader now check the existence of the input file before training ([#4665](https://github.com/dotnet/machinelearning/pull/4665))
- ImageClassificationTrainer now checks the existence of input folder before training ([#4691]( https://github.com/dotnet/machinelearning/pull/4691))
- Sweep Trimming of Whitespace in AutoML ([#3918](https://github.com/dotnet/machinelearning/pull/3918))
- Use random file name for AutoML experiment folder ([#4657](https://github.com/dotnet/machinelearning/pull/4657))


## **Enhancements**
- Added in support for System.DateTime type for the DateTimeTransformer ([#4661](https://github.com/dotnet/machinelearning/pull/4661))
- Additional changes to ExpressionTransformer ([#4614](https://github.com/dotnet/machinelearning/pull/4614))

## **Test updates**
- **Code analysis updates**
  - Update analyzer test library ([#4740](https://github.com/dotnet/machinelearning/pull/4740))
  - Enable the internal code analyzer for test projects ([#4731](https://github.com/dotnet/machinelearning/pull/4731))
  - Implement MSML_ExtendBaseTestClass (Test classes should be derived from BaseTestClass) ([#4746](https://github.com/dotnet/machinelearning/pull/4746))
  - Enable MSML_TypeParamName for the full solution ([#4762](https://github.com/dotnet/machinelearning/pull/4762))
- **Better logging from tests**
  - Ensure tests capture the full log ([#4710](https://github.com/dotnet/machinelearning/pull/4710))
  - Fix failure to capture test failures ([#4716](https://github.com/dotnet/machinelearning/pull/4716))
  - Collect crash dump upload dump and pdb to artifact ([#4666](https://github.com/dotnet/machinelearning/pull/4666))
- Enable Conditional Numerical Reproducibility for tests ([#4569](https://github.com/dotnet/machinelearning/pull/4569))
- Changed all MLContext creation to include a fixed seed ([#4736](https://github.com/dotnet/machinelearning/pull/4736))
- Fix incorrect SynchronizationContext use in TestSweeper ([#4779](https://github.com/dotnet/machinelearning/pull/4779))

## **Documentation Updates**
- Update documentation to stop mentioning interfaces that no longer exist ([#4673](https://github.com/dotnet/machinelearning/pull/4673))
- Roadmap update ([#4704](https://github.com/dotnet/machinelearning/pull/4704))
- Added release process documentation to README.md ([#4402](https://github.com/dotnet/machinelearning/pull/4402))
- Fix documentation of SvmLightLoader ([#4616](https://github.com/dotnet/machinelearning/pull/4616))
- Correct KMeans scoring function doc ([#4705](https://github.com/dotnet/machinelearning/pull/4705))
- Several typo fixes thanks to [@MaherJendoubi](https://github.com/MaherJendoubi) ([#4627](https://github.com/dotnet/machinelearning/pull/4627), [#4631](https://github.com/dotnet/machinelearning/pull/4631), [#4626](https://github.com/dotnet/machinelearning/pull/4626) [#4617](https://github.com/dotnet/machinelearning/pull/4617), [#4633](https://github.com/dotnet/machinelearning/pull/4633), [#4629](https://github.com/dotnet/machinelearning/pull/4629), [#4642](https://github.com/dotnet/machinelearning/pull/4642))
- Other typo fixes:([#4628](https://github.com/dotnet/machinelearning/pull/4628), [#4685](https://github.com/dotnet/machinelearning/pull/4685))

## **CLI and AutoML API**
  - CodeGen For AzureAttach ([#4498](https://github.com/dotnet/machinelearning/pull/4498))

## **Breaking Changes**
- None





