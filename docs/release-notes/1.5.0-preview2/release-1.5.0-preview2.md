# [ML.NET](http://dot.net/ml) 1.5.0-preview2

## **New Features (IN-PREVIEW, please provide feedback)**
- **TimeSeriesImputer** ([#4623](https://github.com/dotnet/machinelearning/pull/4623)) This data transformer can be used to impute missing rows in time series data.
- **LDSVM Trainer** ([#4060](https://github.com/dotnet/machinelearning/pull/4060)) The "Local Deep SVM" usess trees as its SVM kernel to create a non-linear binary trainer. A sample can be found [here](https://github.com/dotnet/machinelearning/blob/c819d77e9250c68883713d5f1cd79b8971a11faf/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/LdSvm.cs).
- **Onnxruntime updated to v1.2** This also includes support for GPU execution of onnx models
- **Export-to-ONNX for below components:**
    - SlotsDroppingTransformer ([#4562](https://github.com/dotnet/machinelearning/pull/4562))
    - ColumnSelectingTransformer ([#4590](https://github.com/dotnet/machinelearning/pull/4590))
    - VectorWhiteningTransformer ([#4577](https://github.com/dotnet/machinelearning/pull/4577))
    - NaiveBayesMulticlassTrainer ([#4636](https://github.com/dotnet/machinelearning/pull/4636))
  	- PlattCalibratorTransformer ([#4699](https://github.com/dotnet/machinelearning/pull/4699))
  	- TokenizingByCharactersTransformer ([#4805](https://github.com/dotnet/machinelearning/pull/4805))
  	- TextNormalizingTransformer ([#4781](https://github.com/dotnet/machinelearning/pull/4781))

  
## **Bug Fixes**
  - Fix issue in WaiterWaiter caused by race condition ([#4829](https://github.com/dotnet/machinelearning/pull/4829))
	- Onnx Export change to allow for running inference on multiple rows in OnnxRuntime ([#4783](https://github.com/dotnet/machinelearning/pull/4783))
  - Data splits to default to MLContext seed when not specified ([#4764](https://github.com/dotnet/machinelearning/pull/4764))
  - Add Seed property to MLContext and use as default for data splits ([#4775](https://github.com/dotnet/machinelearning/pull/4775))
- **Onnx bug fixes**
  - Updating onnxruntime version ([#4882](https://github.com/dotnet/machinelearning/pull/4882))
  - Calculate ReduceSum row by row in ONNX model from OneVsAllTrainer ([#4904](https://github.com/dotnet/machinelearning/pull/4904))
  - Several onnx export fixes related to KeyToValue and ValueToKey transformers ([#4900](https://github.com/dotnet/machinelearning/pull/4900), [#4866](https://github.com/dotnet/machinelearning/pull/4866), [#4841](https://github.com/dotnet/machinelearning/pull/4841), [#4889](https://github.com/dotnet/machinelearning/pull/4889), [#4878](https://github.com/dotnet/machinelearning/pull/4878), [#4797](https://github.com/dotnet/machinelearning/pull/4797))
  - Fixes to onnx export for text related transforms ([#4891](https://github.com/dotnet/machinelearning/pull/4891), [#4813](https://github.com/dotnet/machinelearning/pull/4813))
  - Fixed bugs in OptionalColumnTransform and ColumnSelecting ([#4887](https://github.com/dotnet/machinelearning/pull/4887), [#4815](https://github.com/dotnet/machinelearning/pull/4815))
  - Alternate solution for ColumnConcatenatingTransformer ([#4875](https://github.com/dotnet/machinelearning/pull/4875))
  - Added slot names support for OnnxTransformer ([#4857](https://github.com/dotnet/machinelearning/pull/4857))
  - Fixed output schema of OnnxTransformer ([#4849](https://github.com/dotnet/machinelearning/pull/4849))
  - Changed Binarizer node to be cast to the type of the predicted label … ([#4818](https://github.com/dotnet/machinelearning/pull/4818))
  - Fix for OneVersusAllTrainer ([#4698](https://github.com/dotnet/machinelearning/pull/4698)) 
  - Enable OnnxTransformer to accept KeyDataViewTypes as if they were UInt32 ([#4824](https://github.com/dotnet/machinelearning/pull/4824))
  - Fix off by 1 error with the cats_int64s attribute for the OneHotEncoder ONNX operator ([#4827](https://github.com/dotnet/machinelearning/pull/4827))
  - Changed Binarizer node to be cast to the type of the predicted label … ([#4818](https://github.com/dotnet/machinelearning/pull/4818))
  - Updated handling of missing values with LightGBM, and added ability to use (0) as missing value ([#4695](https://github.com/dotnet/machinelearning/pull/4695))
  - Double cast to float for some onnx estimators  ([#4745](https://github.com/dotnet/machinelearning/pull/4745))
  - Fix onnx output name for GcnTransform ([#4786](https://github.com/dotnet/machinelearning/pull/4786))
- Added support to run PFI on uncalibrated binary classification models ([#4587](https://github.com/dotnet/machinelearning/pull/4587))
- Fix bug in WordBagEstimator when training on empty data ([#4696](https://github.com/dotnet/machinelearning/pull/4696))
- Added Cancellation mechanism to Image Classification (through the experimental nuget) (fixes #4632) ([#4650](https://github.com/dotnet/machinelearning/pull/4650))
-	Changed F1 score to return 0 instead of NaN when Precision + Recall is 0 ([#4674](https://github.com/dotnet/machinelearning/pull/4674))
-	TextLoader, BinaryLoader and SvmLightLoader now check the existence of the input file before training ([#4665](https://github.com/dotnet/machinelearning/pull/4665))
- ImageLoadingTransformer now checks the existence of input folder before training ([#4691]( https://github.com/dotnet/machinelearning/pull/4691))
- Use random file name for AutoML experiment folder ([#4657](https://github.com/dotnet/machinelearning/pull/4657))
- Using invariance culture when converting to string  ([#4635](https://github.com/dotnet/machinelearning/pull/4635))
- Fix NullReferenceException when it comes to Recommendation in AutoML and CodeGenerator ([#4774](https://github.com/dotnet/machinelearning/pull/4774))

## **Enhancements**
- Added in support for System.DateTime type for the DateTimeTransformer ([#4661](https://github.com/dotnet/machinelearning/pull/4661))
- Additional changes to ExpressionTransformer ([#4614](https://github.com/dotnet/machinelearning/pull/4614))
- Optimize generic MethodInfo for Func<TResult> ([#4588](https://github.com/dotnet/machinelearning/pull/4588))
- Data splits to default to MLContext seed when not specified ([#4764](https://github.com/dotnet/machinelearning/pull/4764))
- Added in DateTime type support for TimeSeriesImputer ([#4812](https://github.com/dotnet/machinelearning/pull/4812))

## **Test updates**
- **Code analysis updates**
  - Update analyzer test library ([#4740](https://github.com/dotnet/machinelearning/pull/4740))
  - Enable the internal code analyzer for test projects ([#4731](https://github.com/dotnet/machinelearning/pull/4731))
  - Implement MSML_ExtendBaseTestClass (Test classes should be derived from BaseTestClass) ([#4746](https://github.com/dotnet/machinelearning/pull/4746))
  - Enable MSML_TypeParamName for the full solution ([#4762](https://github.com/dotnet/machinelearning/pull/4762))
  - Enable MSML_ParameterLocalVarName for the full solution ([#4833](https://github.com/dotnet/machinelearning/pull/4833))
  - Enable MSML_SingleVariableDeclaration for the full solution ([#4765](https://github.com/dotnet/machinelearning/pull/4765))
- **Better logging from tests**
  - Ensure tests capture the full log ([#4710](https://github.com/dotnet/machinelearning/pull/4710))
  - Fix failure to capture test failures ([#4716](https://github.com/dotnet/machinelearning/pull/4716))
  - Collect crash dump upload dump and pdb to artifact ([#4666](https://github.com/dotnet/machinelearning/pull/4666))
- Enable Conditional Numerical Reproducibility for tests ([#4569](https://github.com/dotnet/machinelearning/pull/4569))
- Changed all MLContext creation to include a fixed seed ([#4736](https://github.com/dotnet/machinelearning/pull/4736))
- Fix incorrect SynchronizationContext use in TestSweeper ([#4779](https://github.com/dotnet/machinelearning/pull/4779))

## **Documentation Updates**
- Update cookbook to latest API ([#4706](https://github.com/dotnet/machinelearning/pull/4706))
- Update documentation to stop mentioning interfaces that no longer exist ([#4673](https://github.com/dotnet/machinelearning/pull/4673))
- Roadmap update ([#4704](https://github.com/dotnet/machinelearning/pull/4704))
- Added release process documentation to README.md ([#4402](https://github.com/dotnet/machinelearning/pull/4402))
- Fix documentation of SvmLightLoader ([#4616](https://github.com/dotnet/machinelearning/pull/4616))
- Correct KMeans scoring function doc ([#4705](https://github.com/dotnet/machinelearning/pull/4705))
- Several typo fixes thanks to [@MaherJendoubi](https://github.com/MaherJendoubi) ([#4627](https://github.com/dotnet/machinelearning/pull/4627), [#4631](https://github.com/dotnet/machinelearning/pull/4631), [#4626](https://github.com/dotnet/machinelearning/pull/4626) [#4617](https://github.com/dotnet/machinelearning/pull/4617), [#4633](https://github.com/dotnet/machinelearning/pull/4633), [#4629](https://github.com/dotnet/machinelearning/pull/4629), [#4642](https://github.com/dotnet/machinelearning/pull/4642))
- Other typo fixes: ([#4628](https://github.com/dotnet/machinelearning/pull/4628), [#4685](https://github.com/dotnet/machinelearning/pull/4685), [#4885](https://github.com/dotnet/machinelearning/pull/4885))


## **Breaking Changes**
- None





