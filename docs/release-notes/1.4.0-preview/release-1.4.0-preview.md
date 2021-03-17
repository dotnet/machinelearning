# [ML.NET](http://dot.net/ml) 1.4.0-preview

## **New Features**
- **Deep Neural Networks Training (0.16.0-preview)** ([#4151](https://github.com/dotnet/machinelearning/pull/4151))

  Improves the in-preview `ImageClassification` API further:
  - Increases DNN training speed by ~10x compared to the same API in 0.15.1 release.
  - Prevents repeated computations by caching featurized image values to disk from intermediate layers to train the final fully-connected layer.
  - Reduced and constant memory footprint.
  - Simplifies the API by not requiring the user to pre-process the image.
  - Introduces callback to provide metrics during training such as accuracy, cross-entropy.
  - Improved image classification sample.

  ```cs
        public static ImageClassificationEstimator ImageClassification(
            this ModelOperationsCatalog catalog,
            string featuresColumnName,
            string labelColumnName,
            string scoreColumnName = "Score",
            string predictedLabelColumnName = "PredictedLabel",
            Architecture arch = Architecture.InceptionV3,
            int epoch = 100,
            int batchSize = 10,
            float learningRate = 0.01f,
            ImageClassificationMetricsCallback metricsCallback = null,
            int statisticFrequency = 1,
            DnnFramework framework = DnnFramework.Tensorflow,
            string modelSavePath = null,
            string finalModelPrefix = "custom_retrained_model_based_on_",
            IDataView validationSet = null,
            bool testOnTrainSet = true,
            bool reuseTrainSetBottleneckCachedValues = false,
            bool reuseValidationSetBottleneckCachedValues = false,
            string trainSetBottleneckCachedValuesFilePath = "trainSetBottleneckFile.csv",
            string validationSetBottleneckCachedValuesFilePath = "validationSetBottleneckFile.csv"
            )

  ```

  [Design specification](https://github.com/dotnet/machinelearning/blob/cd591dd492833964b6829e8bb2411fb81665ac6d/docs/specs/DNN/dnn_api_spec.md)

  [Sample](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/ImageClassification/ResnetV2101TransferLearningTrainTestSplit.cs)

- **Database Loader (0.16.0-preview)** ([#4070](https://github.com/dotnet/machinelearning/pull/4070),[#4091](https://github.com/dotnet/machinelearning/pull/4091),[#4138](https://github.com/dotnet/machinelearning/pull/4138))

  Additional DatabaseLoader support:
  -  Support DBNull.
  -  Add `CreateDatabaseLoader<TInput>` to map columns from a .NET Type.
  -  Read multiple columns into a single vector

  [Design specification](https://github.com/dotnet/machinelearning/pull/3857)

  [Sample](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DatabaseLoader)

  ```cs
    string connectionString = "YOUR_RELATIONAL_DATABASE_CONNECTION_STRING";

    string commandText = "SELECT * from URLClicks";

    DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<UrlClick>();

    DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance,
                                                    connectionString,
                                                    commandText);
    IDataView dataView = loader.Load(dbSource);
  ```

- **Enhanced .NET Core 3.0 Support**

  -  Use C# hardware intrinsics detection to support AVX, SSE and software fallbacks
  -  Allows for faster training on AVX-supported machines
  -  Allows for scoring core ML .NET models on ARM processors. (Note: some components do not support ARM yet, ex. FastTree, LightGBM, OnnxTransformer)

## **Bug Fixes**
None.

## **Samples**
- DeepLearning Image Classification Training sample (DNN Transfer Learning) ([#633](https://github.com/dotnet/machinelearning-samples/pull/633))
- DatabaseLoader sample loading an IDataView from SQL Server localdb ([#611](https://github.com/dotnet/machinelearning-samples/pull/617))

## **Breaking Changes**
None

## **Enhancements**
None.

## **CLI and AutoML API**
  - AutoML codebase has moved from feature branch to master branch ([#3882](https://github.com/dotnet/machinelearning/pull/3882)).

## **Remarks**
None.





