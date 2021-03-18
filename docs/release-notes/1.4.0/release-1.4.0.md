# [ML.NET](http://dot.net/ml) 1.4.0

## **New Features**
- **General Availability of [Image Classification API](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.visioncatalog.imageclassification?view=ml-dotnet#Microsoft_ML_VisionCatalog_ImageClassification_Microsoft_ML_MulticlassClassificationCatalog_MulticlassClassificationTrainers_System_String_System_String_System_String_System_String_Microsoft_ML_IDataView_)**
  Introduces [`Microsoft.ML.Vision`](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.vision?view=ml-dotnet) package that enables image classification by leveraging an existing pre-trained deep neural network model. Here the API trains the last classification layer using TensorFlow by using its C# bindings from TensorFlow .NET. This is a high level API that is simple yet powerful. Below are some of the key features:
  - `GPU training`: Supported on Windows and Linux, more information [here](https://github.com/dotnet/machinelearning/blob/main/docs/api-reference/tensorflow-usage.md).
  - `Early stopping`: Saves time by stopping training automatically when model has been stabelized.
  - `Learning rate scheduler`: Learning rate is an integral and potentially difficult part of deep learning. By providing learning rate schedulers, we give users a way to optimize the learning rate with high initial values which can decay over time. High initial learning rate helps to introduce randomness into the system, allowing the Loss function to better find the global minima. While the decayed learning rate helps to stabilize the loss over time. We have implemented [Exponential Decay Learning rate scheduler](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/exponential_decay) and [Polynomial Decay Learning rate scheduler](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay).
  - `Pre-trained DNN Architectures`: The supported DNN architectures used internally for `transfer learning` are below:
    - Inception V3.
    - ResNet V2 101.
    - ResNet V2 50.
    - MobileNet V2.

  #### Example code:

  ```cs
  var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(
                  featureColumnName: "Image", labelColumnName: "Label");

  ITransformer trainedModel = pipeline.Fit(trainDataView);

  ```

  #### Samples

  [Defaults](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/ImageClassification/ImageClassificationDefault.cs)

  [Learning rate scheduling](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/ImageClassification/LearningRateSchedulingCifarResnetTransferLearning.cs)

  [Early stopping](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/ImageClassification/ResnetV2101TransferLearningEarlyStopping.cs)

  [ResNet V2 101 train-test split](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/ImageClassification/ResnetV2101TransferLearningTrainTestSplit.cs)

  [End-to-End](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_Training)

- **General Availability of [Database Loader](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.databaseloadercatalog?view=ml-dotnet)**
  The database loader enables to load data from databases into the `IDataView` and therefore enables model training directly against relational databases. This loader supports any relational database provider supported by System.Data in .NET Core or .NET Framework, meaning that you can use any RDBMS such as SQL Server, Azure SQL Database, Oracle, SQLite, PostgreSQL, MySQL, Progress, etc.

  It is important to highlight that in the same way as when training from files, when training with a database ML .NET also supports data streaming, meaning that the whole database doesn’t need to fit into memory, it’ll be reading from the database as it needs so you can handle very large databases (i.e. 50GB, 100GB or larger).

  #### Example code:
  ```cs
  //Lines of code for loading data from a database into an IDataView for a later model training
  //...
  string connectionString = @"Data Source=YOUR_SERVER;Initial Catalog= YOUR_DATABASE;Integrated Security=True";

  string commandText = "SELECT * from SentimentDataset";

  DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader();
  DbProviderFactory providerFactory = DbProviderFactories.GetFactory("System.Data.SqlClient");
  DatabaseSource dbSource = new DatabaseSource(providerFactory, connectionString, commandText);

  IDataView trainingDataView = loader.Load(dbSource);

  // ML.NET model training code using the training IDataView
  //...

  public class SentimentData
  {
      public string FeedbackText;
      public string Label;
  }
  ```

  [Design specification](https://github.com/dotnet/machinelearning/pull/3857)

  [Sample](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DatabaseLoader)

  [How to doc](https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/load-data-ml-net#load-data-from-a-relational-database)

- **General Availability of PredictionEnginePool for scalable deployment**
  When deploying an ML model into multi-threaded and scalable .NET Core web applications and services (such as ASP .NET Core web apps, WebAPIs or an Azure Function) it is recommended to use the PredictionEnginePool instead of directly creating the PredictionEngine object on every request due to performance and scalability reasons. For further background information on why the PredictionEnginePool is recommended, read [this](https://devblogs.microsoft.com/cesardelatorre/how-to-optimize-and-run-ml-net-models-on-scalable-asp-net-core-webapis-or-web-apps/) blog post.

  [Sample](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/end-to-end-apps/ScalableMLModelOnWebAPI-IntegrationPkg)

- **General Availability of Enhanced for .NET Core 3.0**
  This means ML .NET can take advantage of the new features when running in a .NET Core 3.0 application. The first new feature we are using is the new hardware intrinsics feature, which allows .NET code to accelerate math operations by using processor specific instructions.

## **Bug Fixes**
- Adds reasonable exception when user tries to use `OnnxSequenceType` attribute without specifing sequence type. ([#4272](https://github.com/dotnet/machinelearning/pull/4272))
- Image Classification API: Fix processing incomplete batch(<batchSize), images processed per epoch , enable EarlyStopping without Validation Set. ([#4289](https://github.com/dotnet/machinelearning/pull/4289))
- Exception is thrown if NDCG > 10 is used with LightGbm for evaluating ranking. ([##4081](https://github.com/dotnet/machinelearning/pull/4081))
- DatabaseLoader error when using attributes (i.e ColumnName). ([#4308](https://github.com/dotnet/machinelearning/pull/4308))
- Recommendation experiment got SMAC local search exception during training. ([#4358](https://github.com/dotnet/machinelearning/pull/4358))
- TensorFlow exception triggered: input ended unexpectedly in the middle of a field. ([#4314](https://github.com/dotnet/machinelearning/pull/4314))
- `PredictionEngine` breaks after saving/loading a Model. ([#4321](https://github.com/dotnet/machinelearning/pull/4321))
- Data file locked even after TextLoader goes out of context. ([#4404](https://github.com/dotnet/machinelearning/pull/4404))
- ImageClassification API should save cache files/meta files in user temp directory or user provided workspace path. ([#4410](https://github.com/dotnet/machinelearning/pull/4410))

## **Breaking Changes**
None

## **Enhancements**
- Publish latest nuget to [public feed](https://dev.azure.com/dnceng/public/_packaging?_a=feed&feed=MachineLearning) from master branch when commits are made. ([#4406](https://github.com/dotnet/machinelearning/pull/4406))
- Defaults for ImageClassification API. ([#4415](https://github.com/dotnet/machinelearning/pull/4415))

## **CLI and AutoML API**
  - Recommendation Task. ([#4246](https://github.com/dotnet/machinelearning/pull/4246), [4391](https://github.com/dotnet/machinelearning/pull/4391))
  - Image Classification Task. ([#4395](https://github.com/dotnet/machinelearning/pull/4395))
  - Move AutoML CodeGen to master from feature branch. ([#4365](https://github.com/dotnet/machinelearning/pull/4365))

## **Remarks**
- None.





