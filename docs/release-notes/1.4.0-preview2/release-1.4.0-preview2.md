# [ML.NET](http://dot.net/ml) 1.4.0-preview2

## **New Features**
- **Deep Neural Networks Training (0.16.0-preview2)**

  Improves the in-preview `ImageClassification` API further:
  - Early stopping feature stops the training when optimal accuracy is reached ([#4237](https://github.com/dotnet/machinelearning/pull/4237))
  - Enables inferencing on in-memory images ([#4242](https://github.com/dotnet/machinelearning/pull/4242))
  - `PredictedLabel` output column now contains actual class labels instead of `uint32` class index values ([#4228](https://github.com/dotnet/machinelearning/pull/4228))
  - GPU support on Windows and Linux ([#4270](https://github.com/dotnet/machinelearning/pull/4270), [#4277](https://github.com/dotnet/machinelearning/pull/4277))

  [In-memory image inferencing sample](https://github.com/dotnet/machinelearning/blob/master/docs/samples/Microsoft.ML.Samples/Dynamic/ImageClassification/ResnetV2101TransferLearningTrainTestSplit.cs)  
  [Early stopping sample](https://github.com/dotnet/machinelearning/blob/master/docs/samples/Microsoft.ML.Samples/Dynamic/ImageClassification/ResnetV2101TransferLearningEarlyStopping.cs)  
  [GPU samples](https://github.com/dotnet/machinelearning/tree/master/docs/samples/Microsoft.ML.Samples.GPU)  

- **Database Loader (0.16.0-preview2)** ([#4070](https://github.com/dotnet/machinelearning/pull/4070),[#4091](https://github.com/dotnet/machinelearning/pull/4091),[#4138](https://github.com/dotnet/machinelearning/pull/4138))  

  Additional DatabaseLoader support:
  -  Support DBNull.
  -  Add `CreateDatabaseLoader<TInput>` to map columns from a .NET Type.
  -  Read multiple columns into a single vector

  [Design specification](https://github.com/dotnet/machinelearning/pull/3857) 
  
  [Sample](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/DatabaseLoader)

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





