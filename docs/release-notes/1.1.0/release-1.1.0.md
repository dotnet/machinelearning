# [ML.NET](http://dot.net/ml) 1.1.0 
## **New Features**
- **Image type support in IDataView**  
   [PR#3263](https://github.com/dotnet/machinelearning/pull/3263) added support
  for in-memory image as a type in IDataView. Previously to consume image it was
  impossible to directly use it in IDataView rather a user would have to specify
  the file path as a string in IDataView and then the image would be loaded use a
  transform. This resulted in the closure of the following github issues:  [3162](https://github.com/dotnet/machinelearning/issues/3162), [3723](https://github.com/dotnet/machinelearning/issues/3723), [3369](https://github.com/dotnet/machinelearning/issues/3369), [3274](https://github.com/dotnet/machinelearning/issues/3274), [445](https://github.com/dotnet/machinelearning/issues/445), [3460](https://github.com/dotnet/machinelearning/issues/3460), [2121](https://github.com/dotnet/machinelearning/issues/2121), [2495](https://github.com/dotnet/machinelearning/issues/2495), [3784](https://github.com/dotnet/machinelearning/issues/3784) and was a much requested feature by the users.  

- **Super-Resolution based Anomaly Detector**  
   [PR#3693](https://github.com/dotnet/machinelearning/pull/3693) adds a new anomaly detection algorithm to the time series nuget. This algorithm is based on Super-Resolution using Deep Convolutional Networks and also got accepted in KDD'2019 conference as oral presentation. One of the advantages of this algorithm is that it does not require any prior training and based on benchmarks using grid parameter search to find upper bounds it out performs the IID and SSA based anomaly detection algorithms in accuracy. This contribution comes from the [Azure Anomaly Detector](https://azure.microsoft.com/en-us/services/cognitive-services/anomaly-detector/) team.

    Algo | Precision | Recall | F1 | #TruePositive | #Positives | #Anomalies | Fine tuned   parameters
    -- | -- | -- | -- | -- | -- | -- | --
    SSA (requires training) | 0.582 | 0.585 | 0.583 | 2290 | 3936 | 3915 | Confidence=99,   PValueHistoryLength=32, Season=11, and use half the data of each series to do   the training.
    IID | 0.668 | 0.491 | 0.566 | 1924 | 2579 | 3915 | Confidence=99,   PValueHistoryLength=56
    SR | 0.601 | 0.670 | 0.634 | 2625 | 4370 | 3915 | WindowSize=64,   BackAddWindowSize=5, LookaheadWindowSize=5, AveragingWindowSize=3,   JudgementWindowSize=64, Threshold=0.45

- **Time Series Forecasting**  
   [PR#1900](https://github.com/dotnet/machinelearning/pull/1900) introduces a framework for time series forecasting models and exposes an API for Singular Spectrum Analysis(SSA) based forecasting model. This framework allows to forecast w/o confidence intervals, update model with new observations and save the model to persistent storage. This closes [issue#929](https://github.com/dotnet/machinelearning/issues/929) and was a much requested feature by the github community. With this change time series nuget is feature complete for RTM.

## **Bug Fixes**
### Serious
- **Math Kernel Library fails to load with latest libomp:** Fixed by [PR#3721](https://github.com/dotnet/machinelearning/pull/3721) this bug made it impossible for anyone to check code into master branch because it was causing build failures.

- **Transform Wrapper fails at deserialization:** Fixed by [PR#3700](https://github.com/dotnet/machinelearning/pull/3700) this bug affected first party(1P) customer. A model trained models using [NimbusML](https://github.com/microsoft/NimbusML)(Python bindings for [ML.NET](http://dot.net/ml)) and then loaded for scoring/inferencing using ML.NET will hit this bug. 

- **Index out of bounds exception in KeyToVector transformer:** Fixed by [PR#3763](https://github.com/dotnet/machinelearning/pull/3763) this bug closes following github issues: [3757](https://github.com/dotnet/machinelearning/issues/3757),[1751](https://github.com/dotnet/machinelearning/issues/1751),[2678](https://github.com/dotnet/machinelearning/issues/2678). It affected first party customer and also github users. 

### Other
- Download images only when not present on disk and print warning messages when converting unsupported pixel format by [PR#3625](https://github.com/dotnet/machinelearning/pull/3625)
- [ML.NET](http://dot.net/ml) source code does not build in VS2019 by [PR#3742](https://github.com/dotnet/machinelearning/pull/3742)
- Fix SoftMax precision by utilizing double in the internal calculations by [PR#3676](https://github.com/dotnet/machinelearning/pull/3676)
- Fix to the official build due to API Compat tool change by [PR#3667](https://github.com/dotnet/machinelearning/pull/3667)

## **Breaking Changes**
None

## **Enhancements**
- API Compat tool by [PR#3623](https://github.com/dotnet/machinelearning/pull/3623) ensures future changes to ML.NET will not break the stable API released in 1.0.0.
- Upgrade the TensorFlow version from 1.12.0 to 1.13.1 by [PR#3758](https://github.com/dotnet/machinelearning/pull/3758)

## **Documentations and Samples**
- L1-norm and L2-norm regularization documentation by [PR#3586](https://github.com/dotnet/machinelearning/pull/3586)
- Sample for data save and load from text and binary files by [PR#3745](https://github.com/dotnet/machinelearning/pull/3745)
- Sample for LoadFromEnumerable with a SchemaDefinition by [PR#3696](https://github.com/dotnet/machinelearning/pull/3696)
- Sample for LogLossPerClass metric for multiclass trainers by [PR#3724](https://github.com/dotnet/machinelearning/pull/3724)
- Sample for WithOnFitDelegate by [PR#3738](https://github.com/dotnet/machinelearning/pull/3738)