# [ML.NET](http://dot.net/ml) 1.2.0
## **General Availability**
- **Microsoft.ML.TimeSeries**
    - Anomaly detection algorithms (Spike and Change Point):
      - Independent and identically distributed.
      - Singular spectrum analysis.
      - Spectral residual from Azure Anomaly Detector/Kensho team.
    - Forecasting models:
      - Singular spectrum analysis.
    - Prediction Engine for online learning
      - Enables updating time series model with new observations at scoring so that the user does not have to re-train the time series with old data each time.

     [Samples](https://github.com/dotnet/machinelearning/tree/main/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TimeSeries)

- **Microsoft.ML.OnnxTransformer**
   Enables scoring of ONNX models in the learning pipeline. Uses ONNX Runtime v0.4.

   [Sample](https://github.com/dotnet/machinelearning/blob/main/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/ApplyOnnxModel.cs)

- **Microsoft.ML.TensorFlow**
   Enables scoring of TensorFlow models in the learning pipeline. Uses TensorFlow v1.13. Very useful for image and text classification. Users can featurize images or text using DNN models and feed the result into a classical machine learning model like a decision tree or logistic regression trainer.

   [Samples](https://github.com/dotnet/machinelearning/tree/main/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlow)

## **New Features**
- **Tree-based featurization** ([#3812](https://github.com/dotnet/machinelearning/pull/3812))

    Generating features using tree structure has been a popular technique in data mining. Useful for capturing feature interactions when creating a stacked model, dimensionality reduction, or featurizing towards an alternative label. [ML.NET](dot.net/ml)'s tree featurization trains a tree-based model and then maps input feature vector to several non-linear feature vectors. Those generated feature vectors are:
  - The leaves it falls into. It's a binary vector with ones happens at the indexes of reached leaves,
  - The paths that the input vector passes before hitting the leaves, and
  - The reached leaves values.

  Here are two references.
  - [p. 9](https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf) (a Kaggle solution adopted by FB below).
  - [Section 3](http://www.quinonero.net/Publications/predicting-clicks-facebook.pdf). (Facebook)
  - [Section of Entity-level personalization with GLMix](https://engineering.linkedin.com/blog/2019/04/ai-behind-linkedin-recruiter-search-and-recommendation-systems). (LinkedIn)

  [Samples](https://github.com/dotnet/machinelearning/tree/main/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/TreeFeaturization)

- **Microsoft.Extensions.ML integration package.** ([#3827](https://github.com/dotnet/machinelearning/pull/3827))

    This package makes it easier to use [ML.NET](dot.net/ml) with app models that support Microsoft.Extensions - i.e. ASP.NET and Azure Functions.

    Specifically it contains functionality for:
    - Dependency Injection
    - Pooling PredictionEngines
    - Reloading models when the file or URI has changed
    - Hooking ML.NET logging to Microsoft.Extensions.Logging

## **Bug Fixes**
### Serious
- **Time series Sequential Transform needs to have a binding mechanism:** This bug made it impossible to use time series in NimbusML. ([#3875](https://github.com/dotnet/machinelearning/pull/3875))

- **Build errors resulting from upgrading to VS2019 compilers:** The default CMAKE_C_FLAG for debug configuration sets /ZI to generate a PDB capable of edit and continue. In the new compilers, this is incompatible with /guard:cf which we set for security reasons. ([#3894](https://github.com/dotnet/machinelearning/pull/3894))

- **LightGBM Evaluation metric parameters:** In LightGbm EvaluateMetricType where if a user specified EvaluateMetricType.Default, the metric would not get added to the options Dictionary, and LightGbmWrappedTraining would throw because of that. ([#3815](https://github.com/dotnet/machinelearning/pull/3815))

- **Change default EvaluationMetric for LightGbm:** In [ML.NET](dot.net/ml), the default EvaluationMetric for LightGbm is set to EvaluateMetricType.Error for multiclass, EvaluationMetricType.LogLoss for binary etc. This leads to inconsistent behavior from the user's perspective. ([#3859](https://github.com/dotnet/machinelearning/pull/3859))
### Other
- CustomGains should allow multiple values in argument attribute. ([#3854](https://github.com/dotnet/machinelearning/pull/3854))

## **Breaking Changes**
None

## **Enhancements**
- Fixes the Hardcoded Sigmoid value from -0.5 to the value specified during training. ([#3850](https://github.com/dotnet/machinelearning/pull/3850))
- Fix TextLoader constructor and add exception message. ([#3788](https://github.com/dotnet/machinelearning/pull/3788))
- Introduce the `FixZero` argument to the LogMeanVariance normalizer. ([#3916](https://github.com/dotnet/machinelearning/pull/3916))
- Ensembles trainer now work with ITrainerEstimators instead of ITrainers. ([#3796](https://github.com/dotnet/machinelearning/pull/3796))
- LightGBM Unbalanced Data Argument. ([#3925](https://github.com/dotnet/machinelearning/pull/3925))
- Tree based trainers implement ICanGetSummaryAsIDataView. ([#3892](https://github.com/dotnet/machinelearning/pull/3892))

- **CLI and AutoML API**
  - Internationalization fixes to generate proper [ML.NET](dot.net/ml) C# code. ([#3725](https://github.com/dotnet/machinelearning/pull/3725))
  - Automatic Cross Validation for small datasets, and CV stability fixes. ([#3794](https://github.com/dotnet/machinelearning/pull/3794))
  - Code cleanup to match .NET style. ([#3823](https://github.com/dotnet/machinelearning/pull/3823))


## **Documentation and Samples**
- Samples for applying ONNX model to in-memory images. ([#3851](https://github.com/dotnet/machinelearning/pull/3851))
- Reformatted all ~200 samples to 85 character width so the horizontal scrollbar does not appear on docs webpage. ([#3930](https://github.com/dotnet/machinelearning/pull/3930), [3941](https://github.com/dotnet/machinelearning/pull/3941), [3949](https://github.com/dotnet/machinelearning/pull/3949), [3950](https://github.com/dotnet/machinelearning/pull/3950), [3947](https://github.com/dotnet/machinelearning/pull/3947), [3943](https://github.com/dotnet/machinelearning/pull/3943), [3942](https://github.com/dotnet/machinelearning/pull/3942), [3946](https://github.com/dotnet/machinelearning/pull/3946), [3948](https://github.com/dotnet/machinelearning/pull/3948))

## **Remarks**
- Roughly 200 Github issues were closed, the count decreased from **~550 to 351**. Most of the issues got resolved due to the release of stable API and availability of samples.