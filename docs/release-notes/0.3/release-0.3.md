# ML.NET 0.3 Release Notes

Today we are releasing ML.NET 0.3. This release focuses on adding components
to ML.NET from the internal codebase (such as Factorization Machines,
LightGBM, Ensembles, and LightLDA), enabling export to the ONNX model format,
and bug fixes.

### Installation

ML.NET supports Windows, MacOS, and Linux. See [supported OS versions of .NET
Core
2.0](https://github.com/dotnet/core/blob/master/release-notes/2.0/2.0-supported-os.md)
for more details.

You can install ML.NET NuGet from the CLI using:
```
dotnet add package Microsoft.ML
```

From package manager:
```
Install-Package Microsoft.ML
```

### Release Notes

Below are some of the highlights from this release.

* Added Field-Aware Factorization Machines (FFM) as a learner for binary
  classification (#383)

    * FFM is useful for various large sparse datasets, especially in areas
      such as recommendations and click prediction. It has been used to win
      various click prediction competitions such as the [Criteo Display
      Advertising Challenge on
      Kaggle](https://www.kaggle.com/c/criteo-display-ad-challenge). You can
      learn more about the winning solution
      [here](https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf).
    * FFM is a streaming learner so it does not require the entire dataset to
      fit in memory.
    * You can learn more about FFM
      [here](http://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) and some of the
      speedup approaches that are used in ML.NET
      [here](https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf).

* Added [LightGBM](https://github.com/Microsoft/LightGBM) as a learner for
  binary classification, multiclass classification, and regression (#392)

    * LightGBM is a tree based gradient boosting machine. It is under the
      umbrella of the [DMTK](http://github.com/microsoft/dmtk) project at
      Microsoft.
    * The LightGBM repository shows various [comparison
      experiments](https://github.com/Microsoft/LightGBM/blob/6488f319f243f7ff679a8e388a33e758c5802303/docs/Experiments.rst#comparison-experiment)
      that show good accuracy and speed, so it is a great learner to try out.
      It has also been used in winning solutions in various [ML
      challenges](https://github.com/Microsoft/LightGBM/blob/a6e878e2fc6e7f545921cbe337cc511fbd1f500d/examples/README.md).
    * This addition wraps LightGBM and exposes it in ML.NET.
    * Note that LightGBM can also be used for ranking, but the ranking
      evaluator is not yet exposed in ML.NET.

* Added Ensemble learners for binary classification, multiclass
  classification, and regression (#379)

    * [Ensemble learners](https://en.wikipedia.org/wiki/Ensemble_learning)
      enable using multiple learners in one model. As an example, the Ensemble
      learner could train both `FastTree` and `AveragedPerceptron` and average
      their predictions to get the final prediction. 
    * Combining multiple models of similar statistical performance may lead to
      better performance than each model separately.

* Added LightLDA transform for topic modeling (#377)

    * LightLDA is an implementation of [Latent Dirichlet
      Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
      which infers topical structure from text data. 
    * The implementation of LightLDA in ML.NET is based on [this
      paper](https://arxiv.org/abs/1412.1576). There is a distributed
      implementation of LightLDA
      [here](https://github.com/Microsoft/lightlda).

* Added One-Versus-All (OVA) learner for multiclass classification (#363)

    * [OVA](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest)
      (sometimes known as One-Versus-Rest) is an approach to using binary
      classifiers in multiclass classification problems. 
    * While some binary classification learners in ML.NET natively support
      multiclass classification (e.g. Logistic Regression), there are others
      that do not (e.g. Averaged Perceptron). OVA enables using the latter
      group for multiclass classification as well.

* Enabled export of ML.NET models to the [ONNX](https://onnx.ai/) format
  (#248)

    * ONNX is a common format for representing deep learning models (also
      supporting certain other types of models) which enables developers to
      move models between different ML toolkits.
    * ONNX models can be used in [Windows
      ML](https://docs.microsoft.com/en-us/windows/uwp/machine-learning/overview)
      which enables evaluating models on Windows 10 devices and taking
      advantage of capabilities like hardware acceleration.
    * Currently, only a subset of ML.NET components can be used in a model
      that is converted to ONNX. 

Additional issues closed in this milestone can be found
[here](https://github.com/dotnet/machinelearning/milestone/2?closed=1).

### Acknowledgements

Shoutout to [pkulikov](https://github.com/pkulikov),
[veikkoeeva](https://github.com/veikkoeeva),
[ross-p-smith](https://github.com/ross-p-smith),
[jwood803](https://github.com/jwood803),
[Nepomuceno](https://github.com/Nepomuceno), and the ML.NET team for their
contributions as part of this release! 
