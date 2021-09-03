# The ML.NET Roadmap

The goal of ML.NET is to democratize machine learning for .NET developers. This document outlines the current roadmap for the ML.NET framework and APIs.

To see the plans for ML.NET tooling, check out the [Model Builder repo](https://github.com/dotnet/machinelearning-modelbuilder/issues/1707).

## Feedback and contributions

ML.NET is a community effort and we welcome community feedback on our plans. The best way to give feedback is to [open an issue](https://github.com/dotnet/machinelearning/issues/new/choose) in this repo.

We also invite contributions. The [first good issue](https://github.com/dotnet/machinelearning/labels/good%20first%20issue) and [up-for-grabs issues](https://github.com/dotnet/machinelearning/issues?q=is%3Aopen+is%3Aissue+label%3Aup-for-grabs) on GitHub are a good place to start. You can also help work on any of the features we've listed below or work on features that you want to add to the framework.

## Goals through June 2022

The following sections outline the major areas and features we plan to work on in the next year.

Note, that this is an aspirational list of what we hope to get to. Many of the items on this list will require more investigations and design, which can result in changes in our plans. We may have to cut things as we go, or we may be able to add more things.

As we prioritize, cost, and continue planning, we will try to keep the Roadmap up to date to reflect our progress and learnings.

### Keep docs, samples, and repo up to date

We heard your feedback loud and clear that our outdated docs and samples were a top pain point when learning and using ML.NET.

We have invested more resources into content development to make sure our Docs stay relevant and that we add documentation for new features faster as well as add more relevant samples.

You can file issues and make suggestions for ML.NET documentation in the [dotnet/docs repo](https://github.com/dotnet/docs) and for ML.NET samples in the [dotnet/machinelearning-samples](https://github.com/dotnet/machinelearning-samples) repo.

We are also taking steps to organize the [dotnet/machinelearning](https://github.com/dotnet/machinelearning) repo and updating our triage processes so that we can address your issues and feedback faster. Issues will be linked to version releases in the [Projects](https://github.com/dotnet/machinelearning/projects) section of the repo so you can see what we're actively working on and when we plan to release.

### Get on the .NET release schedule

ML.NET is .NET, and to make it feel more a part of .NET, we've decided to align with the .NET release schedule.

This means that we will ship our next version of ML.NET (v1.7.0) with .NET 6.0 in November 2021.

While we'll have major releases of ML.NET once a year with the major .NET releases, we will maintain release branches to optionally service ML.NET with bug fixes and/or minor features on the same cadence as .NET servicing.

### Deep learning

This past year we've been working on our plan for deep learning in .NET, and now we are ready to execute that plan to expand ML.NET's deep learning support.

As part of this plan, we will:

1. Make it easier to consume ONNX models in ML.NET using the ONNX Runtime (RT)
2. Fully support and productionize [TorchSharp](https://github.com/xamarin/TorchSharp) for building neural networks in .NET
3. Build a bridge between TorchSharp and ML.NET

Read more about the deep learning plan and leave your feedback in this [tracking issue](https://github.com/dotnet/machinelearning/issues/5918).

### Move from System.Drawing to ImageSharp

Starting in .NET 6, System.Drawing.Common will only be supported on Windows (you can read more about this decision in this [design doc](https://github.com/dotnet/designs/blob/main/accepted/2021/system-drawing-win-only/system-drawing-win-only.md)).

To ensure ML.NET works great on all platforms, we will replace System.Drawing with the [ImageSharp](https://github.com/SixLabors/ImageSharp) graphics library.

*Related issues*:

- [#3154](https://github.com/dotnet/machinelearning/issues/3154)

### New features and scenarios

#### Named Entity Recognition (NER)

Named Entity Recognition, or NER, is the process of identifying and classifying/tagging information in text. For example, an NER model might look at a block of text and pick out "Seattle" and "Space Needle" and categorize them as locations or might find and tag "Microsoft" as a company.

Currently you can consume a pre-trained ONNX model in ML.NET for NER, but it is not possible to train a custom NER model in ML.NET which has been a highly requested feature for several years.

This year, we will work on adding support for training custom NER models in ML.NET.

*Related issues*:

- [#630](https://github.com/dotnet/machinelearning/issues/630)

#### Dynamic IDataView

In ML.NET, you must first define your model input and output schemas as new classes before loading data into an IDataView.

This year, we will work on adding a way to create dynamic IDataViews, meaning that you don't have to define your schemas beforehand and instead the shape of the training data defines the schemas.

*Related issues*:

- [#5895](https://github.com/dotnet/machinelearning/issues/5895)

#### Multivariate time series forecasting

Currently ML.NET only supports univariate time series forecasting with the [SSA algorithm](https://docs.microsoft.com/dotnet/api/microsoft.ml.transforms.timeseries.ssaforecastingestimator?view=ml-dotnet) which is currently being [added to Model Builder](https://github.com/dotnet/machinelearning-modelbuilder/issues/1750).

Univariate time series has one time-dependent variable whose values only depend on its past values through time. Multivariate time series has more than one time-dependent variable where each variable depends on its past values as well as the other variables.

This year, we will work on adding support for multivariate time series forecasting to ML.NET.

*Related issues*:

- [#5638](https://github.com/dotnet/machinelearning/issues/5638)
- [#1696](https://github.com/dotnet/machinelearning/issues/1696)

#### Multilabel Classification

Currently, ML.NET's classification algorithms will return one Predicted Label as well as an array of Scores which correspond to each possible class. However, mapping each label to the Score is currently not a great experience.

This year we will work on making the prediction info more user-friendly so that it is easy to assign multiple classes to one prediction.

*Related issues*:

- [#3909](https://github.com/dotnet/machinelearning/issues/3909)
- [#2278](https://github.com/dotnet/machinelearning/issues/2278)

### Model explainability & Responsible AI

Model Explainability and Responsible AI are becoming increasingly important areas of focus in the Machine Learning space and at Microsoft. Model explainability and fairness features are important because they let you debug and improve your models and answer questions about bias, building trust, and complying with regulations.

ML.NET currently offers two main model explainability features: [Permutation Feature Importance](https://docs.microsoft.com/dotnet/api/microsoft.ml.permutationfeatureimportanceextensions?view=ml-dotnet) (PFI) and the [Feature Contribution Calculator](https://docs.microsoft.com/dotnet/api/microsoft.ml.transforms.featurecontributioncalculatingestimator?view=ml-dotnet) (FCC).

We got a lot of feedback that the PFI API was difficult to use, so our first step is to improve the current experience in ML.NET. These improvements can be tracked in this [issue](https://github.com/dotnet/machinelearning/issues/5625) which will be merged soon.

This year we also plan to expand the number of model explainability and fairness features. We are currently working on this plan and will update the roadmap as we finalize which model explainability and fairness techniques we will bring into ML.NET.

### Define the plan for data prep

While we are working on developing the features mentioned above, we will also be working on our plan for data preparation and wrangling in .NET.

#### DataFrame API

The plan for data prep will include the roadmap for the DataFrame API (Microsoft.Data.Analysis) which we will add and update to this Roadmap doc.

*Related issues*:

- [#5870](https://github.com/dotnet/machinelearning/issues/5870)
- [#5716](https://github.com/dotnet/machinelearning/issues/5716)
- [#1696](https://github.com/dotnet/machinelearning/issues/1696)
