# The ML.NET Roadmap

The goal of ML.NET is to democratize machine learning for .NET developers. This document outlines the current roadmap for ML.NET framework and APIs.

To see the plans for ML.NET tooling, check out the [Model Builder repo](https://github.com/dotnet/machinelearning-modelbuilder).

## Goals through June 2022

### Keep docs, samples, and repo up to date

We heard your feedback loud and clear that our outdated docs and samples were a top pain point for using ML.NET. We have invested more resources into making sure our Docs stay relevant and that we add documentation for new features faster as well as add more relevant samples.

You can file issues for ML.NET documentation in the [dotnet/docs repo](https://github.com/dotnet/docs) and for ML.NET samples in the [dotnet/machinelearning-samples](https://github.com/dotnet/machinelearning-samples) repo.

We are also taking steps to organize the dotnet/machinelearning repo and updating our triage processes so that we can address your issues and feedback faster.

### Get on the .NET release schedule

ML.NET is .NET, and just like .NET, it's here to stay. So, we've decided to align with the .NET release schedule.

This means that we will ship our next major version of ML.NET with .NET 6.0.

While we'll have major releases of ML.NET once a year with the major .NET releases, we will still be shipping production-ready preview version releases in between so that we can continue to deliver awesome new features throughout the year.

### Deep learning

This past year we've been working on our plan for deep learning in .NET, and this year we will execute that plan to expand our deep learning support.

As part of this plan, we will:

- Make ONNX model consumption via ML.NET easier
- Productionize [TorchSharp](https://github.com/xamarin/TorchSharp) and
- Build a bridge between TorchSharp and ML.NET. This includes using TorchSharp to power simplified ML.NET APIs for:
  - Scenario-focused transfer learning scenarios (like object detection)
  - Generic transfer learning for custom scenarios that don't fit the scenario-focused APIs
  - Building neural networks from scratch

Read more about the deep learning plan and leave your feedback in this [tracking issue](https://github.com/dotnet/machinelearning/issues/X).

*Related issues*:

- [#5372](https://github.com/dotnet/machinelearning/issues/5372)

### New features and scenarios

#### Named Entity Recognition (NER)

Named Entity Recognition, or NER, is the process of identifying and classifying/tagging information in text. For example, an NER model might look at a block of text and pick out "Seattle" and "Space Needle" and categorize them as locations or might find and tag "Microsoft" as a company.

Currently you can consume a pre-trained ONNX model in ML.NET for NER, but it is not possible to train a custom NER model in ML.NET which has been a highly requested feature for several years.

X.

*Related issues*:

- [#630](https://github.com/dotnet/machinelearning/issues/630)

#### Dynamic IDataView

*Related issues*:

- [#5895](https://github.com/dotnet/machinelearning/issues/5895)

#### Multivariate time series forecasting

Currently ML.NET only supports univariate time series forecasting with the [SSA algorithm](https://docs.microsoft.com/dotnet/api/microsoft.ml.transforms.timeseries.ssaforecastingestimator?view=ml-dotnet) which is currently being [added to Model Builder](linkhere).



*Related issues*:

- [#5638](https://github.com/dotnet/machinelearning/issues/5638)
- [#1696](https://github.com/dotnet/machinelearning/issues/1696)

#### Multilabel Classification

*Related issues*:

- [#3909](https://github.com/dotnet/machinelearning/issues/3909)

### Model explainability & Responsible AI

Model Explainability and Responsible AI are becoming increasingly important areas of focus in the Machine Learning space and at Microsoft. Model explainability and fairness features are important because they let you debug and improve your models and answer questions about bias, building trust, and complying with regulations.

ML.NET currently offers two main model explainability features: [Permutation Feature Importance](https://docs.microsoft.com/dotnet/api/microsoft.ml.permutationfeatureimportanceextensions?view=ml-dotnet) (PFI) and the [Feature Contribution Calculator](https://docs.microsoft.com/dotnet/api/microsoft.ml.transforms.featurecontributioncalculatingestimator?view=ml-dotnet) (FCC).

We got a lot of feedback that the PFI API was difficult to use, so our first step is to improve the current experience in ML.NET. These improvements can be tracked in this [issue](https://github.com/dotnet/machinelearning/issues/5625) which will be merged soon.

This year we also plan to expand the number of model explainability and fairness features. We are currently working on this plan and will update the roadmap as we finalize which model explainability and fairness techniques we will bring into ML.NET.

### DataFrame API

*Related issues*:

- [#5716](https://github.com/dotnet/machinelearning/issues/5716)
- [#1696](https://github.com/dotnet/machinelearning/issues/1696)

### Define the plan for data prep

While we are working on developing the features mentioned above, we will also be working on our plan for data preparation and wrangling in ML.NET.

## Have feedback or want to contribute?

ML.NET is a community effort and we welcome community feedback on our plans. The best way to give feedback is to [open an issue](https://github.com/dotnet/machinelearning/issues/new/choose) in this repo.

We also invite contributions. The [up-for-grabs issues](https://github.com/dotnet/machinelearning/issues?q=is%3Aopen+is%3Aissue+label%3Aup-for-grabs) on GitHub are a good place to start.
