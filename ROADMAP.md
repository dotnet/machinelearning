# The ML.NET Roadmap

The goal of ML.NET is to democratize machine learning for .NET developers. This document outlines the current roadmap for the ML.NET framework and APIs.

To see the plans for ML.NET tooling, check out the [Model Builder repo](https://github.com/dotnet/machinelearning-modelbuilder/issues?q=is%3Aissue+is%3Aopen+label%3AEpic).

## Feedback and contributions

ML.NET is a community effort and we welcome community feedback on our plans. The best way to give feedback is to [open an issue](https://github.com/dotnet/machinelearning/issues/new/choose) in this repo.

We also invite contributions. The [first good issue](https://github.com/dotnet/machinelearning/labels/good%20first%20issue) and [up-for-grabs issues](https://github.com/dotnet/machinelearning/issues?q=is%3Aopen+is%3Aissue+label%3Aup-for-grabs) on GitHub are a good place to start. You can also help work on any of the features we've listed below or work on features that you want to add to the framework.

## Goals through November 2023

The following sections outline the major areas and features we plan to work on in the next year.

Note, that this is an aspirational list of what we hope to get to. Many of the items on this list will require more investigations and design, which can result in changes in our plans. We may have to cut things as we go, or we may be able to add more things.

As we prioritize, cost, and continue planning, we will try to keep the Roadmap up to date to reflect our progress and learnings.

### Keep docs, samples, and repo up to date

We heard your feedback loud and clear that our outdated docs and samples were a top pain point when learning and using ML.NET.

As we continue to drive improvements in ML.NET and add new features, it's important to us that you're successful in adopting and using these enhanced capabilities to deliver value. Documentation and samples are an key part of that. Over the next year we plan to dedicate more resoures to deliver quality documentation and samples.  

This [tracking issue](https://github.com/dotnet/docs/issues/32112) lists a few of the areas we plan to build documentation around over the next few months, 

You can file issues and make suggestions for ML.NET documentation in the [dotnet/docs repo](https://github.com/dotnet/docs) and for ML.NET samples in the [dotnet/machinelearning-samples](https://github.com/dotnet/machinelearning-samples) repo.

We are also taking steps to organize the [dotnet/machinelearning](https://github.com/dotnet/machinelearning) repo and updating our triage processes so that we can address your issues and feedback faster. Issues will be linked to version releases in the [Projects](https://github.com/dotnet/machinelearning/projects) section of the repo so you can see what we're actively working on and when we plan to release.

### Deep learning

This past year we've been working on our plan for deep learning in .NET, and now we are ready to execute that plan to expand ML.NET's deep learning support.

As part of this plan, we will:

1. Make it easier to consume ONNX models in ML.NET using the ONNX Runtime (RT)
1. Continue to bring more scenario-based APIs backed by TorchSharp transformer-based architectures. The next few scenarios we're looking to enable are:
    - Object detection
    - Named Entity Recognition (NER)
    - Question Answering
1. Enable integrations with TorchSharp for scenarios and models not supported out of the box by ML.NET.
1. Accelerate deep learning workflows by improving batch support and enabling easier use of accelerators such as ONNX Runtime Execution Providers.

Read more about the deep learning plan and leave your feedback in this [tracking issue](https://github.com/dotnet/machinelearning/issues/5918).

Performance-related improvements are being tracked in this [issue](https://github.com/dotnet/machinelearning/issues/6422).

### LightGBM

LightGBM is a flexible framework for classical machine learning tasks such as classification and regression. To make the best of the features LightGBM provides, we plan to:

- Upgrade the version included in ML.NET to the latest LightGBM version
- Make interoperability with other frameworks easier by enabling saving and loading models in the native LightGBM format.

We're tracking feedback and progress on LightGBM in this [issue](https://github.com/dotnet/machinelearning/issues/6337). 

### Define the plan for data prep

While we are working on developing the features mentioned above, we will also be working on our plan for data preparation and wrangling in .NET.

#### DataFrame API

Data processing is an important part of any analytics and machine learning workflow. This process often involves loading, inspecting, transforming, and visualizing your data. We've heard your feedback that one of the ways you'd like to perform some of these tasks is by using the DataFrame API in the `Microsoft.Data.Analysis` NuGet package. This past year we worked on making the loading experience more robust and adding new column types like DateTime to enable better interoperability with the ML.NET `IDataView`. In the next year, we plan to continue focusing on the areas of:

- Improving interoperability with the `IDataView` by supporting `VBuffer` and `KeyType` columns.
- Improving stability for common operations such as loading, filtering, merging, and joining data. 

This [tracking issue](https://github.com/dotnet/machinelearning/issues/6144) is intended to collect feedback and track progress of the work we're doing on the DataFrame. 

#### Untyped / Dynamic training and prediction engine

In ML.NET, you must first define your model input and output schemas as new classes before loading data into an IDataView.

In ML.NET 2.0 we made progress in this area by leveraging the `InferColumns` method as a source of information for the AutoML `Featurizer`. The `Featurizer` helps automate common data preprocessing tasks to get your data in a state that's ready for training. When used together, you don't have to define schema classes. This is convenient when working with large datasets.

Similarly, using the DataFrame API, you can load data into the `DataFrame`, apply any transformations to your data, use the data as input to an ML.NET pipeline, train a model, and use the model to make predictions. At that point, you can call `ToDataFrame` and convert your predictions to a DataFrame making it easier to post-process and visualize those predictions. As mentioned in the `DataFrame` section, there's still some work that needs to be done to make the experience of going between a `DataFrame` and `IDataView` seamless but the `DataFrame` is another option for working with ML.NET without having to define schemas.  

However, for single predictions, there are currently no solutions. For tasks like forecasting, we've made modifications to how the PredictionEngine behaves. As a result, we expect being able to do something similar to enable untyped prediction engines.  

While the details of what that implementation looks like, this year we plan to provide ways to create dynamic prediciton engines, meaning that you don't have to define your schemas beforehand and instead the shape of the training data defines the schemas.

*Related issues*:

- [#5895](https://github.com/dotnet/machinelearning/issues/5895)

### Model explainability & Responsible AI

Model Explainability and Responsible AI are becoming increasingly important areas of focus in the Machine Learning space and at Microsoft. Model explainability and fairness features are important because they let you debug and improve your models and answer questions about bias, building trust, and complying with regulations.

ML.NET currently offers two main model explainability features: [Permutation Feature Importance](https://docs.microsoft.com/dotnet/api/microsoft.ml.permutationfeatureimportanceextensions?view=ml-dotnet) (PFI) and the [Feature Contribution Calculator](https://docs.microsoft.com/dotnet/api/microsoft.ml.transforms.featurecontributioncalculatingestimator?view=ml-dotnet) (FCC).

In ML.NET 2.0 we made improvements to simpify the PFI API. 

We also worked on porting fairness assessment and mitigation components from the [Fairlearn library to .NET](https://github.com/dotnet/machinelearning/pull/6279). Those components are not integrated into ML.NET yet. This year we plan on exposing them into ML.NET.  