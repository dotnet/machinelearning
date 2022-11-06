# [ML.NET](http://dot.net/ml) 2.0.0

This release is going out alongside .NET 7 continuing with our plan to align with the broader .NET release cycle.

## Release Notes

The main themes for this release are:

- New natural language processing (NLP) APIs powered by [TorchSharp](https://github.com/dotnet/torchsharp)
- AutoML improvements

Below are some of the highlights from this release:

### Deep Learning

- **Text Classification API** ([#4512](https://github.com/dotnet/machinelearning/pull/4512)) - The Text Classification API lets use you a TorchSharp implementation of the [NAS-BERT](https://arxiv.org/abs/2105.14444) model developed by Microsoft Research, and added it to ML.NET. The NAS-BERT model is a variant of BERT. If you're interested in learning more, you can check out the [blog post](https://devblogs.microsoft.com/dotnet/introducing-the-ml-dotnet-text-classification-api-preview/) from earlier this year.
- **Sentence Similarity API** ([#6390](https://github.com/dotnet/machinelearning/pull/6390)) - The Sentence Similarity API uses the same NAS-BERT model as the Text Classification API. However, instead of classifying text, it determines the similarity of two sentence. Similar to a regression problem, given two sentences as input, the model provides a score or numerical value determining the similarity between those two sentences.
- **Tokenizers support** ([#6272](https://github.com/dotnet/machinelearning/pull/6272)) - As part of the work to introduce the Text Classification and Sentence Similarity APIs, we needed tokenizers for processing text. In this release, the `EnglishRoberta` tokenization model used by the text classification and sentence similarity APIs is supported. For more generic scenarios other than `EnglishRoberta`, there's also support for [Byte-Pair Encoding (BPE)](https://en.wikipedia.org/wiki/Byte_pair_encoding) algorithm which means you can load custom vocabulary files and use them to process your text. These tokenization APIs are available as part of the `Microsoft.ML.Tokenizers` NuGet package.

### AutoML

- **Featurizer API** ([#6205](https://github.com/dotnet/machinelearning/pull/6205)) - The Featurizer API automates parts of the data preparation steps in the model development cycle.
- **Sweepable API** ([#6108](https://github.com/dotnet/machinelearning/pull/6108)) - The Sweepable API allows ML.NET users to create their own search space and pipeline for hyper-parameter optimization(HPO). For more information, see the [proposal](https://github.com/dotnet/machinelearning/issues/5992).
- **Search Space API** ([#6132](https://github.com/dotnet/machinelearning/pull/6132)) - The Search Space API allows you to configure the hyperparameter search range of parameters for a pipeline and estimator.
- **Tuner API** ([#6140](https://github.com/dotnet/machinelearning/pull/6140)) - The Tuner API provides the option to define and choose the tuner which defines the strategy used to navigate the search space during hyperparameter optimization. As of this release you have an option to choose from:
  - CostFrugalTuner: low-cost HPO algorithm, this is an implementation of [Frugal  Optimization for Cost-related Hyperparameters](https://arxiv.org/abs/2005.01571).
  - SMAC: Bayesian optimization using random forest as regression model.
  - EciCostFrugalTuner: CostFrugalTuner for hierarchical search space. This will be used as default tuner if AutoMLExperiment.SetPipeline get called.
  - GridSearch
  - RandomSearch  
- **Experiment API** ([#6140](https://github.com/dotnet/machinelearning/pull/6140)) - The Experiment API builds on the Sweepable API and enables defaults for the separate pipeline, search space, and tuner making it simpler to create and train models using AutoML. For more information, see the [proposal](https://github.com/dotnet/machinelearning/pull/6118).

## Enhancements

### ONNX

- **Support for 1 unknown dimension** ([#6265](https://github.com/dotnet/machinelearning/pull/6265))
- **Support saving model with ONNX GPU flag** ([#6143](https://github.com/dotnet/machinelearning/pull/6143)) - Prior to this change, when you saved an ML.NET model that contained ONNX as part of the pipeline, the flag to use the GPU was not saved. This change fixes that.  

### DataFrame

- **DateTime column support** ([#6302](https://github.com/dotnet/machinelearning/pull/6302)) - Adds the `DateTime` type as a `PrimitiveDataFrameColumn`. This allows better conversion between `IDataView` and the `DataFrame`.
- **Improve performance of DataFrame merge operation** ([#6150](https://github.com/dotnet/machinelearning/pull/6150)) Thanks @mzasov!
- **Create DataFrame from TabularDataResource** ([#6099](https://github.com/dotnet/machinelearning/pull/6099) & [#6123](https://github.com/dotnet/machinelearning/pull/6123)) - This change enables users to convert KQL and SQL kernel values to `DataFrame` in notebooks. Thanks @colombod!

## Breaking changes

- **Remove System.Drawing dependency** ([#6363](https://github.com/dotnet/machinelearning/pull/6363)) - Starting with .NET 6, `System.Drawing` is [only supported on Windows](https://learn.microsoft.com/dotnet/core/compatibility/core-libraries/6.0/system-drawing-common-windows-only). As a result, we've replaced it with the `MLImage` class for image handing. In code where you previously represented image data as `Bitmap`, use `MLImage` instead.
- **Default Transformer scope set to Scoring** - ML.NET provides the ability to set the `TransformerScope` depending on the purpose. Prior to this change, calling `Transform` to apply the transformations to data defaulted to use the `Everything` scope unless otherwise specified. As of this release, the scope has changed to `Scoring` which means for scoring scenarios, you no longer need to provide an empty label as part of your inputs.
