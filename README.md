

# Machine Learning for .NET

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers.

ML.NET allows .NET developers to develop their own models and infuse custom ML into their applications without prior expertise in developing or tuning machine learning models, all in .NET.

ML.NET was originally developed in Microsoft Research and evolved into a significant framework over the last decade and is used across many product groups in Microsoft like Windows, Bing, PowerPoint, Excel and more.

With this first preview release ML.NET enables ML tasks like classification (for example, support text classification, sentiment analysis) and regression (for example, price-prediction). 

Along with these ML capabilities this first release of ML.NET also brings the first draft of .NET APIs for training models, using models for predictions, as well as the core components of this framework such as learning algorithms, transforms, and ML data structures. 

## Installation

[![NuGet Status](https://img.shields.io/nuget/v/Microsoft.ML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML/)

ML.NET runs on Windows, Linux, and macOS - any platform where 64 bit [.NET Core](https://github.com/dotnet/core) or later is available.

The current release is 0.5. Check out the [release notes](docs/release-notes/0.5/release-0.5.md).

First ensure you have installed [.NET Core 2.0](https://www.microsoft.com/net/learn/get-started) or later. ML.NET also works on the .NET Framework. Note that ML.NET currently must run in a 64 bit process.

Once you have an app, you can install the ML.NET NuGet package from the .NET Core CLI using:
```
dotnet add package Microsoft.ML
```

or from the NuGet package manager:
```
Install-Package Microsoft.ML
```

Or alternatively you can add the Microsoft.ML package from within Visual Studio's NuGet package manager or via [Paket](https://github.com/fsprojects/Paket).

Daily NuGet builds of the project are also available in our MyGet feed:

> [https://dotnet.myget.org/F/dotnet-core/api/v3/index.json](https://dotnet.myget.org/F/dotnet-core/api/v3/index.json)

## Building

To build ML.NET from source please visit our [developers guide](docs/project-docs/developer-guide.md).

|    | x64 Debug | x64 Release |
|:---|----------------:|------------------:|
|**Linux**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|
|**macOS**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|
|**Windows**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|

## Contributing

We welcome contributions! Please review our [contribution guide](CONTRIBUTING.md).

## Community

Please join our community on Gitter [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This project has adopted the code of conduct defined by the [Contributor Covenant](https://contributor-covenant.org/) to clarify expected behavior in our community.
For more information, see the [.NET Foundation Code of Conduct](https://dotnetfoundation.org/code-of-conduct).

## Examples

Here's an example of code to train a model to predict sentiment from text samples. 
(You can see the complete sample [here](test/Microsoft.ML.Tests/Scenarios/SentimentPredictionTests.cs)):

```C#
var pipeline = new LearningPipeline();
pipeline.Add(new TextLoader(dataPath).CreateFrom<SentimentData>(separator: ','));
pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
pipeline.Add(new FastTreeBinaryClassifier());
var model = pipeline.Train<SentimentData, SentimentPrediction>();
```

Now from the model we can make inferences (predictions):

```C#
SentimentData data = new SentimentData
{
    SentimentText = "Today is a great day!"
};

SentimentPrediction prediction = model.Predict(data);

Console.WriteLine("prediction: " + prediction.Sentiment);
```
## Samples

We have a [repo of samples](https://github.com/dotnet/machinelearning-samples) that you can look at.

## License

ML.NET is licensed under the [MIT license](LICENSE).

## .NET Foundation

ML.NET is a [.NET Foundation](https://www.dotnetfoundation.org/projects) project.

There are many .NET related projects on GitHub.

- [.NET home repo](https://github.com/Microsoft/dotnet) - links to 100s of .NET projects, from Microsoft and the community.

