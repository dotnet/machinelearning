# Machine Learning for .NET

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers with the same code that powers machine learning across many Microsoft products, including Power BI, Windows Defender, and Azure. 

ML.NET allows .NET developers to develop/train their own models and infuse custom machine learning into their applications using .NET, even without prior expertise in developing or tuning machine learning models. It provides data loading from files and databases, enables data transformations and includes many ML algorithms.

ML.NET enables machine learning (ML) tasks like classification (for example, text classification, sentiment analysis), regression (for example, price prediction), and many other ML tasks such as anomaly detection, time-series-forecast, clustering, ranking, etc.

## Getting started with machine learning by using ML.NET

If you are new to machine learning, start by learning the basics from this collection of resources targeting ML.NET:

[Learn ML.NET](https://dotnet.microsoft.com/learn/ml-dotnet)

## ML.NET Documentation, tutorials and reference

Please check our [documentation and tutorials](https://docs.microsoft.com/en-us/dotnet/machine-learning/). 

See the [API Reference documentation](https://docs.microsoft.com/en-us/dotnet/api/?view=ml-dotnet).

## Sample apps

We have a GitHub repo with [ML.NET sample apps](https://github.com/dotnet/machinelearning-samples) with many scenarios such as Sentiment analysis, Fraud detection, Product Recommender, Price Prediction, Anomaly Detection, Image Classification, Object Detection and many more. 

In addition to the ML.NET samples provided by Microsoft, we're also highlighting many more samples created by the community showcased in this separate page [ML.NET Community Samples](https://github.com/dotnet/machinelearning-samples/blob/master/docs/COMMUNITY-SAMPLES.md)


## ML.NET videos playlist at YouTube

The [ML.NET videos playlist](https://aka.ms/mlnetyoutube) on YouTube contains several short videos. Each video focuses on a particular topic of ML.NET.

## Operating systems and processor architectures supported by ML.NET

ML.NET runs on Windows, Linux, and macOS using [.NET Core](https://github.com/dotnet/core), or Windows using .NET Framework. 

64 bit is supported on all platforms. 32 bit is supported on Windows, except for TensorFlow and LightGBM related functionality.

## ML.NET Nuget packages status

[![NuGet Status](https://img.shields.io/nuget/vpre/Microsoft.ML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML/)

## Release notes

Check out the [release notes](docs/release-notes) to see what's new.

## Using ML.NET packages

First, ensure you have installed [.NET Core 2.1](https://www.microsoft.com/net/learn/get-started) or later. ML.NET also works on the .NET Framework 4.6.1 or later, but 4.7.2 or later is recommended.

Once you have an app, you can install the ML.NET NuGet package from the .NET Core CLI using:
```
dotnet add package Microsoft.ML
```

or from the NuGet package manager:
```
Install-Package Microsoft.ML
```

Alternatively, you can add the Microsoft.ML package from within Visual Studio's NuGet package manager or via [Paket](https://github.com/fsprojects/Paket).

Daily NuGet builds of the project are also available in our Azure DevOps feed:

> [https://pkgs.dev.azure.com/dnceng/public/_packaging/MachineLearning/nuget/v3/index.json](https://pkgs.dev.azure.com/dnceng/public/_packaging/MachineLearning/nuget/v3/index.json)

## Building ML.NET (For contributors building ML.NET open source code)

To build ML.NET from source please visit our [developers guide](docs/project-docs/developer-guide.md).

[![codecov](https://codecov.io/gh/dotnet/machinelearning/branch/master/graph/badge.svg?flag=production)](https://codecov.io/gh/dotnet/machinelearning)

|    | Debug | Release |
|:---|----------------:|------------------:|
|**CentOS**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Centos_x64_NetCoreApp31&configuration=Centos_x64_NetCoreApp31%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Centos_x64_NetCoreApp31&configuration=Centos_x64_NetCoreApp31%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Ubuntu**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Ubuntu_x64_NetCoreApp21&configuration=Ubuntu_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Ubuntu_x64_NetCoreApp21&configuration=Ubuntu_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**macOS**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=MacOS_x64_NetCoreApp21&configuration=MacOS_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=MacOS_x64_NetCoreApp21&configuration=MacOS_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Windows x64**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_NetCoreApp21&configuration=Windows_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_NetCoreApp21&configuration=Windows_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Windows FullFramework**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_NetFx461&configuration=Windows_x64_NetFx461%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_NetFx461&configuration=Windows_x64_NetFx461%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Windows x86**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x86_NetCoreApp21&configuration=Windows_x86_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x86_NetCoreApp21&configuration=Windows_x86_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Windows NetCore3.1**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_NetCoreApp31&configuration=Windows_x64_NetCoreApp31%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_NetCoreApp31&configuration=Windows_x64_NetCoreApp31%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|

## Release process and versioning

Check out the [release process documentation](docs/release-notes) to understand the different kinds of ML.NET releases.

## Contributing

We welcome contributions! Please review our [contribution guide](CONTRIBUTING.md).

## Community

Please join our community on Gitter [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This project has adopted the code of conduct defined by the [Contributor Covenant](https://contributor-covenant.org/) to clarify expected behavior in our community.
For more information, see the [.NET Foundation Code of Conduct](https://dotnetfoundation.org/code-of-conduct).


## Code examples

Here is a snippet code for training a model to predict sentiment from text samples. You can find complete samples in [samples repo](https://github.com/dotnet/machinelearning-samples).

```C#
var dataPath = "sentiment.csv";
var mlContext = new MLContext();
var loader = mlContext.Data.CreateTextLoader(new[]
    {
        new TextLoader.Column("SentimentText", DataKind.String, 1),
        new TextLoader.Column("Label", DataKind.Boolean, 0),
    },
    hasHeader: true,
    separatorChar: ',');
var data = loader.Load(dataPath);
var learningPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
        .Append(mlContext.BinaryClassification.Trainers.FastTree());
var model = learningPipeline.Fit(data);
```

Now from the model we can make inferences (predictions):

```C#
var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
var prediction = predictionEngine.Predict(new SentimentData
{
    SentimentText = "Today is a great day!"
});
Console.WriteLine("prediction: " + prediction.Prediction);
```
A cookbook that shows how to use these APIs for a variety of existing and new scenarios can be found [here](docs/code/MlNetCookBook.md).

## License

ML.NET is licensed under the [MIT license](LICENSE) and it is free to use commercially.

## .NET Foundation

ML.NET is a [.NET Foundation](https://www.dotnetfoundation.org/projects) project.

There are many .NET related projects on GitHub.

- [.NET home repo](https://github.com/Microsoft/dotnet) - links to 100s of .NET projects, from Microsoft and the community.
