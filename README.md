# Machine Learning for .NET

[ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) is a cross-platform open-source machine learning (ML) framework for .NET.

ML.NET allows developers to easily build, train, deploy, and consume custom models in their .NET applications without requiring prior expertise in developing machine learning models or experience with other programming languages like Python or R. The framework provides data loading from files and databases, enables data transformations, and includes many ML algorithms.

With ML.NET, you can train models for a [variety of scenarios](https://docs.microsoft.com/dotnet/machine-learning/resources/tasks), like classification, forecasting, and anomaly detection.

You can also consume both TensorFlow and ONNX models within ML.NET which makes the framework more extensible and expands the number of supported scenarios.

## Getting started with machine learning and ML.NET

- Learn more about the [basics of ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet).
- Build your first ML.NET model by following our [ML.NET Getting Started tutorial](https://dotnet.microsoft.com/learn/ml-dotnet/get-started-tutorial/intro).
- Check out our [documentation and tutorials](https://docs.microsoft.com/dotnet/machine-learning/).
- See the [API Reference documentation](https://docs.microsoft.com/dotnet/api/?view=ml-dotnet).
- Clone our [ML.NET Samples GitHub repo](https://github.com/dotnet/machinelearning-samples) and run some sample apps.
- Take a look at some [ML.NET Community Samples](https://github.com/dotnet/machinelearning-samples/blob/main/docs/COMMUNITY-SAMPLES.md).
- Watch some videos on the [ML.NET videos YouTube playlist](https://aka.ms/mlnetyoutube).

## Roadmap

Take a look at ML.NET's [Roadmap](ROADMAP.md) to see what the team plans to work on in the next year.

## Operating systems and processor architectures supported by ML.NET

ML.NET runs on Windows, Linux, and macOS using .NET Core, or Windows using .NET Framework.

ML.NET also runs on ARM64, Apple M1, and Blazor Web Assembly. However, there are some [limitations](docs/project-docs/platform-limitations.md).

64-bit is supported on all platforms. 32-bit is supported on Windows, except for TensorFlow and LightGBM related functionality.

## ML.NET NuGet packages status

[![NuGet Status](https://img.shields.io/nuget/vpre/Microsoft.ML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML/)

## Release notes

Check out the [release notes](docs/release-notes) to see what's new. You can also read the [blog posts](https://devblogs.microsoft.com/dotnet/category/ml-net/) for more details about each release.

## Using ML.NET packages

First, ensure you have installed [.NET Core 2.1](https://www.microsoft.com/net/learn/get-started) or later. ML.NET also works on the .NET Framework 4.6.1 or later, but 4.7.2 or later is recommended.

Once you have an app, you can install the ML.NET NuGet package from the .NET Core CLI using:
```
dotnet add package Microsoft.ML
```

or from the NuGet Package Manager:
```
Install-Package Microsoft.ML
```

Alternatively, you can add the Microsoft.ML package from within Visual Studio's NuGet package manager or via [Paket](https://github.com/fsprojects/Paket).

Daily NuGet builds of the project are also available in our Azure DevOps feed:

> [https://pkgs.dev.azure.com/dnceng/public/_packaging/MachineLearning/nuget/v3/index.json](https://pkgs.dev.azure.com/dnceng/public/_packaging/MachineLearning/nuget/v3/index.json)

## Building ML.NET (For contributors building ML.NET open source code)

To build ML.NET from source please visit our [developer guide](docs/project-docs/developer-guide.md).

[![codecov](https://codecov.io/gh/dotnet/machinelearning/branch/main/graph/badge.svg?flag=production)](https://codecov.io/gh/dotnet/machinelearning)

|    | Debug | Release |
|:---|----------------:|------------------:|
|**CentOS**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Centos_x64_NetCoreApp31&configuration=Centos_x64_NetCoreApp31%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Centos_x64_NetCoreApp31&configuration=Centos_x64_NetCoreApp31%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Ubuntu**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Ubuntu_x64_NetCoreApp21&configuration=Ubuntu_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Ubuntu_x64_NetCoreApp21&configuration=Ubuntu_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**macOS**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=MacOS_x64_NetCoreApp21&configuration=MacOS_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=MacOS_x64_NetCoreApp21&configuration=MacOS_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Windows x64**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetCoreApp21&configuration=Windows_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetCoreApp21&configuration=Windows_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Windows FullFramework**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetFx461&configuration=Windows_x64_NetFx461%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetFx461&configuration=Windows_x64_NetFx461%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Windows x86**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x86_NetCoreApp21&configuration=Windows_x86_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x86_NetCoreApp21&configuration=Windows_x86_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|
|**Windows NetCore3.1**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetCoreApp31&configuration=Windows_x64_NetCoreApp31%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=main&jobName=Windows_x64_NetCoreApp31&configuration=Windows_x64_NetCoreApp31%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=main)|

## Release process and versioning

Major releases of ML.NET are shipped once a year with the major .NET releases, starting with ML.NET 1.7 in November 2021 with .NET 6, then ML.NET 2.0 with .NET 7, etc. We will maintain release branches to optionally service ML.NET with bug fixes and/or minor features on the same cadence as .NET servicing.

Check out the [Release Notes](docs/release-notes) to see all of the past ML.NET releases.

## Contributing

We welcome contributions! Please review our [contribution guide](CONTRIBUTING.md).

## Community

- Join our community on [Discord](https://aka.ms/dotnet-discord).
- Tune into the [.NET Machine Learning Community Standup](https://dotnet.microsoft.com/live/community-standup) every other Wednesday at 10AM Pacific Time.

This project has adopted the code of conduct defined by the [Contributor Covenant](https://contributor-covenant.org/) to clarify expected behavior in our community.
For more information, see the [.NET Foundation Code of Conduct](https://dotnetfoundation.org/code-of-conduct).

## Code examples

Here is a code snippet for training a model to predict sentiment from text samples. You can find complete samples in the [samples repo](https://github.com/dotnet/machinelearning-samples).

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

## License

ML.NET is licensed under the [MIT license](LICENSE), and it is free to use commercially.

## .NET Foundation

ML.NET is a part of the [.NET Foundation](https://www.dotnetfoundation.org/projects).
