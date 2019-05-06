

# Machine Learning for .NET

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers.

ML.NET allows .NET developers to develop their own models and infuse custom machine learning into their applications, using .NET, even without prior expertise in developing or tuning machine learning models.

ML.NET was originally developed in Microsoft Research, and evolved into a significant framework over the last decade and is used across many product groups in Microsoft like Windows, Bing, PowerPoint, Excel and more.

ML.NET enables machine learning tasks like classification (for example: support text classification, sentiment analysis) and regression (for example, price-prediction).

Along with these ML capabilities, this first release of ML.NET also brings the first draft of .NET APIs for training models, using models for predictions, as well as the core components of this framework such as learning algorithms, transforms, and ML data structures. 

## Installation

[![NuGet Status](https://img.shields.io/nuget/vpre/Microsoft.ML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML/)

ML.NET runs on Windows, Linux, and macOS using [.NET Core](https://github.com/dotnet/core), or Windows using .NET Framework. 64 bit is supported on all platforms. 32 bit is supported on Windows, except for TensorFlow, LightGBM, and ONNX related functionality.

The current release is 1.0.0. Check out the [release notes](docs/release-notes/1.0.0/release-1.0.0.md) to see what's new.

First, ensure you have installed [.NET Core 2.1](https://www.microsoft.com/net/learn/get-started) or later. ML.NET also works on the .NET Framework 4.6.1 or later, but 4.7.2 or later is recommended.

Once you have an app, you can install the ML.NET NuGet package from the .NET Core CLI using:
```
dotnet add package Microsoft.ML
```

or from the NuGet package manager:
```
Install-Package Microsoft.ML
```

Or alternatively, you can add the Microsoft.ML package from within Visual Studio's NuGet package manager or via [Paket](https://github.com/fsprojects/Paket).

Daily NuGet builds of the project are also available in our [MyGet](https://dotnet.myget.org/feed/dotnet-core/package/nuget/Microsoft.ML) feed:

> [https://dotnet.myget.org/F/dotnet-core/api/v3/index.json](https://dotnet.myget.org/F/dotnet-core/api/v3/index.json)

## Building

To build ML.NET from source please visit our [developers guide](docs/project-docs/developer-guide.md).

[![codecov](https://codecov.io/gh/dotnet/machinelearning/branch/master/graph/badge.svg?flag=production)](https://codecov.io/gh/dotnet/machinelearning)

|    | Debug | Release |
|:---|----------------:|------------------:|
|**CentOS**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Centos_x64_Netcoreapp30&configuration=Centos_x64_NetCoreApp30%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Centos_x64_Netcoreapp30&configuration=Centos_x64_NetCoreApp30%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Ubuntu**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Ubuntu_x64_Netcoreapp21&configuration=Ubuntu_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Ubuntu_x64_Netcoreapp21&configuration=Ubuntu_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**macOS**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=MacOS_x64_Netcoreapp21&configuration=MacOS_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=MacOS_x64_Netcoreapp21&configuration=MacOS_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Windows x64**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_Netcoreapp21&configuration=Windows_x64_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_Netcoreapp21&configuration=Windows_x64_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Windows FullFramework**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_NetFx461&configuration=Windows_x64_NetFx461%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_NetFx461&configuration=Windows_x64_NetFx461%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Windows x86**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x86_Netcoreapp21&configuration=Windows_x86_NetCoreApp21%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x86_Netcoreapp21&configuration=Windows_x86_NetCoreApp21%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|
|**Windows NetCore3.0**|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_Netcoreapp30&configuration=Windows_x64_NetCoreApp30%20Debug_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|[![Build Status](https://dev.azure.com/dnceng/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master&jobName=Windows_x64_Netcoreapp30&configuration=Windows_x64_NetCoreApp30%20Release_Build)](https://dev.azure.com/dnceng/public/_build/latest?definitionId=104&branchName=master)|

## Contributing

We welcome contributions! Please review our [contribution guide](CONTRIBUTING.md).

## Community

Please join our community on Gitter [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This project has adopted the code of conduct defined by the [Contributor Covenant](https://contributor-covenant.org/) to clarify expected behavior in our community.
For more information, see the [.NET Foundation Code of Conduct](https://dotnetfoundation.org/code-of-conduct).

## Examples

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

## API Documentation

See the [ML.NET API Reference Documentation](https://docs.microsoft.com/en-us/dotnet/api/?view=ml-dotnet).

## Samples

We have a [repo of samples](https://github.com/dotnet/machinelearning-samples) that you can look at.

## License

ML.NET is licensed under the [MIT license](LICENSE).

## .NET Foundation

ML.NET is a [.NET Foundation](https://www.dotnetfoundation.org/projects) project.

There are many .NET related projects on GitHub.

- [.NET home repo](https://github.com/Microsoft/dotnet) - links to 100s of .NET projects, from Microsoft and the community.
