# ML.NET AutoML

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers.

ML.NET allows .NET developers to develop their own models and infuse custom machine learning into their applications, using .NET, even without prior expertise in developing or tuning machine learning models.

Automated machine learning (automated ML) builds high quality machine learning models for you by automating feature engineering, model, and hyperparameter selection. 
Bring a dataset that you want to build a model for, automated ML will give you a high quality machine learning model that you can use for predictions.

If you are new to Data Science, AutoML will help you get jumpstarted by simplifying machine learning model building. 
It abstracts you from needing to perform feature engineering, model selection, hyperparameter selection and in one step creates a high quality trained model for you to use.

If you are an experienced data scientist, AutoML will help increase your productivity by intelligently performing the feature engineering, model, and hyperparameter selection for your training and generates high quality models much more quickly than manually specifying several combinations of the parameters and running training jobs. 
AutoML provides visibility and access to all the training jobs and the performance characteristics of the models to help you further tune the pipeline if you desire.

## Installation

[![NuGet Status](https://img.shields.io/nuget/v/Microsoft.ML.AutoML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML.AutoML/)

ML.NET runs on Windows, Linux, and macOS using [.NET Core](https://github.com/dotnet/core), or Windows using .NET Framework. 64 bit is supported on all platforms. 32 bit is supported on Windows, except for TensorFlow, LightGBM, and ONNX related functionality.

First, ensure you have installed [.NET Core 2.1](https://www.microsoft.com/net/learn/get-started) or later. ML.NET also works on the .NET Framework 4.6.1 or later, but 4.7.2 or later is recommended.

Once you have an app, you can install the ML.NET AutoML NuGet package from the .NET Core CLI using:
```
dotnet add package Microsoft.ML.AutoML
```

or from the NuGet package manager:
```
Install-Package Microsoft.ML.AutoML
```

Or alternatively, you can add the Microsoft.ML.AutoML package from within Visual Studio's NuGet package manager or via [Paket](https://github.com/fsprojects/Paket).

Daily NuGet builds of the project are also available in our [MyGet](https://dotnet.myget.org/feed/dotnet-core/package/nuget/Microsoft.ML.AutoML) feed:

> [https://dotnet.myget.org/F/dotnet-core/api/v3/index.json](https://dotnet.myget.org/F/dotnet-core/api/v3/index.json)

## Building

To build ML.NET AutoML from source please visit our [developers guide](docs/project-docs/developer-guide.md).

|    | Debug | Release |
|:---|----------------:|------------------:|
|**Linux**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobname=Linux&configuration=Build_Debug)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=312&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobname=Linux&configuration=Build_Release)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=312&branch=master)|
|**macOS**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobname=macOS&configuration=Build_Debug)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=312&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobname=macOS&configuration=Build_Release)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=312&branch=master)|
|**Windows x64**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobname=Windows_x64&configuration=Build_Debug)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=312&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobname=Windows_x64&configuration=Build_Release)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=312&branch=master)|
|**Windows x86**|[![Build Status](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobName=Windows_x86&configuration=Build_Debug)](https://dnceng.visualstudio.com/public/_build/latest?definitionId=312?branchName=master)|[![Build Status](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobName=Windows_x86&configuration=Build_Release)](https://dnceng.visualstudio.com/public/_build/latest?definitionId=312?branchName=master)|
|**Core 3.0**|[![Build Status](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobName=core30&configuration=Build_Debug_Intrinsics)](https://dnceng.visualstudio.com/public/_build/latest?definitionId=312?branchName=master)|[![Build Status](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/machinelearning-automl-ci?branchName=master&jobName=core30&configuration=Build_Release_Intrinsics)](https://dnceng.visualstudio.com/public/_build/latest?definitionId=312?branchName=master)|

## Contributing

We welcome contributions! Please review our [contribution guide](https://github.com/dotnet/machinelearning/blob/master/CONTRIBUTING.md).

## Community

Please join our community on Gitter [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This project has adopted the code of conduct defined by the [Contributor Covenant](https://contributor-covenant.org/) to clarify expected behavior in our community.
For more information, see the [.NET Foundation Code of Conduct](https://dotnetfoundation.org/code-of-conduct).

## Examples

Here's an example of code to automatically train a model to predict sentiment from text samples.

```C#
// Example to come 

```

Now from the model we can make inferences (predictions):

```C#
var predictionEngine = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);
var prediction = predictionEngine.Predict(new SentimentData
{
    SentimentText = "Today is a great day!"
});
Console.WriteLine("prediction: " + prediction.Prediction);
```
A cookbook that shows how to use these APIs for a variety of existing and new scenarios can be found [here](docs/code/MlNetCookBook.md).


## Samples

We have a [repo of samples](https://github.com/dotnet/machinelearning-samples) that you can look at.

## License

ML.NET is licensed under the [MIT license](LICENSE).

## .NET Foundation

ML.NET is a [.NET Foundation](https://www.dotnetfoundation.org/projects) project.

There are many .NET related projects on GitHub.

- [.NET home repo](https://github.com/Microsoft/dotnet) - links to 100s of .NET projects, from Microsoft and the community.
