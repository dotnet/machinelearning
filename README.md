# Machine Learning for .NET

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) provides state-of-the-art machine learning (ML) algorithms, transforms, and components, and powers ML pipelines in many Microsoft products.  Developed and used internally at Microsoft for over 5 years, the goal is to make ML.NET useful for all developers, data scientists, and information workers and helpful in all products, services, and devices.

ML.NET runs on Windows, Linux, and macOS - any platform where 64 bit [.NET Core](https://github.com/dotnet/core) or later is available.

With ML.NET you can use the latest ML algorithms to create and evaluate a model from training data. Once you have a model, you can add to your app just a few lines of .NET code to make predictions from the model. 

### Examples

Imagine you want to predict the sale price of a house. Given a large dataset of information about other houses, including their sale prices, you can use ML.NET to create and evaluate a model. Then, you can deploy the model with your app.

Here's a different example, with code, to train a model to predict sentiment from text samples. (You can see the complete sample [here](https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/Scenarios/Scenario3_SentimentPrediction.cs)):

```C#
var pipeline = new LearningPipeline();
pipeline.Add(new TextLoader<SentimentData>(dataPath, separator: ","));
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

### Installation

The current release is 0.1. Check out the [release notes](https://github.com/dotnet/machinelearning/blob/master/Documentation/release-notes/0.1/release-0.1.md).

First ensure you have installed [.NET Core 2.0](https://www.microsoft.com/net/learn/get-started) or later. ML.NET also works on the .NET Framework. Note that ML.NET currently must run in a 64 bit process.

Once you have an app, you can install ML.NET NuGet from the .NET Core CLI using:
```
dotnet add package Microsoft.ML
```

or from the package manager:
```
Install-Package Microsoft.ML
```

Or alternatively you can add the Microsoft.ML package from within Visual Studio's NuGet package manager.

### Building

To build ML.NET from source please visit our [developers guide.](https://github.com/dotnet/machinelearning/blob/master/Documentation/project-docs/developer-guide.md)

Live build status is coming soon.

### Contributing

We welcome contributions! Please review our [contribution guide](https://github.com/dotnet/machinelearning/blob/master/CONTRIBUTING.md).

## Community

Please join our community on Gitter [![Join the chat at https://gitter.im/dotnet/corefx](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This project has adopted the code of conduct defined by the [Contributor Covenant](http://contributor-covenant.org/) to clarify expected behavior in our community.
For more information, see the [.NET Foundation Code of Conduct](https://dotnetfoundation.org/code-of-conduct).

## License

ML.NET is licensed under the [MIT license](LICENSE).

## .NET Foundation

ML.NET is a [.NET Foundation](http://www.dotnetfoundation.org/projects) project.

There are many .NET related projects on GitHub.

- [.NET home repo](https://github.com/Microsoft/dotnet) - links to 100s of .NET projects, from Microsoft and the community.

