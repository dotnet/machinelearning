# Machine Learning in .NET

ML.NET provides state-of-the-art machine learning (ML) algorithms, transforms, and components, and powers ML pipelines in many Microsoft products.  Developed and used internally at Microsoft for over 5 years, the goal is to make ML.NET useful for all developers, data scientists, and information workers and helpful in all products, services, and devices.

### Build Status

Coming soon

### Installation

You can install ML.NET NuGet from the CLI using:
```
dotnet add package Microsoft.ML
```

From package manager:
```
Install-Package Microsoft.ML
```
For an example of getting started with .NET Core, see [here](https://www.microsoft.com/net/learn/get-started).

### Building
To build ML.NET from source go to [developers guide](https://github.com/dotnet/machinelearning/blob/master/Documentation/project-docs/developer-guide.md)

### Example

Simple snippet to train a model for sentiment classification (See the complete sample [here](https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/Scenarios/Scenario3_SentimentPrediction.cs)):
```C#
var pipeline = new LearningPipeline();
pipeline.Add(new TextLoader<SentimentData>(dataPath, separator: ","));
pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
pipeline.Add(new FastTreeBinaryClassifier());
var model = pipeline.Train<SentimentData, SentimentPrediction>();
```

Infer the trained model for predictions:

```C#
SentimentData data = new SentimentData
{
    SentimentText = "Today is a great day!"
};

SentimentPrediction prediction = model.Predict(data);

Console.WriteLine("prediction: " + prediction.Sentiment);
```

### Code of Conduct

This project has adopted the code of conduct defined by the [Contributor Covenant](http://contributor-covenant.org/) to clarify expected behavior in our community.
For more information, see the [.NET Foundation Code of Conduct](https://dotnetfoundation.org/code-of-conduct).

## License

ML.NET is licensed under the [MIT license](LICENSE.TXT).

## .NET Foundation

ML.NET is a [.NET Foundation](http://www.dotnetfoundation.org/projects) project.

There are many .NET related projects on GitHub.

- [.NET home repo](https://github.com/Microsoft/dotnet) - links to 100s of .NET projects, from Microsoft and the community.