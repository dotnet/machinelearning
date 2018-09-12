# ML.NET 0.5 Release Notes

Today we are excited to release ML.NET 0.5. This release adds
[TensorFlow](https://www.tensorflow.org/) model scoring as a transform to
ML.NET. This enables using an existing TensorFlow model within an ML.NET
experiment. In addition to this, we have continued the work on new APIs that
enable currently missing functionality. We welcome feedback and contributions
to the conversation: relevant issues can be found
[here](https://github.com/dotnet/machinelearning/projects/4). A simple example
of the new APIs can be found
[here](https://github.com/dotnet/machinelearning/blob/21b61447a342718c93f4b47ef8b5f2ec6d9f0c44/test/Microsoft.ML.Tests/Scenarios/Api/AspirationalExamples.cs).

### Installation

ML.NET supports Windows, MacOS, and Linux. See [supported OS versions of .NET
Core
2.0](https://github.com/dotnet/core/blob/master/release-notes/2.0/2.0-supported-os.md)
for more details.

You can install ML.NET NuGet from the CLI using:
```
dotnet add package Microsoft.ML
```

From package manager:
```
Install-Package Microsoft.ML
```

### Release Notes

Below are some of the highlights from this release.

* Added a TensorFlow model scoring transform (TensorFlowTransform)
  ([#704](https://github.com/dotnet/machinelearning/pull/704))

    * [TensorFlow](https://www.tensorflow.org/) is a popular machine learning
      toolkit that enables training deep neural networks (and general numeric
      computations).
    * This transform enables taking an existing TensorFlow model, either
      trained by you or downloaded from somewhere else, and get the scores
      from the model in ML.NET.
    * For now, these scores can be used within a `LearningPipeline` as inputs
      to a learner. However, with the upcoming ML.NET APIs, the scores from
      the TensorFlow model will be directly accessible.
    * The implementation of this transform is based on code from
      [TensorFlowSharp](https://github.com/migueldeicaza/TensorFlowSharp).
    * Example usage of the transform with the existing `LearningPipeline` API
      can be found
      [here](https://github.com/dotnet/machinelearning/blob/6ac380a4d3f44ee7b015461f74c4298b0ed5184b/test/Microsoft.ML.Tests/Scenarios/TensorflowTests.cs)
    * In the future, we will add functionality in ML.NET to enable identifying
      the expected inputs and outputs of TensorFlow models. For now, the
      TensorFlow APIs or a tool like
      [Netron](https://github.com/lutzroeder/Netron) can be used.

Additional issues closed in this milestone can be found
[here](https://github.com/dotnet/machinelearning/milestone/4?closed=1).

### Acknowledgements

Shoutout to [adamsitnik](https://github.com/adamsitnik),
[Jongkeun](https://github.com/Jongkeun), and the ML.NET team for their
contributions as part of this release! 