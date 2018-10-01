# ML.NET 0.6 Release Notes

Today we are excited to release ML.NET 0.6, the biggest release of ML.NET ever (or at least since 0.5)! This release unveils the first iteration of new ML.NET APIs. These APIs enable various new tasks that weren't possible with the old APIs. Furthermore, we have added a transform to get predictions from [ONNX](http://onnx.ai/) models, expanded functionality of the TensorFlow scoring transform, aligned various ML.NET types with .NET types, and more!

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

* New APIs for ML.NET
    
    * While the `LearningPipeline` APIs that were released with ML.NET 0.1 were easy to get started with, they had obvious limitations in functionality. Certain tasks that were possible with the internal version of ML.NET like inspecting model weights, creating a transform-only pipeline, and training from an initial predictor could not be done with `LearningPipeline`.
    * The important concepts for understanding the new API are introduced [here](https://github.com/dotnet/machinelearning/blob/3cdd3c8b32705e91dcf46c429ee34196163af6da/docs/code/MlNetHighLevelConcepts.md). 
    * A cookbook that shows how to use these APIs for a variety of existing and new scenarios can be found [here](https://github.com/dotnet/machinelearning/blob/3cdd3c8b32705e91dcf46c429ee34196163af6da/docs/code/MlNetCookBook.md). 
    * These APIs are still evolving, so we would love to hear any feedback or questions. 
    * The `LearningPipeline` APIs have moved to the `Microsoft.ML.Legacy` namespace.

* Added a transform to score ONNX models ([#942](https://github.com/dotnet/machinelearning/pull/942))

    * [ONNX](http://onnx.ai/) is an open model format that enables developers to more easily move models between different tools.
    * There are various [collections of ONNX models](https://github.com/onnx/models) that can be used for tasks like image classification, emotion recognition, and object detection.
    * The [ONNX transform](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.onnxtransform?view=ml-dotnet) in ML.NET enables providing some data to an existing ONNX model (such as the models above) and getting the score (prediction) from it.

* Enhanced TensorFlow model scoring functionality ([#853](https://github.com/dotnet/machinelearning/pull/853), [#862](https://github.com/dotnet/machinelearning/pull/862))

    * The [TensorFlow scoring transform](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.tensorflowtransform?view=ml-dotnet) released in ML.NET 0.5 enabled using 'frozen' TensorFlow models. In ML.NET 0.6, 'saved' TensorFlow models can also be used. 
    * An API was added to extract information about the nodes in a TensorFlow model. This can help identifying the input and output of a TensorFlow model. Example usage can be found [here](https://github.com/dotnet/machinelearning/blob/3cdd3c8b32705e91dcf46c429ee34196163af6da/src/Microsoft.ML.DnnAnalyzer/Microsoft.ML.DnnAnalyzer/DnnAnalyzer.cs).

* Replaced ML.NET's Dv type system with .NET's standard type system ([#863](https://github.com/dotnet/machinelearning/pull/863))

    * ML.NET previously had its own type system which helped it more efficiently deal with things like missing values (a common case in ML). This type system required users to work with types like `DvText`, `DvBool`, `DvInt4`, etc. 
    * This update replaces the Dv type system with .NET's standard type system to make ML.NET easier to use and to take advantage of innovation in .NET.

* Up to ~200x speedup in prediction engine performance ([#973](https://github.com/dotnet/machinelearning/pull/973))

    * This improvement leads to a significant speedup when making predictions for single records.

* Improved approach to dependency injection ([#970](https://github.com/dotnet/machinelearning/pull/970), [#1022](https://github.com/dotnet/machinelearning/pull/1022))

    * This enables ML.NET to be used in additional .NET app models without messy workarounds (e.g. Azure Functions).

Additional issues closed in this milestone can be found
[here](https://github.com/dotnet/machinelearning/milestone/5?closed=1).

### Acknowledgements

Shoutout to [feiyun0112](https://github.com/feiyun0112), [jwood803](https://github.com/jwood803), [adamsitnik](https://github.com/adamsitnik), and the ML.NET team for their contributions as part of this release! 