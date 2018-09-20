# ML.NET 0.2 Release Notes

We would like to thank the community for the engagement so far and helping us
shape ML.NET.

Today we are releasing ML.NET 0.2. This release focuses on addressing
questions/issues, adding clustering to the list of supported machine learning
tasks, enabling using data from memory to train models, easier model
validation, and more.

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

* Added clustering to the list of supported machine learning tasks

    * Clustering is an unsupervised learning task that groups sets of items
      based on their features. It identifies which items are more similar to
      each other than other items. This might be useful in scenarios such as
      organizing news articles into groups based on their topics, segmenting
      users based on their shopping habits, and grouping viewers based on
      their taste in movies. 

    * ML.NET 0.2 exposes `KMeansPlusPlusClusterer` which implements [K-Means++
      clustering](https://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf)
      with [Yinyang K-means
      acceleration](https://www.microsoft.com/en-us/research/publication/yinyang-k-means-a-drop-in-replacement-of-the-classic-k-means-with-consistent-speedup/?from=http%3A%2F%2Fresearch.microsoft.com%2Fapps%2Fpubs%2Fdefault.aspx%3Fid%3D252149).
      [This
      test](https://github.com/dotnet/machinelearning/blob/78810563616f3fcb0b63eb8a50b8b2e62d9d65fc/test/Microsoft.ML.Tests/Scenarios/ClusteringTests.cs)
      shows how to use it (from
      [#222](https://github.com/dotnet/machinelearning/pull/222)).

* Train using data objects in addition to loading data from a file using
  `CollectionDataSource`. ML.NET 0.1 enabled loading data from a delimited
  text file. `CollectionDataSource` in ML.NET 0.2 adds the ability to use a
  collection of objects as the input to a `LearningPipeline`. See sample usage
  [here](https://github.com/dotnet/machinelearning/blob/78810563616f3fcb0b63eb8a50b8b2e62d9d65fc/test/Microsoft.ML.Tests/CollectionDataSourceTests.cs#L133)
  (from [#106](https://github.com/dotnet/machinelearning/pull/106)). 

* Easier model validation with cross-validation and train-test

    * [Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
      is an approach to validating how well your model statistically performs.
      It does not require a separate test dataset, but rather uses your
      training data to test your model (it partitions the data so different
      data is used for training and testing, and it does this multiple times).
      [Here](https://github.com/dotnet/machinelearning/blob/78810563616f3fcb0b63eb8a50b8b2e62d9d65fc/test/Microsoft.ML.Tests/Scenarios/SentimentPredictionTests.cs#L51)
      is an example for doing cross-validation (from
      [#212](https://github.com/dotnet/machinelearning/pull/212)).

    * Train-test is a shortcut to testing your model on a separate dataset.
      See example usage
      [here](https://github.com/dotnet/machinelearning/blob/78810563616f3fcb0b63eb8a50b8b2e62d9d65fc/test/Microsoft.ML.Tests/Scenarios/SentimentPredictionTests.cs#L36).

    * Note that the `LearningPipeline` is prepared the same way in both cases.
      
* Speed improvement for predictions: by not creating a parallel cursor for
  dataviews that only have one element, we get a significant speed-up for
  predictions (see
  [#179](https://github.com/dotnet/machinelearning/issues/179) for a few
  measurements).

* Updated `TextLoader` API: the `TextLoader` API is now code generated and was
  updated to take explicit declarations for the columns in the data, which is
  required in some scenarios. See
  [#142](https://github.com/dotnet/machinelearning/pull/142).

* Added daily NuGet builds of the project: daily NuGet builds of ML.NET are
  now available
  [here](https://dotnet.myget.org/feed/dotnet-core/package/nuget/Microsoft.ML).

Additional issues closed in this milestone can be found [here](https://github.com/dotnet/machinelearning/milestone/1?closed=1).

### Acknowledgements

Shoutout to tincann, rantri, yamachu, pkulikov, Sorrien, v-tsymbalistyi, Ky7m,
forki, jessebenson, mfaticaearnin, and the ML.NET team for their contributions
as part of this release! 
