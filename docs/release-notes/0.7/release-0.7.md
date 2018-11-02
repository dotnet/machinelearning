# ML.NET 0.7 Release Notes

Today we are excited to release ML.NET 0.7, which our algorithms strongly
recommend you to try out! This release enables making recommendations with
matrix factorization, identifying unusual events with anomaly detection,
adding custom transformations to your ML pipeline, and more! We also have a
small surprise for those who work in teams that use both .NET and Python.
Finally, we wanted to thank the many new contributors to the project since the
last release! 

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

* Added Matrix factorization for recommendation problems
  ([#1263](https://github.com/dotnet/machinelearning/pull/1263))

    * Matrix factorization (MF) is a common approach to recommendations when
      you have data on how users rated items in your catalog. For example, you
      might know how users rated some movies and want to recommend which other
      movies they are likely to watch next.
    * ML.NET's MF uses [LIBMF](https://github.com/cjlin1/libmf).
    * Example usage of MF can be found
      [here](https://github.com/dotnet/machinelearning/blob/d68388a1c9994a5b429b194b64b2b0782834cb78/docs/samples/Microsoft.ML.Samples/Dynamic/MatrixFactorization.cs).
      The example is general but you can imagine that the matrix rows
      correspond to users, matrix columns correspond to movies, and matrix
      values correspond to ratings. This matrix would be quite sparse as users
      have only rated a small subset of the catalog.
    * Note: [ML.NET
      0.3](https://github.com/dotnet/machinelearning/blob/d68388a1c9994a5b429b194b64b2b0782834cb78/docs/release-notes/0.3/release-0.3.md)
      included Field-Aware Factorization Machines (FFM) as a learner for
      binary classification. FFM is a generalization of MF, but there are a
      few differences:
        * FFM enables taking advantage of other information beyond the rating
          a user assigns to an item (e.g. movie genre, movie release date,
          user profile). 
        * FFM is currently limited to binary classification (the ratings needs
          to be converted to 0 or 1), whereas MF solves a regression problem
          (the ratings can be continuous numbers).
        * If the only information available is the user-item ratings, MF is
          likely to be significantly faster than FFM.
        * A more in-depth discussion can be found
          [here](https://www.csie.ntu.edu.tw/~cjlin/talks/recsys.pdf).

* Enabled anomaly detection scenarios
  ([#1254](https://github.com/dotnet/machinelearning/pull/1254))

    * [Anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection)
      enables identifying unusual values or events. It is used in scenarios
      such as fraud detection (identifying suspicious credit card
      transactions) and server monitoring (identifying unusual activity). 
    * This release includes the following anomaly detection techniques:
      SSAChangePointDetector, SSASpikeDetector, IidChangePointDetector, and
      IidSpikeDetector. 
    * Example usage can be found
      [here](https://github.com/dotnet/machinelearning/blob/7fb76b026d0035d6da4d0b46bd3f2a6e3c0ce3f1/test/Microsoft.ML.TimeSeries.Tests/TimeSeriesDirectApi.cs).

* Enabled using ML.NET in Windows x86 apps
  ([#1008](https://github.com/dotnet/machinelearning/pull/1008))

    * ML.NET can now be used in x86 apps. 
    * Some components that are based on external dependencies (e.g.
      TensorFlow) will not be available in x86. Please open an issue on GitHub
      for discussion if this blocks you.

* Added the `CustomMappingEstimator` for custom data transformations
  [#1406](https://github.com/dotnet/machinelearning/pull/1406)

    * ML.NET has a wide variety of data transformations for pre-processing and
      featurizing data (e.g. processing text, images, categorical features,
      etc.).
    * However, there might be application-specific transformations that would
      be useful to do within an ML.NET pipeline (as opposed to as a
      pre-processing step). For example, calculating [cosine
      similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between two
      text columns (after featurization) or something as simple as creating a
      new column that adds the values in two other columns.
    * An example of the `CustomMappingEstimator` can be found
      [here](https://github.com/dotnet/machinelearning/blob/d68388a1c9994a5b429b194b64b2b0782834cb78/test/Microsoft.ML.Tests/Transformers/CustomMappingTests.cs#L55).

* Consolidated several API concepts in `MLContext`
  [#1252](https://github.com/dotnet/machinelearning/pull/1252)

    * `MLContext` replaces `LocalEnvironment` and `ConsoleEnvironment` but
      also includes properties for ML tasks like
      `BinaryClassification`/`Regression`, various transforms/trainers, and
      evaluation. More information can be found in
      [#1098](https://github.com/dotnet/machinelearning/issues/1098).
    * Example usage can be found
      [here](https://github.com/dotnet/machinelearning/blob/d68388a1c9994a5b429b194b64b2b0782834cb78/docs/code/MlNetCookBook.md)

* Open sourced [NimbusML](https://github.com/microsoft/nimbusml): experimental
  Python bindings for ML.NET. 

    * Some teams at Microsoft found it useful to use ML.NET capabilities in
      Python environments. NimbusML provides Python APIs to ML.NET and easily
      integrates into [Scikit-Learn](http://scikit-learn.org/stable/)
      pipelines. 
    * Note that NimbusML is an experimental project without the same support
      level as ML.NET.

### Acknowledgements

Shoutout to [dzban2137](https://github.com/dzban2137),
[beneyal](https://github.com/beneyal),
[pkulikov](https://github.com/pkulikov),
[amiteshenoy](https://github.com/amiteshenoy),
[DAXaholic](https://github.com/DAXaholic),
[Racing5372](https://github.com/Racing5372),
[ThePiranha](https://github.com/ThePiranha),
[helloguo](https://github.com/helloguo),
[elbruno](https://github.com/elbruno),
[harshsaver](https://github.com/harshsaver),
[f1x3d](https://github.com/f1x3d), [rauhs](https://github.com/rauhs),
[nihitb06](https://github.com/nihitb06),
[nandaleite](https://github.com/nandaleite),
[timitoc](https://github.com/timitoc),
[feiyun0112](https://github.com/feiyun0112),
[Pielgrin](https://github.com/Pielgrin),
[malik97160](https://github.com/malik97160),
[Niladri24dutta](https://github.com/Niladri24dutta),
[suhailsinghbains](https://github.com/suhailsinghbains),
[terop](https://github.com/terop), [Matei13](https://github.com/Matei13),
[JorgeAndd](https://github.com/JorgeAndd), and the ML.NET team for their
contributions as part of this release! 