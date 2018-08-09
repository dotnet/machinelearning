# ML.NET 0.4 Release Notes

Today we are releasing ML.NET 0.4. During this release we have started
exploring new APIs for ML.NET that enable functionality that is missing from
the current APIs. We welcome feedback and contributions to the
conversation (relevant issues can be found [here](https://github.com/dotnet/machinelearning/projects/4)). While the
focus has been on designing the new APIs, we have also moved several
components from the internal codebase to ML.NET.

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

* Added SymSGD learner for binary classification
  ([#624](https://github.com/dotnet/machinelearning/pull/624))

    * [SymSGD](https://arxiv.org/abs/1705.08030) is a technique for
      parallelizing
      [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
      (Stochastic Gradient Descent). This enables it to sometimes perform
      faster than existing SGD implementations (e.g. [Hogwild
      SGD](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.stochasticgradientdescentbinaryclassifier?view=ml-dotnet)).
    * SymSGD is available for binary classification, but can be used in
      multiclass classification with
      [One-Versus-All](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.models.oneversusall?view=ml-dotnet)
    * SymSGD requires adding the Microsoft.ML.HalLearners NuGet package to your project
    * The current implementation in ML.NET does not yet have multi-threading
      enabled due to build system limitations (tracked by
      [#655](https://github.com/dotnet/machinelearning/issues/655)), but
      SymSGD can still be helpful in scenarios where you want to try many
      different learners and limit each of them to a single thread. 
    * Documentation can be found
      [here](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.symsgdbinaryclassifier?view=ml-dotnet)

* Added Word Embeddings Transform for text scenarios
  ([#545](https://github.com/dotnet/machinelearning/pull/545))

    * [Word embeddings](https://en.wikipedia.org/wiki/Word_embedding) is a
      technique for mapping words or phrases to numeric vectors of relatively low
      dimension (in comparison with the high dimensional n-gram extraction).
      These numeric vectors are intended to capture some of the meaning of the
      words so they can be used for training a better model. As an example,
      SSWE (Sentiment-Specific Word Embedding) can be useful for sentiment
      related tasks.
    * This transform enables using pretrained models to get the embeddings
      (i.e. the embeddings are already trained and available for use).
    * Several options for pretrained embeddings are available:
      [GloVe](https://nlp.stanford.edu/projects/glove/),
      [fastText](https://en.wikipedia.org/wiki/FastText), and
      [SSWE](http://anthology.aclweb.org/P/P14/P14-1146.pdf). The pretrained model is downloaded automatically on first use.
    * Documentation can be found
      [here](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.wordembeddings?view=ml-dotnet).
    
* Improved support for F# by allowing use of property-based row classes ([#616](https://github.com/dotnet/machinelearning/pull/616))
    
    * ML.NET now supports F# record types.
    * The ML.NET samples repository is being updated to include F# samples as part of [#36](https://github.com/dotnet/machinelearning-samples/pull/36).

Additional issues closed in this milestone can be found
[here](https://github.com/dotnet/machinelearning/milestone/3?closed=1).

### Acknowledgements

Shoutout to [dsyme](https://github.com/dsyme),
[SolyarA](https://github.com/SolyarA),
[dan-drews](https://github.com/dan-drews),
[bojanmisic](https://github.com/bojanmisic),
[jwood803](https://github.com/jwood803),
[sharwell](https://github.com/sharwell),
[JoshuaLight](https://github.com/JoshuaLight), and the ML.NET team for their
contributions as part of this release! 