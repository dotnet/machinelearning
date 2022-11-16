# AutoML.Net Sweepable API proposal
## Overview
Sweepable api allows mlnet users to create their own search space and pipeline for hyper-parameter optimization (HPO). It comes with three major part: [search space](#search-space), [sweepable estimator](#sweepable-estimator) and [tuner](#tuner). And all API lives under `Sweepable()` extension (for now).

## search space
Search space defines a range of hyper-parameter for tuner to search from. Sweepable API provides two way to create a search space.

via attribute
```csharp
public class Option
{
    [Range(2, 32768, init: 2, logBase: true)]
    public int WindowSize {get; set;}

    // one of [2, 3, 4]
    [Choice(2, 3, 4)]
    public int SeriesLength {get; set;}

    // one of [true, false]
    [Choice]
    public bool UseSoftmax {get; set;}

    // nested search space
    [Option]
    public Option AnotherOption {get;set;}
}

var ss = new SearchSpace<Option>();

// each search space has a 1-d feature space where each feature is [0, 1). And search space will handle the mapping between hpo space and feature space so that tuner only needs to perform search on feature space, which both dimension and range are known.
var parameter = ss.SampleFromFeatureSpace(new []{0,0,0,0,0,0});

// auto-binding
parameter.WindowSize.Should().Be(2);
parameter.SeriesLength.Should().Be(2);
parameter.UseSoftmax.Should().BeTrue();
parameter.AnotherOption.WindowSize.Should().Be(2);

// search space can also map parameter back to feature space
ss.MappingToFeatureSpace(parameter).Should().BeEquivalantTo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
```

or correspondingly, via scratch
``` csharp
var ss = new SearchSpace();
ss.Add("WindowSize", new UniformIntOption(2, 32768, true, 2));
ss.Add("SeriesLength", new ChoiceOption(2,3,4));
ss.Add("UseSoftmax", new ChoiceOption(true, false));
ss.Add("AnotherOption", ss.Clone());

var parameter = ss.SampleFromFeatureSpace(new []{0,0,0,0,0,0});

// auto-binding doesn't exist for scratch api
parameter["WindowSize"].AsType<int>.Should().Be(2);
parameter["SeriesLength"].AsType<int>.Should().Be(2);
parameter["UseSoftmax"].AsType<bool>.Should().BeTrue();
parameter["AnotherOption"]["WindowSize"].AsType<int>.Should().Be(2);

// search space can also map parameter back to feature space
ss.MappingToFeatureSpace(parameter).Should().BeEquivalantTo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
```

Currently, in order to make auto-binding work, there's a limitation on the parameter type that can be added to search space, which has to be either a Json primitive type or a nested search space.

## sweepable estimator
sweepable estimator allows user to combine search space with estimators in a similar way how ml.net estimator/pipeline are created. You use `CreateSweepableEstimator`, which accepts a lambda function and a search space to create a sweepable estimator. And it also provides `.Append` extension method so you can append sweepable estimator similiar with how you append other ml.net extimator. The bellow example presents how to create a pipeline with two sweepable estimators, one for text featurizor and the other for fast tree, for `titanic` dataset via `Sweepable()` extension.

``` csharp
var context = new MLContext();
var fastTreeSS = new SearchSpace<FastTreeOption>();
var textFeaturizeSS = new SearchSpace<FeaturizeTextOption>();

var pipeline = context.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair(@"Sex", @"Sex"), new InputOutputColumnPair(@"Embarked", @"Embarked") })
                .Append(context.Transforms.Concatenate(@"TextFeature", @"Name", "Ticket", "Cabin"))
                .Append(context.Sweepable().CreateSweepableEstimator(
                    (mlContext, option) =>
                    {
                        var textOption = new TextFeaturizingEstimator.Options
                        {
                            CaseMode = option.CaseMode,
                            KeepDiacritics = option.KeepDiacritics,
                            KeepNumbers = option.KeepNumbers,
                            KeepPunctuations = option.KeepPunctuations,
                            CharFeatureExtractor = new WordBagEstimator.Options()
                            {
                                NgramLength = option.WordBagEstimatorOption.NgramLength,
                                UseAllLengths = option.WordBagEstimatorOption.UseAllLengths,
                                Weighting = option.WordBagEstimatorOption.WeightingCriteria,
                            },
                        };

                        return mlContext.Transforms.Text.FeaturizeText("TextFeature", textOption);
                    },
                    textFeaturizeSS))
                .Append(context.Transforms.Concatenate(@"Features", new[] { @"Sex", @"Embarked", @"Pclass", @"Age", @"SibSp", @"Parch", @"Fare", "TextFeature" }))
                .Append(context.Transforms.Conversion.ConvertType("Survived", "Survived", Data.DataKind.Boolean))
                .Append(context.Sweepable().CreateSweepableEstimator(
                    (mlContext, option) => mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Survived", featureColumnName: "Features", numberOfLeaves: option.NumberOfLeavenumberOfTrees: option.NumberOfTrees),
                    fastTreeSS))
                .Append(context.BinaryClassification.Calibrators.Naive(labelColumnName: @"Survived", scoreColumnName: @"Score"));
```

After sweepable pipeline is created, one can call `BuildTrainingPipeline` to convert it to a ml.net pipeline.

## tuner

tuner takes in a search space and performs hpo algos. There're a few default tuning alogs provided by Sweepable API (grid search/random search), and there'll be more smart hpo algos coming soon.

The way to use a tuner is quite similar with the way to use an enumerator. The code below shows how tuner works with search space and sweepable estimator/pipeline

```csharp
var ss = pipeline.SearchSpace
var tuner = new GridSearchTuner(ss);
var df = DataFrame.LoadCsv(@"titanic.csv");
var trainTestSplit = context.Data.TrainTestSplit(df, 0.1);
var bestAccuracy = 0.0;
var i = 0;
foreach (var param in tuner.Propose())
{
    Console.WriteLine($"trial {i++}");

    // convert sweepable pipeline to ml.net pipeline
    var trainingPipeline = pipeline.BuildTrainingPipeline(context, param);
    var model = trainingPipeline.Fit(trainTestSplit.TrainSet);
    var eval = model.Transform(trainTestSplit.TestSet);
    var accuracy = context.BinaryClassification.Evaluate(eval, "Survived").Accuracy;
    if (accuracy > bestAccuracy)
    {
        Console.WriteLine("Found best accuracy");
        Console.WriteLine("Current best parameter");
        Console.WriteLine(JsonConvert.SerializeObject(param));
        bestAccuracy = accuracy;

        Console.WriteLine($"Trial {i}: Current Best Accuracy {bestAccuracy}, Current Accuracy {accuracy}");
    }
}

```

You can visit [here](https://github.com/dotnet/machinelearning-tools/blob/main/src/Microsoft.ML.ModelBuilder.AutoMLService.Example/Program.cs) to try out the complete training code.

## The difference between sweepable api and existing api in AutoML.Net
The exsiting API in AutoML.Net performs hpo on pre-defined search space and learners with smac tuning algo while sweepable api allows user to customize those settings: they can define their own search space, they can create pipeline similarly with how it is created in ml.net, and they can pick the tuner which suit their experiment the best.

## Q & A

What's the difference between Sweepable API and AutoML.Net experiments
- AutoML.Net experiments provide an oobe experience for automl with fixed hpo-related option, while Sweepable API requires user to set up those options (search space, pipeline, tuner) manually before training. Sweepable API is not aimed to replace existing AutoML.Net API, but to provide another options for those customers whose request can't be satified by default automl experiment.

Will AutoML.Net benefits from Sweepable API
- Yes it will, along with sweepable API, we can also leverage [flaml](https://github.com/microsoft/FLAML) technique in the tuning algothrim, which helps improves tuning speed and performance.

Why do we need sweepable API, who will be the beneficaries, what's the most common user-case.
- Sweepable API is prepared for HPO, which is a feature required by community from ever since mlnet being released. ([#613](https://github.com/dotnet/machinelearning/issues/613), [#2260](https://github.com/dotnet/machinelearning/issues/2260), [#5930](https://github.com/dotnet/machinelearning/issues/5930),[#1875](https://github.com/dotnet/machinelearning-modelbuilder/issues/1875) etc...). The major beneficiaries will be those who'd like to run automl on different search space or trainers other than the fixed parameters provided by AutoML.Net. The most common user-case, as I'm imagining, is on Notebook along with DataFrame API.

What's the timeline for intergrating Sweepable API into ML.Net
- it depends. Sweepable API has three major parts and all are available in model builder repo. On top of which is a thin layer which makes the three part work with ML.Net. So if the plan is only moving that thin layer into AutoML.Net first, and make the rest of three parts open source later, it will take 2-3 weeks. If we want to move all parts into mlnet all together, it will cost much more time.


## Current feedback
- the lambda function in `CreateSweepableEstimator` should not accept `MLContext` as first parameter.
- Should provide a higher level API for training similar with Experiment API in AutoML.Net.
- (From @torronen) Provides ways to access immediate results (accuracy/parameter/pipeline) to help better understand training process
