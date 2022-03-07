## Overview
Experiment API is a set of API build to work with `SweepablePipeline`. Its aim is to make the interaction among `Tuner`, `Search Space` and `Sweepable Pipeline` transparent to external customers.

## Problem
Suppose that you have `sweepable pipeline`, `tuner` and `searchSpace`, and you want to optimize the  `sweepable pipeline` over given `searchSpace` using that `tuner`. In order to make that happen, without `Experiment API`, you would need to mannually interact with tuner and searchSpace, building mlnet training pipeline, train and evaluate model, and tracing the best model/parameter. Thus process expose too many details and would not be considered easy-to-use for users to start with.

```csharp
// this training process just expose too many details.
var pipeline, tuner;

// search space comes with pipeline.
var searchSpace = pipeline.SearchSpace;

tuner.SetSearchSpace(searchSPace);
foreach(var parameter in tuner.Proposal())
{
    // construct ML.Net pipeline from parameter
    var mlPipeline = pipeline.BuildTrainingPipeline(parameter);
    
    // evaluate
    var model = mlPipeline.Fit(trainData);
    var score = model.Evaluate(testData);

    // code to save best model
    if(score > bestScore)
    {
        bestScore = score;
        bestModel = model;
    }

    // update tuner with score
    tuner.Update(parameter, score);
}
```

## Solution: Experiment API

With Experiment api, we can make `pipeline`, `tuner` and `searchspace` transparent to users so they don't have to know how those parts work with each other. What replaces them is a higher-level concept: `Experiment`. `Experiment` will take the input from users, like training time, searching strategy, train/test/validation dataset, model saving strategy... After all input is given, experiment will take care the rest of training process.

```csharp
// Experiment api.
var pipeline, tuner;

var experiment = pipeline.CreateExperiment(trainTime = 100, trainDataset = "train.csv", split = "cv", folds = 10, metric = "AUC", tuner = tuner, monitor = monitor);

// or fluent-style api
experiment = pipeline.CreateExperiment();

experiment.SetTrainingTime(100)
    .SetDataset(trainDataset = "train.csv", split = "cv", fold = 10)
    .SetEvaluationMetric(metric = "AUC") // or a lambda function which return a score
    .SetTuner(tuner)
    .SetMonitor(monitor);

experiment.Run()
monitor.ShowProgress();

// trial 1: score ... parameter ...
// trial 2: score ... parameter ...
```

### default classifiers and regressors

It will be useful if we provide an API that returns a combination of all available trainers with default search spaces.

```csharp
var featurizePipeline

// regression
var pipeline = featurizePipeline.Append(context.AutoML().Regressors(labelColumn = "label", useLgbm = true, useFastTree = false, ...));

// binary classification
var pipeline = featurizePipeline.Append(context.AutoML().BinaryClassification(labelColumn = "label", useLgbm = true, useFastTree = false, ...));

// multi-class classification
var pipeline = featurizePipeline.Append(context.AutoML().MultiClassification(labelColumn = "label", useLgbm = true, useFastTree = false, ...));

// univariant forecasting
var pipeline = featurizePipeline.Append(context.AutoML().Forcasting(labelColumn = "label", horizon ...));

// create Experiment
var exp = pipeline.CreateExperiment();
...

```