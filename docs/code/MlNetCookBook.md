# ML.NET Cookbook

This document is intended to provide essential samples for common usage patterns of ML.NET. 
It is advisable to be at least minimally familiar with [high-level concepts of ML.NET](MlNetHighLevelConcepts.md), otherwise the terminology in this document may be foreign to you.

## How to use this cookbook

Developers often work by copying and pasting source code from somewhere and then adapting it to their needs. We do it all the time.

So, we decided to embrace the pattern and provide an authoritative set of example usages of ML.NET, for many common scenarios that you may encounter.
These examples are multi-purpose:

- They can kickstart your development, so that you don't start from nothing,
- They are annotated and verbose, so you have easier time adapting them to your needs.

Each sample also contains a snippet of the data file used in the sample. We mostly use snippets from our test datasets for that.

Please feel free to search this page and use any code that suits your needs.

### List of recipes

- [How do I load data from a text file?](#how-do-i-load-data-from-a-text-file)
- [How do I load data with many columns from a CSV?](#how-do-i-load-data-with-many-columns-from-a-csv)
- [How do I look at the intermediate data?](#how-do-i-look-at-the-intermediate-data)
- [How do I train a regression model?](#how-do-i-train-a-regression-model)
- [How do I verify the model quality?](#how-do-i-verify-the-model-quality)
- [How do I save and load the model?](#how-do-i-save-and-load-the-model)
- [How do I use the model to make one prediction?](#how-do-i-use-the-model-to-make-one-prediction)
- [What if my training data is not in a text file?](#what-if-my-training-data-is-not-in-a-text-file)
- [I want to look at my model's coefficients](#i-want-to-look-at-my-models-coefficients)
- [What is normalization and why do I need to care?](#what-is-normalization-and-why-do-i-need-to-care)
- [How do I train my model on categorical data?](#how-do-i-train-my-model-on-categorical-data)
- [How do I train my model on textual data?](#how-do-i-train-my-model-on-textual-data)
- [How do I train using cross-validation?](#how-do-i-train-using-cross-validation)
- [Can I mix and match static and dynamic pipelines?](#can-i-mix-and-match-static-and-dynamic-pipelines)

### General questions about the samples

As this document is reviewed, we found that certain general clarifications are in order about all the samples together. We try to address them in this section.

- *My compiler fails to find some of the methods that are present in the samples!*
This is because we rely on extension methods a lot, and they only become available after you say `using TheRightNamespace`.
We are still re-organizing the namespaces, and trying to improve the story. In the meantime, the following namespaces prove useful for extension methods:
```csharp
using Microsoft.ML.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
```

- *Why is there two ways of doing things? Which one is better, static or dynamic?*
We don't know yet, that's why we have two ways! 

If you are loading an existing model from a stream, there's no need to use static types (and it's also pretty hard to do). Also, if the data view's schema is only known at runtime, there is no way to use static types. In other scenarios we tend to prefer the static types, since this way gives you compiler support: it's more likely that, if your code compiles, it will also work as intended.

- *Why do we call `reader.MakeNewEstimator` to create a pipeline?*
In the static pipeline, we need to know the two 'schema' types: the input and the output to the pipeline.
One of them is already known: typically, the output schema of `reader` (which is the same as the schema of `reader.Read()`) is also the input schema of the learning pipeline. 

The call to `x.MakeNewEstimator` is only using the `x`'s *schema* to create an empty pipeline, it doesn't use anything else from `x`. So, the following three lines would create the exactly same (empty) pipeline:
```csharp
var p1 = reader.MakeNewEstimator();
var p2 = reader.Read(dataLocation).MakeNewEstimator();
var p3 = p1.MakeNewEstimator();
```

- *Can we use `reader` to read more than one file?*
Absolutely! This is why we separated `reader` from the data. This is completely legitimate (and recommended):
```csharp
var trainData = reader.Read(trainDataLocation);
var testData = reader.Read(testDataLocation);
```

## How do I load data from a text file?

`TextLoader` is used to load data from text files. You will need to specify what are the data columns, what are their types, and where to find them in the text file. 

Note that it's perfectly acceptable to read only some columns of a file, or read the same column multiple times.

[Example file](../../test/data/adult.tiny.with-schema.txt):
```
Label	Workclass	education	marital-status
0	Private	11th	Never-married
0	Private	HS-grad	Married-civ-spouse
1	Local-gov	Assoc-acdm	Married-civ-spouse
1	Private	Some-college	Married-civ-spouse
```

This is how you can read this data:
```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Create the reader: define the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        // A boolean column depicting the 'target label'.
        IsOver50K: ctx.LoadBool(0),
        // Three text columns.
        Workclass: ctx.LoadText(1),
        Education: ctx.LoadText(2),
        MaritalStatus: ctx.LoadText(3)),
    hasHeader: true);

// Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
var data = reader.Read(dataPath);
```

If the schema of the data is not known at compile time, or too cumbersome, you can revert to the dynamically-typed API: 
```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Create the reader: define the data columns and where to find them in the text file.
var reader = new TextLoader(mlContext, new TextLoader.Arguments
{
    Column = new[] {
        // A boolean column depicting the 'label'.
        new TextLoader.Column("IsOver50k", DataKind.BL, 0),
        // Three text columns.
        new TextLoader.Column("Workclass", DataKind.TX, 1),
        new TextLoader.Column("Education", DataKind.TX, 2),
        new TextLoader.Column("MaritalStatus", DataKind.TX, 3)
    },
    // First line of the file is a header, not a data row.
    HasHeader = true
});

// Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
var data = reader.Read(dataPath);
```

## How do I load data from multiple files?

You can again use the `TextLoader`, and specify an array of files to its Read method.
The files need to have the same schema (same number and type of columns) 

[Example file1](../../test/data/adult.train):
[Example file2](../../test/data/adult.test):
```
Label	Workclass	education	marital-status
0	Private	11th	Never-married
0	Private	HS-grad	Married-civ-spouse
1	Local-gov	Assoc-acdm	Married-civ-spouse
1	Private	Some-college	Married-civ-spouse
```

This is how you can read this data:
```csharp
// Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
// as well as the source of randomness.
var env = new LocalEnvironment();

// Create the reader: define the data columns and where to find them in the text file.
var reader = TextLoader.CreateReader(env, ctx => (
        // A boolean column depicting the 'target label'.
        IsOver50K: ctx.LoadBool(14),
        // Three text columns.
        Workclass: ctx.LoadText(1),
        Education: ctx.LoadText(3),
        MaritalStatus: ctx.LoadText(5)),
    hasHeader: true);

// Now read the files (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
var data = reader.Read(exampleFile1, exampleFile2);
```

## How do I load data with many columns from a CSV?
`TextLoader` is used to load data from text files. You will need to specify what are the data columns, what are their types, and where to find them in the text file. 

When the input file contains many columns of the same type, always intended to be used together, we recommend reading them as a *vector column* from the very start: this way the schema of the data is cleaner, and we don't incur unnecessary performance costs.

[Example file](../../test/data/generated_regression_dataset.csv):
```
-2.75,0.77,-0.61,0.14,1.39,0.38,-0.53,-0.50,-2.13,-0.39,0.46,140.66
-0.61,-0.37,-0.12,0.55,-1.00,0.84,-0.02,1.30,-0.24,-0.50,-2.12,148.12
-0.85,-0.91,1.81,0.02,-0.78,-1.41,-1.09,-0.65,0.90,-0.37,-0.22,402.20
0.28,1.05,-0.24,0.30,-0.99,0.19,0.32,-0.95,-1.19,-0.63,0.75,443.51
```

Reading this file using `TextLoader`:
```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Create the reader: define the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        // We read the first 11 values as a single float vector.
        FeatureVector: ctx.LoadFloat(0, 10),
        // Separately, read the target variable.
        Target: ctx.LoadFloat(11)
    ),
    // Default separator is tab, but we need a comma.
    separator: ',');


// Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
var data = reader.Read(dataPath);
```


If the schema of the data is not known at compile time, or too cumbersome, you can revert to the dynamically-typed API: 
```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Create the reader: define the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(new[] {
        // We read the first 10 values as a single float vector.
        new TextLoader.Column("FeatureVector", DataKind.R4, new[] {new TextLoader.Range(0, 9)}),
        // Separately, read the target variable.
        new TextLoader.Column("Target", DataKind.R4, 10)
    },
    // Default separator is tab, but we need a comma.
    s => s.Separator = ",");

// Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
var data = reader.Read(dataPath);
```

## How do I look at the intermediate data?

Oftentimes, when we construct the experiment, we want to make sure that the data processing 'up to a certain moment' produces the results that we want. With ML.NET it is not very easy to do: since all ML.NET operations are lazy, the objects we construct are just 'promises' of data.

We will need to create the cursor and scan the data to obtain the actual values. One way to do this is to use [schema comprehension](SchemaComprehension.md) and map the data to an `IEnumerable` of user-defined objects.

Another mechanism that lets you inspect the intermediate data is the `GetColumn<T>` extension method. It lets you look at the contents of one column of your data in a form of an `IEnumerable`.

Here is all of this in action:

[Example file](../../test/data/adult.tiny.with-schema.txt):
```
Label	Workclass	education	marital-status
0	Private	11th	Never-married
0	Private	HS-grad	Married-civ-spouse
1	Local-gov	Assoc-acdm	Married-civ-spouse
1	Private	Some-college	Married-civ-spouse

```

```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Create the reader: define the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        // A boolean column depicting the 'target label'.
        IsOver50K: ctx.LoadBool(0),
        // Three text columns.
        Workclass: ctx.LoadText(1),
        Education: ctx.LoadText(2),
        MaritalStatus: ctx.LoadText(3)),
    hasHeader: true);

// Start creating our processing pipeline. For now, let's just concatenate all the text columns
// together into one.
var dataPipeline = reader.MakeNewEstimator()
    .Append(row => (
            row.IsOver50K,
            AllFeatures: row.Workclass.ConcatWith(row.Education, row.MaritalStatus)
        ));

// Let's verify that the data has been read correctly. 
// First, we read the data file.
var data = reader.Read(dataPath);

// Fit our data pipeline and transform data with it.
var transformedData = dataPipeline.Fit(data).Transform(data);

// 'transformedData' is a 'promise' of data. Let's actually read it.
var someRows = transformedData.AsDynamic
    // Convert to an enumerable of user-defined type. 
    .AsEnumerable<InspectedRow>(mlContext, reuseRowObject: false)
    // Take a couple values as an array.
    .Take(4).ToArray();

// Extract the 'AllFeatures' column.
// This will give the entire dataset: make sure to only take several row
// in case the dataset is huge.
var featureColumns = transformedData.GetColumn(r => r.AllFeatures)
    .Take(20).ToArray();

// The same extension method also applies to the dynamic-typed data, except you have to
// specify the column name and type:
var dynamicData = transformedData.AsDynamic;
var sameFeatureColumns = dynamicData.GetColumn<string[]>(mlContext, "AllFeatures")
    .Take(20).ToArray();
```

The above code assumes that we defined our `InspectedRow` class as follows:
```csharp
private class InspectedRow
{
    public bool IsOver50K;
    public string Workclass;
    public string Education;
    public string MaritalStatus;
    public string[] AllFeatures;
}
```

## How do I train a regression model?

Generally, in order to train any model in ML.NET, you will go through three steps:
1. Figure out how the training data gets into ML.NET in a form of an `IDataView`
2. Build the 'learning pipeline' as a sequence of elementary 'operators' (estimators).
3. Call `Fit` on the pipeline to obtain the trained model.

[Example file](../../test/data/generated_regression_dataset.csv):
```
feature_0;feature_1;feature_2;feature_3;feature_4;feature_5;feature_6;feature_7;feature_8;feature_9;feature_10;target
-2.75;0.77;-0.61;0.14;1.39;0.38;-0.53;-0.50;-2.13;-0.39;0.46;140.66
-0.61;-0.37;-0.12;0.55;-1.00;0.84;-0.02;1.30;-0.24;-0.50;-2.12;148.12
-0.85;-0.91;1.81;0.02;-0.78;-1.41;-1.09;-0.65;0.90;-0.37;-0.22;402.20
```

In the file above, the last column (12th) is label that we predict, and all the preceding ones are features.

```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Step one: read the data as an IDataView.
// First, we define the reader: specify the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        // We read the first 11 values as a single float vector.
        FeatureVector: ctx.LoadFloat(0, 10),
        // Separately, read the target variable.
        Target: ctx.LoadFloat(11)
    ),
    // The data file has header.
    hasHeader: true,
    // Default separator is tab, but we need a semicolon.
    separator: ';');


// Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
var trainData = reader.Read(trainDataPath);

// Step two: define the learning pipeline. 

// We 'start' the pipeline with the output of the reader.
var learningPipeline = reader.MakeNewEstimator()
    // Now we can add any 'training steps' to it. In our case we want to 'normalize' the data (rescale to be
    // between -1 and 1 for all examples), and then train the model.
    .Append(r => (
        // Retain the 'Target' column for evaluation purposes.
        r.Target,
        // We choose the SDCA regression trainer. Note that we normalize the 'FeatureVector' right here in
        // the the same call.
        Prediction: mlContext.Regression.Trainers.Sdca(label: r.Target, features: r.FeatureVector.Normalize())));

// Step three. Train the pipeline.
var model = learningPipeline.Fit(trainData);
```

## How do I verify the model quality?

This is the first question that arises after you train the model: how good it actually is?
For each of the machine learning tasks, there is a set of 'metrics' that can describe how good the model is: it could be log-loss or F1 score for classification, RMS or L1 loss for regression etc.

You can use the corresponding 'context' of the task to evaluate the model.

Assuming the example above was used to train the model, here's how you calculate the metrics.
```csharp
// Read the test dataset.
var testData = reader.Read(testDataPath);
// Calculate metrics of the model on the test data.
var metrics = mlContext.Regression.Evaluate(model.Transform(testData), label: r => r.Target, score: r => r.Prediction);
```

## How do I save and load the model?

Assuming that the model metrics look good to you, it's time to 'operationalize' the model. This is where ML.NET really shines: the `model` object you just built is ready for immediate consumption, it will apply all the same steps that it has 'learned' during training, and it can be persisted and reused in different environments.

Here's what you do to save the model to a file, and reload it (potentially in a different context).

```csharp
using (var stream = File.Create(modelPath))
{
    // Saving and loading happens to 'dynamic' models, so the static typing is lost in the process.
    mlContext.Model.Save(model.AsDynamic, stream);
}

// Potentially, the lines below can be in a different process altogether.

// When you load the model, it's a 'dynamic' transformer. 
ITransformer loadedModel;
using (var stream = File.OpenRead(modelPath))
    loadedModel = mlContext.Model.Load(stream);
```

## How do I use the model to make one prediction?

Since any ML.NET model is a transformer, you can of course use `model.Transform` to apply the model to the 'data view' and obtain predictions this way. 

A more typical case, though, is when there is no 'dataset' that we want to predict on, but instead we receive one example at a time. For instance, we run the model as part of the ASP.NET website, and we need to make a prediction for an incoming HTTP request.

For this case, ML.NET offers a convenient `PredictionFunction` component, that essentially runs one example at a time through the prediction pipeline. 

Here is the full example. Let's imagine that we have built a model for the famous Iris prediction dataset:

```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Step one: read the data as an IDataView.
// First, we define the reader: specify the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        // The four features of the Iris dataset.
        SepalLength: ctx.LoadFloat(0),
        SepalWidth: ctx.LoadFloat(1),
        PetalLength: ctx.LoadFloat(2),
        PetalWidth: ctx.LoadFloat(3),
        // Label: kind of iris.
        Label: ctx.LoadText(4)
    ),
    // Default separator is tab, but the dataset has comma.
    separator: ',');

// Retrieve the training data.
var trainData = reader.Read(irisDataPath);

// Build the training pipeline.
var learningPipeline = reader.MakeNewEstimator()
    .Append(r => (
        r.Label,
        // Concatenate all the features together into one column 'Features'.
        Features: r.SepalLength.ConcatWith(r.SepalWidth, r.PetalLength, r.PetalWidth)))
    .Append(r => (
        r.Label,
        // Train the multi-class SDCA model to predict the label using features.
        // Note that the label is a text, so it needs to be converted to key using 'ToKey' estimator.
        Predictions: mlContext.MulticlassClassification.Trainers.Sdca(r.Label.ToKey(), r.Features)))
    // Apply the inverse conversion from 'predictedLabel' key back to string value.
    // Note that the final output column is only one, and we didn't assign a name to it.
    // In this case, ML.NET auto-assigns the name 'Data' to the produced column.
    .Append(r => r.Predictions.predictedLabel.ToValue());

// Train the model.
var model = learningPipeline.Fit(trainData).AsDynamic;
```

Now, in order to use [schema comprehension](SchemaComprehension.md) for prediction, we define a pair of classes like following:
```csharp
private class IrisInput
{
    // Unfortunately, we still need the dummy 'Label' column to be present.
    [ColumnName("Label")]
    public string IgnoredLabel { get; set; }
    public float SepalLength { get; set; }
    public float SepalWidth { get; set; }
    public float PetalLength { get; set; }
    public float PetalWidth { get; set; }
}

private class IrisPrediction
{
    [ColumnName("Data")]
    public string PredictedClass { get; set; }
}
```

The prediction code now looks as follows:
```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Use the model for one-time prediction.
// Make the prediction function object. Note that, on average, this call takes around 200x longer
// than one prediction, so you might want to cache and reuse the prediction function, instead of
// creating one per prediction.
var predictionFunc = model.MakePredictionFunction<IrisInput, IrisPrediction>(mlContext);

// Obtain the prediction. Remember that 'Predict' is not reentrant. If you want to use multiple threads
// for simultaneous prediction, make sure each thread is using its own PredictionFunction.
var prediction = predictionFunc.Predict(new IrisInput
{
    SepalLength = 4.1f,
    SepalWidth = 0.1f,
    PetalLength = 3.2f,
    PetalWidth = 1.4f
});
```

## What if my training data is not in a text file?

The commonly demonstrated use case for ML.NET is when the training data resides somewhere on disk, and we use the `TextLoader` to read it.
However, in real-time training scenarios the training data can be elsewhere: in a bunch of SQL tables, extracted from log files, or even generated on the fly.

Here is how we can use [schema comprehension](SchemaComprehension.md) to bring an existing C# `IEnumerable` into ML.NET as a data view.

For the purpose of this example, we will assume that we build the customer churn prediction model, and we can extract the following features from our production system:
- Customer ID (ignored by the model)
- Whether the customer has churned (the target 'label')
- The 'demographic category' (one string, like 'young adult' etc.)
- The number of visits from the last 5 days.
```csharp
private class CustomerChurnInfo
{
    public string CustomerID { get; set; }
    public bool HasChurned { get; set; }
    public string DemographicCategory { get; set; }
    // Visits during last 5 days, latest to newest.
    [VectorType(5)]
    public float[] LastVisits { get; set; }
}
```

Given this information, here's how we turn this data into the ML.NET data view and train on it:
```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Step one: read the data as an IDataView.
// Let's assume that 'GetChurnData()' fetches and returns the training data from somewhere.
IEnumerable<CustomerChurnInfo> churnData = GetChurnInfo();

// Turn the data into the ML.NET data view.
// We can use CreateDataView or CreateStreamingDataView, depending on whether 'churnData' is an IList, 
// or merely an IEnumerable.
var trainData = mlContext.CreateStreamingDataView(churnData);

// Now note that 'trainData' is just an IDataView, so we face a choice here: either declare the static type
// and proceed in the statically typed fashion, or keep dynamic types and build a dynamic pipeline.
// We demonstrate both below.

// Build the learning pipeline. 
// In our case, we will one-hot encode the demographic category, and concatenate that with the number of visits.
// We apply our FastTree binary classifier to predict the 'HasChurned' label.

var dynamicLearningPipeline = mlContext.Transforms.Categorical.OneHotEncoding("DemographicCategory")
    .Append(new ColumnConcatenatingEstimator(mlContext, "Features", "DemographicCategory", "LastVisits"))
    .Append(mlContext.BinaryClassification.Trainers.FastTree("HasChurned", "Features", numTrees: 20));

var dynamicModel = dynamicLearningPipeline.Fit(trainData);

// Build the same learning pipeline, but statically typed.
// First, transition to the statically-typed data view.
var staticData = trainData.AssertStatic(mlContext, c => (
        HasChurned: c.Bool.Scalar,
        DemographicCategory: c.Text.Scalar,
        LastVisits: c.R4.Vector));

// Build the pipeline, same as the one above.
var staticLearningPipeline = staticData.MakeNewEstimator()
    .Append(r => (
        r.HasChurned,
        Features: r.DemographicCategory.OneHotEncoding().ConcatWith(r.LastVisits)))
    .Append(r => mlContext.BinaryClassification.Trainers.FastTree(r.HasChurned, r.Features, numTrees: 20));

var staticModel = staticLearningPipeline.Fit(staticData);

// Note that dynamicModel should be the same as staticModel.AsDynamic (give or take random variance from
// the training procedure).
```

## I want to look at my model's coefficients

Oftentimes, once a model is trained, we are also interested on 'what it has learned'. 

For example, if the linear model assigned zero weight to a feature that we consider important, it could indicate some problem with modeling. The weights of the linear model can also be used as a poor man's estimation of 'feature importance'.

In the static pipeline API, we provide a set of `onFit` delegates that allow introspection of the individual transformers as they are trained.

This is how we can extract the learned parameters out of the model that we trained:
```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Step one: read the data as an IDataView.
// First, we define the reader: specify the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        // The four features of the Iris dataset.
        SepalLength: ctx.LoadFloat(0),
        SepalWidth: ctx.LoadFloat(1),
        PetalLength: ctx.LoadFloat(2),
        PetalWidth: ctx.LoadFloat(3),
        // Label: kind of iris.
        Label: ctx.LoadText(4)
    ),
    // Default separator is tab, but the dataset has comma.
    separator: ',');

// Retrieve the training data.
var trainData = reader.Read(dataPath);

// This is the predictor ('weights collection') that we will train.
MulticlassLogisticRegressionPredictor predictor = null;
// And these are the normalizer scales that we will learn.
ImmutableArray<float> normScales;
// Build the training pipeline.
var learningPipeline = reader.MakeNewEstimator()
    .Append(r => (
        r.Label,
        // Concatenate all the features together into one column 'Features'.
        Features: r.SepalLength.ConcatWith(r.SepalWidth, r.PetalLength, r.PetalWidth)))
    .Append(r => (
        r.Label,
        // Normalize (rescale) the features to be between -1 and 1. 
        Features: r.Features.Normalize(
            // When the normalizer is trained, the below delegate is going to be called.
            // We use it to memorize the scales.
            onFit: (scales, offsets) => normScales = scales)))
    .Append(r => (
        r.Label,
        // Train the multi-class SDCA model to predict the label using features.
        // Note that the label is a text, so it needs to be converted to key using 'ToKey' estimator.
        Predictions: mlContext.MulticlassClassification.Trainers.Sdca(r.Label.ToKey(), r.Features,
            // When the model is trained, the below delegate is going to be called.
            // We use that to memorize the predictor object.
            onFit: p => predictor = p)));

// Train the model. During this call our 'onFit' delegate will be invoked,
// and our 'predictor' will be set.
var model = learningPipeline.Fit(trainData);

// Now we can use 'predictor' to look at the weights.
// 'weights' will be an array of weight vectors, one vector per class.
// Our problem has 3 classes, so numClasses will be 3, and weights will contain
// 3 vectors (of 4 values each).
VBuffer<float>[] weights = null;
predictor.GetWeights(ref weights, out int numClasses);

// Similarly we can also inspect the biases for the 3 classes.
var biases = predictor.GetBiases();

// Inspect the normalizer scales.
Console.WriteLine(string.Join(" ", normScales));
```

## What is normalization and why do I need to care?

In ML.NET we expose a number of [parametric and non-parametric algorithms](https://machinelearningmastery.com/parametric-and-nonparametric-machine-learning-algorithms/).

Typically, parametric learners hold certain assumptions about the training data, and if they are not met, the training is greatly hampered (or sometimes becomes completely impossible).

Most commonly, the assumptions are that
- All the features have values roughly on the same scale;
- Feature values are not too large, and not too small.

Violating the first assumption above can cause the learner to train a sub-optimal model (or even a completely useless one). Violating the second assumption can cause arithmetic error accumulation, which typically breaks the training process altogether.

As a general rule, *if you use a parametric learner, you need to make sure your training data is correctly scaled*. 

ML.NET offers several built-in scaling algorithms, or 'normalizers':
- MinMax normalizer: for each feature, we learn the minimum and maximum value of it, and then linearly rescale it so that the values fit between -1 and 1.
- MeanVar normalizer: for each feature, compute the mean and variance, and then linearly rescale it to zero-mean, unit-variance.
- CDF normalizer: for each feature, compute the mean and variance, and then replace each value `x` with `Cdf(x)`, where `Cdf` is the cumulative density function of normal distribution with these mean and variance. 
- Binning normalizer: discretize the feature value into `N` 'buckets', and then replace each value with the index of the bucket, divided by `N-1`.

These normalizers all have different properties and tradeoffs, but it's not *that* big of a deal if you use one over another. Just make sure you use a normalizer when training linear models or other parametric models. 

An important parameter of ML.NET normalizers is called `fixZero`. If `fixZero` is true, zero input is always mapped to zero output. This is very important when you handle sparse data: if we don't preserve zeroes, we will turn all sparse data into dense, which is usually a bad idea.

It is a good practice to include the normalizer directly in the ML.NET learning pipeline: this way you are sure that the normalization
- is only trained on the training data, and not on your test data,
- is correctly applied to all the new incoming data, without the need for extra pre-processing at prediction time.

Here's a snippet of code that demonstrates normalization in learning pipelines. It assumes the Iris dataset:
```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Define the reader: specify the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        // The four features of the Iris dataset will be grouped together as one Features column.
        Features: ctx.LoadFloat(0, 3),
        // Label: kind of iris.
        Label: ctx.LoadText(4)
    ),
    // Default separator is tab, but the dataset has comma.
    separator: ',');

// Read the training data.
var trainData = reader.Read(dataPath);

// Apply all kinds of standard ML.NET normalization to the raw features.
var pipeline = reader.MakeNewEstimator()
    .Append(r => (
        MinMaxNormalized: r.Features.Normalize(fixZero: true),
        MeanVarNormalized: r.Features.NormalizeByMeanVar(fixZero: false),
        CdfNormalized: r.Features.NormalizeByCumulativeDistribution(),
        BinNormalized: r.Features.NormalizeByBinning(maxBins: 256)
    ));

// Let's train our pipeline of normalizers, and then apply it to the same data.
var normalizedData = pipeline.Fit(trainData).Transform(trainData);

// Inspect one column of the resulting dataset.
var meanVarValues = normalizedData.GetColumn(r => r.MeanVarNormalized).ToArray();
```

## How do I train my model on categorical data?

Generally speaking, *all ML.NET learners expect the features as a float vector*. So, if some of your data is not natively a float, you will need to convert to floats. 

If our data contains 'categorical' features (think 'enum'), we need to 'featurize' them somehow. ML.NET offers several ways of converting categorical data to features:
- One-hot encoding
- Hash-based one-hot encoding
- Binary encoding (convert category index into a bit sequence and use bits as features)

If some of the categories are very high-cardinality (there's lots of different values, but only several are commonly occurring), a one-hot encoding can be wasteful. We can use count-based feature selection to trim down the number of slots that we encode.

Same with normalization, it's a good practice to include categorical featurization directly in the ML.NET learning pipeline: this way you are sure that the categorical transformation
- is only 'trained' on the training data, and not on your test data,
- is correctly applied to all the new incoming data, without the need for extra pre-processing at prediction time.

Below is an example of categorical handling for the [adult census dataset](../../test/data/adult.tiny.with-schema.txt):
```
Label	Workclass	education	marital-status	occupation	relationship	ethnicity	sex	native-country-region	age	fnlwgt	education-num	capital-gain	capital-loss	hours-per-week
0	Private	11th	Never-married	Machine-op-inspct	Own-child	Black	Male	United-States	25	226802	7	0	0	40
0	Private	HS-grad	Married-civ-spouse	Farming-fishing	Husband	White	Male	United-States	38	89814	9	0	0	50
1	Local-gov	Assoc-acdm	Married-civ-spouse	Protective-serv	Husband	White	Male	United-States	28	336951	12	0	0	40
1	Private	Some-college	Married-civ-spouse	Machine-op-inspct	Husband	Black	Male	United-States	44	160323	10	7688	0	40
```

```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Define the reader: specify the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        Label: ctx.LoadBool(0),
        // We will load all the categorical features into one vector column of size 8.
        CategoricalFeatures: ctx.LoadText(1, 8),
        // Similarly, load all numerical features into one vector of size 6.
        NumericalFeatures: ctx.LoadFloat(9, 14),
        // Let's also separately load the 'Workclass' column.
        Workclass: ctx.LoadText(1)
    ), hasHeader: true);

// Read the data.
var data = reader.Read(dataPath);

// Inspect the categorical columns to check that they are correctly read.
var catColumns = data.GetColumn(r => r.CategoricalFeatures).Take(10).ToArray();

// Build several alternative featurization pipelines.
var learningPipeline = reader.MakeNewEstimator()
    .Append(r => (
        r.Label,
        r.NumericalFeatures,
        // Convert each categorical feature into one-hot encoding independently.
        CategoricalOneHot: r.CategoricalFeatures.OneHotEncoding(outputKind: CategoricalStaticExtensions.OneHotVectorOutputKind.Ind),
        // Convert all categorical features into indices, and build a 'word bag' of these.
        CategoricalBag: r.CategoricalFeatures.OneHotEncoding(outputKind: CategoricalStaticExtensions.OneHotVectorOutputKind.Bag),
        // One-hot encode the workclass column, then drop all the categories that have fewer than 10 instances in the train set.
        WorkclassOneHotTrimmed: r.Workclass.OneHotEncoding().SelectFeaturesBasedOnCount(count: 10)
    ));

// Let's train our pipeline, and then apply it to the same data.
var transformedData = learningPipeline.Fit(data).Transform(data);

// Inspect some columns of the resulting dataset.
var categoricalBags = transformedData.GetColumn(x => x.CategoricalBag).Take(10).ToArray();
var workclasses = transformedData.GetColumn(x => x.WorkclassOneHotTrimmed).Take(10).ToArray();

// Of course, if we want to train the model, we will need to compose a single float vector of all the features.
// Here's how we could do this:

var fullLearningPipeline = learningPipeline
    .Append(r => (
        r.Label,
        // Concatenate two of the 3 categorical pipelines, and the numeric features.
        Features: r.NumericalFeatures.ConcatWith(r.CategoricalBag, r.WorkclassOneHotTrimmed)))
    // Now we're ready to train. We chose our FastTree trainer for this classification task.
    .Append(r => mlContext.BinaryClassification.Trainers.FastTree(r.Label, r.Features, numTrees: 50));

// Train the model.
var model = fullLearningPipeline.Fit(data);
```

## How do I train my model on textual data?

Generally speaking, *all ML.NET learners expect the features as a float vector*. So, if some of your data is not natively a float, you will need to convert to floats. 

If we want to learn on textual data, we need to 'extract features' out of the texts. There is an entire research area of NLP (Natural Language Processing) that handles this. In ML.NET we offer some basic mechanisms of text feature extraction:
- Text normalization (removing punctuation, diacritics, switching to lowercase etc.)
- Separator-based tokenization.
- Stopword removal.
- Ngram and skip-gram extraction.
- TF-IDF rescaling.
- Bag of words conversion.

ML.NET offers a "one-stop shop" operation called `TextFeaturizer`, that runs a combination of above steps as one big 'text featurization'. We have tested it extensively on text datasets, and we're confident that it performs reasonably well without the need to deep-dive into the operations. 

However, we also offer a selection of elementary operations that let you customize your NLP processing. Here's the example below where we use them.

Wikipedia detox dataset:
```
Sentiment   SentimentText
1	Stop trolling, zapatancas, calling me a liar merely demonstartes that you arer Zapatancas. You may choose to chase every legitimate editor from this site and ignore me but I am an editor with a record that isnt 99% trolling and therefore my wishes are not to be completely ignored by a sockpuppet like yourself. The consensus is overwhelmingly against you and your trollin g lover Zapatancas,  
1	::::: Why are you threatening me? I'm not being disruptive, its you who is being disruptive.   
0	" *::Your POV and propaganda pushing is dully noted. However listing interesting facts in a netral and unacusitory tone is not POV. You seem to be confusing Censorship with POV monitoring. I see nothing POV expressed in the listing of intersting facts. If you want to contribute more facts or edit wording of the cited fact to make them sound more netral then go ahead. No need to CENSOR interesting factual information. "
0	::::::::This is a gross exaggeration. Nobody is setting a kangaroo court. There was a simple addition concerning the airline. It is the only one disputed here.   
```

```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Define the reader: specify the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        IsToxic: ctx.LoadBool(0),
        Message: ctx.LoadText(1)
    ), hasHeader: true);

// Read the data.
var data = reader.Read(dataPath);

// Inspect the message texts that are read from the file.
var messageTexts = data.GetColumn(x => x.Message).Take(20).ToArray();

// Apply various kinds of text operations supported by ML.NET.
var learningPipeline = reader.MakeNewEstimator()
    .Append(r => (
        // One-stop shop to run the full text featurization.
        TextFeatures: r.Message.FeaturizeText(),

        // NLP pipeline 1: bag of words.
        BagOfWords: r.Message.NormalizeText().ToBagofWords(),

        // NLP pipeline 2: bag of bigrams, using hashes instead of dictionary indices.
        BagOfBigrams: r.Message.NormalizeText().ToBagofHashedWords(ngramLength: 2, allLengths: false),

        // NLP pipeline 3: bag of tri-character sequences with TF-IDF weighting.
        BagOfTrichar: r.Message.TokenizeIntoCharacters().ToNgrams(ngramLength: 3, weighting: NgramTransform.WeightingCriteria.TfIdf),

        // NLP pipeline 4: word embeddings.
        Embeddings: r.Message.NormalizeText().TokenizeText().WordEmbeddings(WordEmbeddingsTransform.PretrainedModelKind.GloVeTwitter25D)
    ));

// Let's train our pipeline, and then apply it to the same data.
// Note that even on a small dataset of 70KB the pipeline above can take up to a minute to completely train.
var transformedData = learningPipeline.Fit(data).Transform(data);

// Inspect some columns of the resulting dataset.
var embeddings = transformedData.GetColumn(x => x.Embeddings).Take(10).ToArray();
var unigrams = transformedData.GetColumn(x => x.BagOfWords).Take(10).ToArray();
```

## How do I train using cross-validation?

[Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) is a useful technique for ML applications. It helps estimate the variance of the model quality from one run to another and also eliminates the need to extract a separate test set for evaluation.

There are a couple pitfalls that await us when we implement our own cross-validation. Essentially, if we are not careful, we may introduce label leakage in the process, so our metrics could become over-inflated.

- It is tempting to apply the same pre-processing to the entire data, and then just cross-validate the final training of the model. If we do this for data-dependent, 'trainable' pre-processing (like text featurization, categorical handling and normalization/rescaling), we cause these processing steps to 'train' on the union of train subset and test subset, thus causing label leakage. The correct way is to apply pre-processing independently for each 'fold' of the cross-validation.
- In many cases there is a natural 'grouping' of the data that needs to be respected. For example, if we are solving a click prediction problem, it's a good idea to group all examples pertaining to one URL to appear in one-fold of the data. If they end up separated, we can introduce label leakage.

ML.NET guards us against both these pitfalls: it will automatically apply the featurization correctly (as long as all of the preprocessing resides in one learning pipeline), and we can use the 'stratification column' concept to make sure that related examples don't get separated.

Here's an example of training on Iris dataset using randomized 90/10 train-test split, as well as a 5-fold cross-validation:
```csharp
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Step one: read the data as an IDataView.
// First, we define the reader: specify the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        // The four features of the Iris dataset.
        SepalLength: ctx.LoadFloat(0),
        SepalWidth: ctx.LoadFloat(1),
        PetalLength: ctx.LoadFloat(2),
        PetalWidth: ctx.LoadFloat(3),
        // Label: kind of iris.
        Label: ctx.LoadText(4)
    ),
    // Default separator is tab, but the dataset has comma.
    separator: ',');

// Read the data.
var data = reader.Read(dataPath);

// Build the training pipeline.
var learningPipeline = reader.MakeNewEstimator()
    .Append(r => (
        // Convert string label to a key.
        Label: r.Label.ToKey(),
        // Concatenate all the features together into one column 'Features'.
        Features: r.SepalLength.ConcatWith(r.SepalWidth, r.PetalLength, r.PetalWidth)))
    .Append(r => (
        r.Label,
        // Train the multi-class SDCA model to predict the label using features.
        Predictions: mlContext.MulticlassClassification.Trainers.Sdca(r.Label, r.Features)));

// Split the data 90:10 into train and test sets, train and evaluate.
var (trainData, testData) = mlContext.MulticlassClassification.TrainTestSplit(data, testFraction: 0.1);

// Train the model.
var model = learningPipeline.Fit(trainData);
// Compute quality metrics on the test set.
var metrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testData), r => r.Label, r => r.Predictions);
Console.WriteLine(metrics.AccuracyMicro);

// Now run the 5-fold cross-validation experiment, using the same pipeline.
var cvResults = mlContext.MulticlassClassification.CrossValidate(data, learningPipeline, r => r.Label, numFolds: 5);

// The results object is an array of 5 elements. For each of the 5 folds, we have metrics, model and scored test data.
// Let's compute the average micro-accuracy.
var microAccuracies = cvResults.Select(r => r.metrics.AccuracyMicro);
Console.WriteLine(microAccuracies.Average());
```

## Can I mix and match static and dynamic pipelines?

Yes, we can have both of them in our codebase. The static pipelines are just a statically-typed way to build dynamic pipelines.

Namely, any statically typed component (`DataView<T>`, `Transformer<T>`, `Estimator<T>`) has its dynamic counterpart as an `AsDynamic` property.

Transitioning from dynamic to static types is more costly: we have to formally declare what is the 'schema shape'. Or, in case of estimators and transformers, what is the input and output schema shape. 

We can do this via `AssertStatic<T>` extensions, as demonstrated in the following example, where we mix and match static and dynamic pipelines.
```c#
// Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
// as a catalog of available operations and as the source of randomness.
var mlContext = new MLContext();

// Read the data as an IDataView.
// First, we define the reader: specify the data columns and where to find them in the text file.
var reader = mlContext.Data.TextReader(ctx => (
        // The four features of the Iris dataset.
        SepalLength: ctx.LoadFloat(0),
        SepalWidth: ctx.LoadFloat(1),
        PetalLength: ctx.LoadFloat(2),
        PetalWidth: ctx.LoadFloat(3),
        // Label: kind of iris.
        Label: ctx.LoadText(4)
    ),
    // Default separator is tab, but the dataset has comma.
    separator: ',');

// Read the data.
var data = reader.Read(dataPath);

// Build the pre-processing pipeline.
var learningPipeline = reader.MakeNewEstimator()
    .Append(r => (
        // Convert string label to a key.
        Label: r.Label.ToKey(),
        // Concatenate all the features together into one column 'Features'.
        Features: r.SepalLength.ConcatWith(r.SepalWidth, r.PetalLength, r.PetalWidth)));

// Now, at the time of writing, there is no static pipeline for OVA (one-versus-all). So, let's
// append the OVA learner to the dynamic pipeline.
IEstimator<ITransformer> dynamicPipe = learningPipeline.AsDynamic;

// Create a binary classification trainer.
var binaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron();

// Append the OVA learner to the pipeline.
dynamicPipe = dynamicPipe.Append(new Ova(mlContext, binaryTrainer));

// At this point, we have a choice. We could continue working with the dynamically-typed pipeline, and
// ultimately call dynamicPipe.Fit(data.AsDynamic) to get the model, or we could go back into the static world.
// Here's how we go back to the static pipeline:
var staticFinalPipe = dynamicPipe.AssertStatic(mlContext,
        // Declare the shape of the input. As you can see, it's identical to the shape of the reader:
        // four float features and a string label.
        c => (
            SepalLength: c.R4.Scalar,
            SepalWidth: c.R4.Scalar,
            PetalLength: c.R4.Scalar,
            PetalWidth: c.R4.Scalar,
            Label: c.Text.Scalar),
        // Declare the shape of the output (or a relevant subset of it).
        // In our case, we care only about the predicted label column (a key type), and scores (vector of floats).
        c => (
            Score: c.R4.Vector,
            // Predicted label is a key backed by uint, with text values (since original labels are text).
            PredictedLabel: c.KeyU4.TextValues.Scalar))
    // Convert the predicted label from key back to the original string value.
    .Append(r => r.PredictedLabel.ToValue());

// Train the model in a statically typed way.
var model = staticFinalPipe.Fit(data);

// And here is how we could've stayed in the dynamic pipeline and train that way.
dynamicPipe = dynamicPipe.Append(new KeyToValueEstimator(mlContext, "PredictedLabel"));
var dynamicModel = dynamicPipe.Fit(data.AsDynamic);

// Now 'dynamicModel', and 'model.AsDynamic' are equivalent.
```
