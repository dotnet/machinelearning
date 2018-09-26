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

## How do I load data from a text file?

`TextLoader` is used to load data from text files. You will need to specify what are the data columns, what are their types, and where to find them in the text file. 

Note that it's perfectly acceptable to read only some columns of a file, or read the same column multiple times.

Example file (https://github.com/dotnet/machinelearning/blob/master/test/data/adult.tiny.with-schema.txt):
```
Label	Workclass	education	marital-status
0	Private	11th	Never-married
0	Private	HS-grad	Married-civ-spouse
1	Local-gov	Assoc-acdm	Married-civ-spouse
1	Private	Some-college	Married-civ-spouse

```

This is how you can read this data:
```c#
// Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
// as well as the source of randomness.
var env = new LocalEnvironment();

// Create the reader: define the data columns and where to find them in the text file.
var reader = TextLoader.CreateReader(env, ctx => (
        // A boolean column depicting the 'target label'.
        IsOver50K: ctx.LoadBool(0),
        // Three text columns.
        Workclass: ctx.LoadText(1),
        Education: ctx.LoadText(2),
        MaritalStatus: ctx.LoadText(3)),
    hasHeader: true);

// Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
var data = reader.Read(new MultiFileSource(dataPath));
```

If the schema of the data is not known at compile time, or too cumbersome, you can revert to the dynamically-typed API: 
```c#
// Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
// as well as the source of randomness.
var env = new LocalEnvironment();

// Create the reader: define the data columns and where to find them in the text file.
var reader = new TextLoader(env, new TextLoader.Arguments
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
var data = reader.Read(new MultiFileSource(dataPath));
```

## How do I load data with many columns from a CSV?
`TextLoader` is used to load data from text files. You will need to specify what are the data columns, what are their types, and where to find them in the text file. 

When the input file contains many columns of the same type, always intended to be used together, we recommend reading them as a *vector column* from the very start: this way the schema of the data is cleaner, and we don't incur unnecessary performance costs.

Example file (https://github.com/dotnet/machinelearning/blob/master/test/data/generated_regression_dataset.csv):
```
-2.75,0.77,-0.61,0.14,1.39,0.38,-0.53,-0.50,-2.13,-0.39,0.46,140.66
-0.61,-0.37,-0.12,0.55,-1.00,0.84,-0.02,1.30,-0.24,-0.50,-2.12,148.12
-0.85,-0.91,1.81,0.02,-0.78,-1.41,-1.09,-0.65,0.90,-0.37,-0.22,402.20
0.28,1.05,-0.24,0.30,-0.99,0.19,0.32,-0.95,-1.19,-0.63,0.75,443.51
```

Reading this file using `TextLoader`:
```c#
// Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
// as well as the source of randomness.
var env = new LocalEnvironment();

// Create the reader: define the data columns and where to find them in the text file.
var reader = TextLoader.CreateReader(env, ctx => (
        // We read the first 11 values as a single float vector.
        FeatureVector: ctx.LoadFloat(0, 10),
        // Separately, read the target variable.
        Target: ctx.LoadFloat(11)
    ),
    // Default separator is tab, but we need a comma.
    separator: ',');


// Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
var data = reader.Read(new MultiFileSource(dataPath));
```


If the schema of the data is not known at compile time, or too cumbersome, you can revert to the dynamically-typed API: 
```c#
// Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
// as well as the source of randomness.
var env = new LocalEnvironment();

// Create the reader: define the data columns and where to find them in the text file.
var reader = new TextLoader(env, new TextLoader.Arguments
{
    Column = new[] {
	    // We read the first 10 values as a single float vector.
        new TextLoader.Column("FeatureVector", DataKind.R4, new[] {new TextLoader.Range(0, 9)}),
        // Separately, read the target variable.
        new TextLoader.Column("Target", DataKind.R4, 10)
    },
    // Default separator is tab, but we need a comma.
    Separator = ","
});

// Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
var data = reader.Read(new MultiFileSource(dataPath));
```

## How do I look at the intermediate data?

Oftentimes, when we construct the experiment, we want to make sure that the data processing 'up to a certain moment' produces the results that we want. With ML.NET it is not very easy to do: since all ML.NET operations are lazy, the objects we construct are just 'promises' of data.

We will need to create the cursor and scan the data to obtain the actual values. One way to do this is to use [schema comprehension](SchemaComprehension.md) and map the data to an `IEnumerable` of user-defined objects.

Example file (https://github.com/dotnet/machinelearning/blob/master/test/data/adult.tiny.with-schema.txt):
```
Label	Workclass	education	marital-status
0	Private	11th	Never-married
0	Private	HS-grad	Married-civ-spouse
1	Local-gov	Assoc-acdm	Married-civ-spouse
1	Private	Some-college	Married-civ-spouse

```

```c#
// Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
// as well as the source of randomness.
var env = new LocalEnvironment();

// Create the reader: define the data columns and where to find them in the text file.
var reader = TextLoader.CreateReader(env, ctx => (
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
    .Append(row =>
        (
            row.IsOver50K,
            AllFeatures: row.Workclass.ConcatWith(row.Education, row.MaritalStatus)
        ));

// Let's verify that the data has been read correctly. 
// First, we read the data file.
var data = reader.Read(new MultiFileSource(dataPath));

// Fit our data pipeline and transform data with it.
var transformedData = dataPipeline.Fit(data).Transform(data);

// 'transformedData' is a 'promise' of data. Let's actually read it.
var someRows = transformedData.AsDynamic
    // Convert to an enumerable of user-defined type. 
    .AsEnumerable<InspectedRow>(env, reuseRowObject: false)
    // Take a couple values as an array.
    .Take(4).ToArray();

// Now we can inspect 'someRows' to see if the data has been read and transformed correctly.
```

The above code assumes that we defined our `InspectedRow` class as follows:
```c#
private class InspectedRow
{
    public bool IsOver50K;
    public string Workclass;
    public string Education;
    public string MaritalStatus;
    public string[] AllFeatures;
}
```

Another mechanism that lets you inspect the intermediate data is the `GetColumn<T>` extension method. It lets you look at the contents of one column of your data in a form of an `IEnumerable`. This code works for the same data pipeline as above:
```c#
// Extract the 'AllFeatures' column.
// This will give the entire dataset: make sure to only take several row
// in case the dataset is huge.
var featureColumns = transformedData.GetColumn(r => r.AllFeatures)
    .Take(20).ToArray();

// The same extension method also applies to the dynamic-typed data, except you have to
// specify the column name and type:
var dynamicData = transformedData.AsDynamic;
var sameFeatureColumns = dynamicData.GetColumn<string[]>(env, "AllFeatures")
    .Take(20).ToArray();
```

## How do I train a regression model?

Generally, in order to train any model in ML.NET, you will go through three steps:
1. Figure out how the training data gets into ML.NET in a form of an `IDataView`
2. Build the 'learning pipeline' as a sequence of elementary 'operators' (estimators).
3. Call `Fit` on the pipeline to obtain the trained model.

Example file (https://github.com/dotnet/machinelearning/blob/master/test/data/generated_regression_dataset.csv):
```
-2.75,0.77,-0.61,0.14,1.39,0.38,-0.53,-0.50,-2.13,-0.39,0.46,140.66
-0.61,-0.37,-0.12,0.55,-1.00,0.84,-0.02,1.30,-0.24,-0.50,-2.12,148.12
-0.85,-0.91,1.81,0.02,-0.78,-1.41,-1.09,-0.65,0.90,-0.37,-0.22,402.20
0.28,1.05,-0.24,0.30,-0.99,0.19,0.32,-0.95,-1.19,-0.63,0.75,443.51
```
