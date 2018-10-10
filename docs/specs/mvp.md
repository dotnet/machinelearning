# MVP Spec

This document is meant to capture the key requirements and rough drafts of the
API surface we need to get ready by Build, which is the conference we're going
to release an initial preview of our API shape.

That means the marker **Currently out of scope** really applies to deliverables
for Build. Post Build, we'll update this document to reflect upcoming release
timelines.

## Prediction

### User Experience

Mikayla works for a real estate company. They have a database of houses that
might go up for sale. They'd like to use machine learning to predict the house
prices. Since Mikayla is new to machine learning, she decides to use the ML.NET
UI to create a model for this. After she create the model, the UI offers to
generate some C# code that she can incorporate into her ASP.NET Core web site.

### Code

The generated code is meant to be easily copy & pasted into a developer's
application or library. The `Main` method is likely being adapted to match the
developer's application but the intention is to give them something to get
started with. It also illustrates how the APIs are meant to be used without
having to consult the documentation.

**Currently out of scope**. Later on, we might decide to integrate the UI and
code generation directly into Visual Studio, in which case we'll likely have to
separate code we expect the developer to use as-is from sample code.

```csharp
// Here is how you'd use these types from your application:

/*
class Program
{
    static async Task Main(string[] args)
    {
        var path = @"C:\users\mikayla\documents\houseprices.zip";
        var model = await PredictionModel.ReadAsync<HousePriceData, HousePricePrediction>(path);
        var house = new HousePriceData
        {
            // TODO: initialize your data type you'd like to get predictions for
        };
        HousePricePrediction prediction = model.Predict(house);
    }
}
*/

/// <summary>
/// This class represents your input data.
/// </summary>
public partial class HousePriceData
{
    [Column(ordinal: "0")]
    public string Id;
    [Column(ordinal: "1")]
    public string Date;
    [Column(ordinal: "2")]
    public float Bedrooms;
    [Column(ordinal: "3")]
    public float Bathrooms;
    [Column(ordinal: "4")]
    public float SqftLiving;
    [Column(ordinal: "5")]
    public float SqftLot;
    [Column(ordinal: "6")]
    public float Floors;
    [Column(ordinal: "7")]
    public float Waterfront;
    [Column(ordinal: "8")]
    public float View;
    [Column(ordinal: "9")]
    public float Condition;
    [Column(ordinal: "10")]
    public float Grade;
    [Column(ordinal: "11")]
    public float SqftAbove;
    [Column(ordinal: "12")]
    public float SqftBasement;
    [Column(ordinal: "13")]
    public float YearBuilt;
    [Column(ordinal: "14")]
    public float YearRenovated;
    [Column(ordinal: "15")]
    public float ZipCode;
    [Column(ordinal: "16")]
    public float Lat;
    [Column(ordinal: "17")]
    public float Long;
    [Column(ordinal: "18")]
    public float SqftLiving15;
    [Column(ordinal: "19")]
    public float SqftLot15;
    [Column(name: "Label", ordinal: "19")]
    public float Price;
}

/// <summary>
/// This class represents the result of a prediction for your input. The
/// predicted value is marked with <c>[ColumnName("Score")]</c>.
/// </summary>
public partial class HousePricePrediction
{
    [Column("Score")]
    public float Price;
}
```

### Design

* We like the term *Model* to represent the persisted graph with the trained
  state. However, *Model* is too generic of a name, so we need a prefix. We
  penciled in `PredictionModel` as that's the only useful API on this type right
  now. Once we looked at training and extended the API surface we can revisit
  that decision.

* The model should be immutable. We'll offer a conversion API to go from the
  model to the building block for training, e.g. `ToPipeline()`.

* The persisted `model.zip` has no notion of .NET types. It only has schema
  information. Strong typing requires the developer to specify the types when
  loading the model but it makes serialization/deserialization simpler and more
  sustainable: the schema is matched by duck typing.

* TLC supports the packing of multiple fields into an array instead of accessing
  the values as individual fields. They did this for performance optimization.
    - We believe this can be a more advanced scenario for the developer who is
      willing to massage the data type more to get better performance.
    - **Currently out of scope**

* We made the sample code a comment rather than code. It's very likely that
  developers will need to adapt it to their code while there are still iterating
  on the model with the GUI. If we always generate it, they would have to delete
  the code every single time they regenerated the source code.

### API Shape

```csharp
namespace Microsoft.MachineLearning
{
    public class PredictionModel
    {
        public static PredictionModel ReadAsync(string path);
        public static PredictionModel ReadAsync(Stream stream);
        public static PredictionModel<TInput, TOutput> ReadAsync<TInput, TOutput>(string path);
        public static PredictionModel<TInput, TOutput> ReadAsync<TInput, TOutput>(Stream stream);

        public IDataView Predict(IDataView data);
        public Task WriteAsync(string path);
        public Task WriteAsync(Stream stream);
    }
    public class PredictionModel<TInput, TOutput> : PredictionModel
    {
        public IEnumerable<TOuput> Predict(IEnumerable<TInput> data);
        public TOuput Predict(TInput data);
    }
}
```

### Open questions

* We currently use properties while the core components (`PredictionEngine<>`
  and `BatchPredictionEngine<>`) only supports fields. Can it support properties
  or should we change the spec?
    - Properties are preferred because it's what .NET developers are used to.
    - It also gives them a way to encapsulate logic that the TLC engine doesn't
      need to understand (like lazily loading data or syncing derived values).
* Should we support binding to immutable data types?
    - The engine would either have to bind to fields or understand typical
      conventions of immutable types (using withers).
    - **Currently out of scope**

## Training

### User Experience

Mikayla is using the ML.NET UI for a while. It helped her to get familiar with
the concepts. She would now like to train using code, as this would enable her
to automate this task. As Mikayla is writing the code, she notices that it can
be challenging to understand what columns are available at a given point in the
training pipeline. She's glad that she can simply set a breakpoint, step over
the items and explore the schema as well as the actual data.

Using edit-and-continue she's able to get the pipeline up and running. This
approach is similar to what she has seen in a Python-focused Pluralsight course
on machine learning, but more familiar to her as she's an experienced C#
developer.

### Code

Similar to prediction, the ML.NET UI might be able to allow for the training
code to be generated as well. However, we've optimized the API shape to be
written by hand, which includes the ability to easily debug the training
pipeline:

```csharp

var path = @"C:\users\mikayla\documents\houses.csv";
var options = CsvOptions.CreateFrom<HousePriceData>(separator: '\t', header: true);

var p = new LearningPipeline<HousePriceData, HousePricePrediction>();

p.Add(new TextLoader<HousePriceData>(path, options));
p.Add(new ConcatColumns(
    "NumericalFeatures",
    "SqftLiving", "SqftLot", "SqftAbove", "SqftBasement", "Lat", "Long",
    "SqftLiving15", "SqftLot15"
));
p.Add(new ConcatColumns(
    "CategoryFeatures",
    "Bedrooms", "Bathrooms", "Floors", "Waterfront", "View", "Condition",
    "Grade", "YearBuilt", "YearRenovated", "Zipcode"
));
p.Add(new CatTransformDict(
    "CategoryFeatures"
));
p.Add(new ConcatColumns(
    "Features",
    "NumericalFeatures", "CategoryFeatures"
));
p.Add(new TrainRegression());

PredictionModel<HousePriceData, HousePricePrediction> model = await p.TrainAsync();

await model.WriteAsync(@"C:\users\mikayla\documents\houseModel.zip");
```

### Debugging

The expectation is that the debugging will be made possible by inspecting the
pipeline object. Consider these lines of code:

```csharp
var p = new Pipeline();
/* 1 */ p.Add(new CsvLoader(trainFilePath, options));
/* 2 */ p.Add(new Categorical(p, "CategoricalFeatures"));
/* 3 */ p.Add(new ColumnConcatenator(p, output: "Features",
              "NumericalFeatures", "CategoricalFeatures"));
/* 4 */ p.Add(new SdcaRegressor(p,
              learningRate: 10,
              label: "price", "zipcode", "waterfront", "bedrooms"));
```

When the developer sets a breakpoint on statement (1), the pipeline is empty.
Thus no data or schema information will be present.

Subsequent breakpoints will allow the developer to inspect the pipeline with the
elements that were added so far. For instance at point (2) the CSV schema
information and data will be present. At point (3) and (4) the categorical and
column concatenation will be available.

It's important to point out that outside the debugger no execution of the
pipeline happens until `PredictionModel.CreateAsync(p)` is called. The execution
during debugging will be achieved by [debugger type proxies].

[debugger type proxies]: https://docs.microsoft.com/en-us/visualstudio/debugger/using-debuggertypeproxy-attribute

### Design

* Our graph execution model is declarative, i.e. the developer isn't
  mechanically pulling or pushing data throught the system. Instead, the
  developer creates a graph and passes it to a graph runner, which is currently
  represented by `PredictionModel.CreateAsync`. The construction of the graph
  itself is imperative, which creates locals for individuals nodes which means
  we have a chance to make it debuggable.
* We need to be able to debug the flow, which means the developer needs to
  be able to view the schema as well as the top N rows of the actual data.
  - Add a `take(100)`` transform after loader or add a parameter to the loader
  - Train a model
  - Predict with a model using some dataset, such as the train dataset
* We need to avoid eager computation/full loading of data, i.e. streaming of
  large amounts must be possible without extra work for the developer.
* The `LearningPipeline` is mutable as immutability doesn't buy us much:
  - No thread safety issues as `TrainAsync()` effectively snapshots the graph.
  - The fluent syntax is mostly useful to avoid saying `p.`` which C# collection
    initializer syntax solves as well
  - Immutability is heavy handed as we'd have to make sure that all nodes are
    too
  - C# developers primarily think in terms mutable APIs
* We believe we want a three-line automatic API where customers only need to
  declare the schema, specify the column they want to predict (AKA the label)
  and then they will get a generated pipeline they can make predictions with
  without knowing anything about machine learning.
    - **Currently out of scope**
* We believe we also want a `Pipeline`-style API that allows us to add items
  and have the type do the wiring so that code can be moved more freely without
  having to perform error prone rewiring.
    - **Currently out of scope**

### API Shape

This captures the abstract base type we expect all nodes, such as `CsvLoader`,
`Categorical`, `ColumnConcatenator`, and `SdcaRegressor` to derive from:

```csharp
namespace Microsoft.MachineLearning
{  
    public abstract class PipelineItem
    {
        public PipelineItem Previous { get; set; }
        // We need to expose a debugger viewer, which might be an attribute
        // public IDataView Output { get; }
    }
}
```

The pipeline items will be spread across multiple namespaces:

* `Microsoft.MachineLearning`
* `Microsoft.MachineLearning.Pipeline`
* `Microsoft.MachineLearning.Pipeline.Loaders`
* `Microsoft.MachineLearning.Pipeline.Transforms`
* `Microsoft.MachineLearning.Pipeline.Learners`
* `Microsoft.MachineLearning.Pipeline.Evaluators`

All nodes will have coverloads for the constructor that accept a previous
as well as ones that do not so that they can be wired later:

```csharp
namespace Microsoft.MachineLearning.Transforms
{  
    public class Categorical : PipelineItem
    {
        public Categorical(PipelineItem previous, params string[] columns);
        public Categorical(params string[] columns);
    }
    public class ColumnConcatenator : PipelineItem
    {
        public ColumnConcatenator(PipelineItem previous, string outputColumn, params string[] columns);
        public ColumnConcatenator(string outputColumn, params string[] columns);
    }
    // ...
}
```

This captures the CSV reader APIs:

```csharp
namespace Microsoft.MachineLearning
{
    public enum DataType
    {
        Boolean,
        Int32,
        Single,
        Double,
        String        
    }
    public class CsvColumn
    {
        public CsvColumn(string name, DataType dataType, int ordinal);
        public string Name { get; }
        public DataType DataType { get; }
        public int Ordinal { get; }
    }
    public class CsvOptions
    {
        public static CsvOptions CreateFrom(char separator = ',', bool hasHeader = true, Type rowType);
        public static CsvOptions CreateFrom<T>(char separator = ',', bool hasHeader = true);

        public CsvOptions();
        public char Separator { get; set; }
        public bool HasHeader { get; set; }
        public CsvColumnCollection Columns { get; }
    }
    public class CsvColumnCollection : Collection<CsvColumn>
    {
        public CsvColumn Add(string name, DataType dataType);
        public CsvColumn Add(string name, DataType dataType, int ordinal);
    }
}
```

## Evaluation

### User Experience

Mikayla has trained her model and would like to see how well her model is able
to predict prices based on known sample data. Using the evaluation APIs she's
able to create a .NET Core console application that allows her to automate
assessing the quality of the current model by printing relevant statistical
measures, such as RMS, R-square, and others.

### Code

```csharp
var model = pipeline.Train<HousePriceData, HousePricePrediction>();
var testData = new TextLoader<HousePriceData>(TestDataPath, useHeader: true, separator: ",");
var evaluator = new RegressionEvaluator();
RegressionMetrics metrics = evaluator.Evaluate(model, testData);

// RMS is the main metric for judging how good the regression prediction
// worked. It stands for "root mean square loss".
Console.WriteLine("Rms=" + metrics.Rms);
```
