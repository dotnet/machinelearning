# Schema comprehension in ML.NET

This document describes in detail the under-the-hood mechanism that ML.NET uses to automate the creation of `IDataView` schema, with the goal to make it as convenient to the end user as possible, while not incurring extra computational costs.

## Introduction

Every dataset in ML.NET is an `IDataView`, which is, for the purposes of this document, a collection of rows that share the same columns. The set of columns, their names, types and other metadata is known as the *schema* of the `IDataView`, and it's represented as an `ISchema` object.

Before any new data enters ML.NET, the user needs to somehow define how the schema of the data will look like.
To do this, the following questions need to be answered:
- What are the column names?
- What are their types?
- What other metadata is associated with the columns?

These items above are very similar to the definition of fields in a C# class: names and types of columns correspond to names and types of fields, and metadata can correspond to field attributes. 
Because of this similarity, ML.NET offers a common convenient mechanism for creating a schema: it is done via defining a C# class.

For example, the below class definition can be used to define a data view with 5 float columns:
```(csharp)
public class IrisData
{
    public float Label;
    public float SepalLength;
    public float SepalWidth;
    public float PetalLength;
    public float PetalWidth;
}
```

## Using schema comprehension to make a data view and to read a data view

The first obvious benefit of schema comprehension is that we can now create `IDataView`s out of in-memory enumerables of user-defined 'data types', without having to define the schema.
It works in the other direction too: you can take an `IDataView`, and read it as an `IEnumerable` of user-defined 'data type' (which will fail if the user-provided schema does not match the real schema).

Let's see how we can create a new `IDataView` out of an in-memory array, run some operations on it, and then read it back into the array.

```(csharp)
public class IrisData
{
    public float Label;
    public float SepalLength;
    public float SepalWidth;
    public float PetalLength;
    public float PetalWidth;
}

public class IrisVectorData
{
    public float Label;
    public float[] Features;
}

static void Main(string[] args)
{
    // Here's a data array that we want to work on.
    var dataArray = new[] {
        new IrisData{Label=1, PetalLength=1, SepalLength=1, PetalWidth=1, SepalWidth=1},
        new IrisData{Label=0, PetalLength=2, SepalLength=2, PetalWidth=2, SepalWidth=2}
    };

    // Create the ML.NET environment.
    var env = new Microsoft.ML.Runtime.Data.TlcEnvironment();

    // Create the data view.
    // This method will use the definition of IrisData to understand what columns there are in the 
    // data view.
    var dv = env.CreateDataView<IrisData>(dataArray);

    // Now let's do something to the data view. For example, concatenate all four non-label columns
    // into 'Features' column.
    dv = new Microsoft.ML.Runtime.Data.ConcatTransform(env, dv, "Features", 
        "SepalLength", "SepalWidth", "PetalLength", "PetalWidth");

    // Read the data into an another array, this time we read the 'Features' and 'Label' columns
    // of the data, and ignore the rest.
    // This method will use the definition of IrisVectorData to understand which columns and of which types
    // are expected to be present in the input data.
    var arr = dv.AsEnumerable<IrisVectorData>(env, reuseRowObject: false)
        .ToArray();
}
```
After this code runs, `arr` will contain two `IrisVectorData` objects, each having `Features` filled with the actual values of the features.

### Streaming data views

What if the original data doesn't support seeking, kile if it's some form of `IEnumerable<IrisData>` instead of `IList<IrisData>`? Well, we can simply use another helper function:
```(csharp)
var streamingDv = env.CreateStreamingDataView<IrisData>(dataEnumerable);
```
The only subtle difference is, the resulting `streamingDv` will not support shuffling (a property that's useful to some ML application).

### AsCursorable and reuseRowObject parameter

When you read a data view as `AsEnumerable<OutType>`, ML.NET will create and populate an object per row. If you do not need multiple row objects to exist in memory (for example, you are writing them to disk one by one, as you scan through the `IEnumerable`), you may want to set `reuseRowObject` to `true`. This will make ML.NET create *only one row object for the entire data view* when you enumerate it, and just re-populate the values every time.

Obviously, in the example above this would lead to incorrect behavior, as the `arr` variable will hold two copies of the same `IrisVectorData` object. Please consider carefully whether you want to reuse the row object, because it is more efficient, but can lead to hard to find issues.

Sometimes, we don't even want to *populate* the row object per row. For example, we only want to see every 100th row of the data, so there's no need to populate the remaining 99% row objects. In this case, you can use `AsCursorable<OutType>` method:

```(csharp)
var cursorable = dv.AsCursorable<IrisVectorData>(env);
// You can create as many simultaneous cursors as you like, they are independent.
using (var cursor = cursorable.GetCursor())
{
    // We are now in charge of creating the row object.
    var myRow = new IrisVectorData();
    while (cursor.MoveNext())
    {
        if (cursor.Position % 100 == 99)
        {
            // Populate the values of the row object.
            cursor.FillValues(myRow);
            // Do something to the row.
        }
    }
}
``` 
Please note that **cursors are not thread-safe**: they have mutable state inside, and they are meant to be used by one thread. If you want to read the data in parallel, use multiple cursors.

## PredictionEngine and PredictorModel

ML.NET's `PredictionEngine` is attempting to turn a sequence of data transforms (maybe capped by a predictor, but not necessarily) into a 'black box' that takes strongly typed inputs and returns strongly typed outputs. The name is a little misleading: the `PredictionEngine` object doesn't require a predictor to be present in the pipeline, it can be just a sequence of transforms like in the below example:

```(csharp)
var engine = env.CreatePredictionEngine<IrisData, IrisVectorData>(dv);
var output = engine.Predict(new IrisData { Label = 1, PetalLength = 1, SepalLength = 1, PetalWidth = 1, SepalWidth = 1 });
```
It is important to note that the `PredictionEngine` actually *validates* that the 'pipeline' conforms to the input and output schema requirements when it is created.

The same can be said about the `PredictorModel<InputType, OutputType>`. This is a somewhat more restricted version of `PredictionEngine` that is created by `LearningPipeline.Train`.

Please note that **`PredictionEngine` and `PredictorModel` are not thread-safe**: they hold an internal cursor object, and therefore cannot be used in a re-entrant fashion.
If you ever see the error message that says: `An attempt was made to keep iterating after the pipe has been reset`, it most likely means that ML.NET has detected a race condition on the `PredictionEngine`.

## Type system mapping

`IDataView` [type system](IDataViewTypeSystem.md) differs slightly from the C# type system, so a 1-1 mapping between column types and C# types is not always feasible. 
Below are the most notable examples of the differences:

* `IDataView` vector columns may have a fixed (and known) size, C# arrays can not. You can use `[VectorType(N)]` attribute to an array field to specify that the column is a vector of fixed size N. This is often necessary: most ML components don't work with variable-size vectors, they require fixed-size ones.
* `IDataView`'s **key types** don't have an underlying C# type either. To declare a key-type column, you need to make your field an `uint`, and decorate it with `[KeyType(Min=A, Count=B)]` to denote that the field is a key with the specified range of values.

### Full list of type mappings
The below table illustrates what C# types are mapped to what `IDataView` types:

| `IDataView` type | C# type     |  C# type with extra conversion |
| ---------------- | ----------- | ------------------------------ |
| `I1`             | `DvInt1`    | `sbyte`, `sbyte?`              |
| `I2`             | `DvInt2`    | `short`, `short?`              |
| `I4`             | `DvInt4`    | `int`, `int?`                  |
| `I8`             | `DvInt8`    | `long`, `long?`                |
| `U1`             | `byte`      | `byte?`                        |
| `U2`             | `ushort`    | `ushort?`                      |
| `U4`             | `uint`      | `uint?`                        |
| `U8`             | `ulong`     | `ulong?`                       |
| `UG`             | `UInt128`   |                                |
| `R4`             | `float`     | `float?`                       |
| `R8`             | `double`    | `double?`                      |
| `TX`             | `DvText`, `string` |                         |
| `BL`             | `DvBool`           | `bool`, `bool?`         |
| `TS`             | `DvTimeSpan`       |                         |
| `DT`             | `DvDateTime`       |                         |
| `DZ`             | `DvDateTimeZone`   |                         |
| Variable-size vector | `VBuffer<T>`   | `T[]`, and the vector is always dense |
| Fixed-size vector    | `VBuffer<T>` with `[VectorType(N)]` | `T[]` with `VectorType(N)`, and the vector is always dense |
| Key type             | `uint` with `[KeyType]`             |                                                            |

### Additional attributes to affect type mapping

There are two more attributes that can affect the way ML.NET conducts schema comprehension:
* `[ColumnName]` lets you choose a different name for the `IDataView` column. By default it is the same as field name.
  * This is a way to create or read back an `IDataView` column with a name containing 'invalid' characters (like whitespace).
* `[NoColumn]` is an attribute that denotes that the below field should not be mapped to a column.

### Using SchemaDefinition for run-time type mapping hints

As you can see from the table and notes above, certain `IDataView` types can only be denoted with an additional field attribute. If the type parameters are not known at compile time (like the size of the fixed-size vector), this is tricky. 

You can use a `SchemaDefinition` object to re-map a type to an `IDataView` schema programmatically. It gives you the same powers as the attributes, but at runtime.
Please see the below example.
```(csharp)
// Vector size is only known at runtime.
int numberOfFeatures = 4;

// Create the default schema definition.
var schemaDef = SchemaDefinition.Create(typeof(IrisVectorData));

// Specify the right vector size.
schemaDef["Features"].ColumnType = new VectorType(NumberType.R4, numberOfFeatures);

// Create a data view.
var dataView = env.CreateDataView<IrisVectorData>(arr, schemaDef);

// Create a prediction engine. You can add custom input and output schema definitions there.
var predictionEngine = env.CreatePredictionEngine<IrisData, IrisVectorData>(dv, outputSchemaDefinition: schemaDef);
```

In addition to the above, you can use `SchemaDefinition` to add per-column metadata, or even a 'value generator' (so that the column value is not read from the field, but computed using a delegate).

## Limitations

Certain things are not possible to do at all using the schema comprehensions, but are possible via the native `IDataView` programmatic interface.
It was our design decision to not allow these scenarios, thus simplifying the other, more common scenarios. 

Here is the list of things that are only possible via the low-level interface:
* Creating or reading a data view, where even column *types* are not known at compile time (so you cannot create a C# class to define the schema)
* Reading a different subset of columns on every row: the cursor always populates the entire row object.
* Reading column metadata from the data view.
* Accessing the 'hidden' data view columns by index.
* Creating 'cursor sets'.
