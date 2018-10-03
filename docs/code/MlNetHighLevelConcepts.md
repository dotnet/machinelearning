# ML.NET high-level concepts

In this document, we give a brief overview of the ML.NET high-level concepts. This document is mainly intended to describe the *model training* scenarios in ML.NET, since not all these concepts are relevant for the more simple scenario of *prediction with existing model*.

## List of high-level concepts

This document is going to cover the following ML.NET concepts:

- *Data*, represented as an `IDataView` interface.
  - In ML.NET, data is very similar to a SQL view: it's a lazily-evaluated, immutable, cursorable, heterogenous, schematized dataset. 
  - An excellent document about the data interface is [IDataView Design Principles](IDataViewDesignPrinciples.md).
- *Transformer*, represented as `ITransformer` interface.
  - In one sentence, a transformer is a component that takes data, does some work on it, and return new 'transformed' data.
  - For example, you can think of a machine learning model as a transformer that takes features and returns predictions.
  - Another example, 'text tokenizer' would take a single text column and output a vector column with individual 'words' extracted out of the texts.
- *Data reader*, represented as an `IDataReader<T>` interface.
  - The data reader is ML.NET component to 'create' data: it takes an instance of `T` and returns data out of it. 
  - For example, a *TextLoader* is an `IDataReader<FileSource>`: it takes the file source and produces data. 
- *Estimator*, represented as an `IEstimator<T>` interface.
  - This is an object that learns from data. The result of the learning is a *transformer*.
  - You can think of a machine learning *algorithm* as an estimator that learns on data and produces a machine learning *model* (which is a transformer).
- *Prediction function*, represented as a `PredictionFunction<TSrc, TDst>` class.
  - The prediction function can be seen as a machine that applies a transformer to one 'row', such as at prediction time.

## Data

In ML.NET, data is very similar to a SQL view: it's a lazily-evaluated, cursorable, heterogenous, schematized dataset.

- It has *Schema* (an instance of an `ISchema` interface), that contains the information about the data view's columns.
  - Each column has a *Name*, a *Type*, and an arbitrary set of *metadata* associated with it.
  - It is important to note that one of the types is the `vector<T, N>` type, which means that the column's values are *vectors of items of type T, with the size of N*. This is a recommended way to represent multi-dimensional data associated with every row, like pixels in an image, or tokens in a text.
  - The column's *metadata* contains information like 'slot names' of a vector column and suchlike. The metadata itself is actually represented as another one-row *data*, that is unique to each column.
- The data view is a source of *cursors*. Think SQL cursors: a cursor is an object that iterates through the data, one row at a time, and presents the available data.
  - Naturally, data can have as many active cursors over it as needed: since data itself is immutable, cursors are truly independent.
  - Note that cursors typically access only a subset of columns: for efficiency, we do not compute the values of columns that are not 'needed' by the cursor.

## Transformer

A transformer is a component that takes data, does some work on it, and return new 'transformed' data.

Here's the interface of `ITransformer`:
```c#
public interface ITransformer
{
    IDataView Transform(IDataView input);
    ISchema GetOutputSchema(ISchema inputSchema);
}
```

As you can see, the transformer can `Transform` an input data to produce the output data. The other method, `GetOutputSchema`, is a mechanism of *schema propagation*: it allows you to see how the output data will look like for a given shape of the input data without actually performing the transformation.

Most transformers in ML.NET tend to operate on one *input column* at a time, and produce the *output column*. For example a `new HashTransformer("foo", "bar")` would take the values from column "foo", hash them and put them into column "bar". 

It is also common that the input and output column names are the same. In this case, the old column is 'replaced' with the new one. For example, a `new HashTransformer("foo")` would take the values from column "foo", hash them and 'put them back' into "foo". 

Any transformer will, of course, produce a new data view when `Transform` is called: remember, data views are immutable.

Another important consideration is that, because data is lazily evaluated, *transformers are lazy too*. Essentially, after you call
```c#
var newData = transformer.Transform(oldData)
```
no actual computation will happen: only after you get a cursor from `newData` and start consuming the value will `newData` invoke the `transformer`'s transformation logic (and even that only if `transformer` in question is actually needed to produce the requested columns).

### Transformer chains

A useful property of a transformer is that *you can phrase a sequential application of transformers as yet another transformer*:

```c#
var fullTransformer = transformer1.Append(transformer2).Append(transformer3);
```

We utilize this property a lot in ML.NET: typically, the trained ML.NET model is a 'chain of transformers', which is, for all intents and purposes, a *transformer*. 

## Data reader

The data reader is ML.NET component to 'create' data: it takes an instance of `T` and returns data out of it. 

Here's the exact interface of `IDataReader<T>`:
```c#
public interface IDataReader<in TSource>
{
    IDataView Read(TSource input);
    ISchema GetOutputSchema();
}
```
As you can see, the reader is capable of reading data (potentially multiple times, and from different 'inputs'), but the resulting data will always have the same schema, denoted by `GetOutputSchema`.

An interesting property to note is that you can create a new data reader by 'attaching' a transformer to an existing data reader. This way you can have 'reader' with transformation behavior baked in:
```c#
var newReader = reader.Append(transformer1).Append(transformer2)
```

Another similarity to transformers is that, since data is lazily evaluated, *readers are lazy*: no (or minimal) actual 'reading' happens when you call `dataReader.Read()`: only when a cursor is requested on the resulting data does the reader begin to work.

## Estimator

The *estimator* is an object that learns from data. The result of the learning is a *transformer*.
Here is the interface of `IEstimator<T>`:
```c#
public interface IEstimator<out TTransformer>
    where TTransformer : ITransformer
{
    TTransformer Fit(IDataView input);
    SchemaShape GetOutputSchema(SchemaShape inputSchema);
}
```

You can easily imagine how *a sequence of estimators can be phrased as an estimator* of its own. In ML.NET, we rely on this property to create 'learning pipelines' that chain together different estimators:

```c#
var env = new LocalEnvironment(); // Initialize the ML.NET environment.
var estimator = new ConcatEstimator(env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
    .Append(new ToKeyEstimator(env, "Label"))
    .Append(new SdcaMultiClassTrainer(env, "Features", "Label")) // This is the actual 'machine learning algorithm'.
    .Append(new ToValueEstimator(env, "PredictedLabel"));

var endToEndModel = estimator.Fit(data); // This now contains all the transformers that were used at training.
```

One important property of estimators is that *estimators are eager, not lazy*: every call to `Fit` is causing 'learning' to happen, which is potentially a time-consuming operation.

## Prediction function

The prediction function can be seen as a machine that applies a transformer to one 'row', such as at prediction time.

Once we obtain the model (which is a *transformer* that we either trained via `Fit()`, or loaded from somewhere), we can use it to make 'predictions' using the normal calls to `model.Transform(data)`. However, when we use this model in a real life scenario, we often don't have a whole 'batch' of examples to predict on. Instead, we have one example at a time, and we need to make timely predictions on them immediately.

Of course, we can reduce this to the batch prediction:
- Create a data view with exactly one row.
- Call `model.Transform(data)` to obtain the 'predicted data view'.
- Get a cursor over the resulting data.
- Advance the cursor one step to get to the first (and only) row.
- Extract the predicted values out of it.

The above algorithm can be implemented using the [schema comprehension](SchemaComprehension.md), with two user-defined objects `InputExample` and `OutputPrediction` as follows:

```c#
var inputData = env.CreateDataView(new InputExample[] { example });
var outputData = model.Transform(inputData);
var output = outputData.AsDynamic.AsEnumerable<OutputPrediction>(env, reuseRowObject: false).Single();
```

But this would be cumbersome, and would incur performance costs. 
Instead, we have a 'prediction function' object that performs the same work, but faster and more convenient, via an extension method `MakePredictionFunction`:

```c#
var predictionFunc = model.MakePredictionFunction<InputExample, OutputPrediction>(env);
var output = predictionFunc.Predict(example);
```

The same `predictionFunc` can (and should!) be used multiple times, thus amortizing the initial cost of `MakePredictionFunction` call. 

The prediction function is *not re-entrant / thread-safe*: if you want to conduct predictions simultaneously with multiple threads, you need to have a prediction function per thread.
