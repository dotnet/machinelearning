## Sweepable Pipeline

In hyper-parameter optimization, there're two kinds of hyper-parameters: hyper-parameter on estimator-wise and hyper-parameter on pipeline-wise. It's more common to the first kind of hyper-parameters in most of hpo tasks: user provides search space for one or several estimators in a pipeline and tuner sweeps over that search space and find out the best parameter. But It's also common to see questions like should I one-hot encoding column `A` or using hash-encoding, or should I use fast tree or light gbm as trainer being asked while training a model. Such a question about how to pick the best estimator over a few pre-defined candidates is what sweepable pipeline tries to resolve.

## Schema in sweepable pipeline
In `SweepablePipeline`, we use a special schema to represent all available candidate estimators and based on which, symbolize the entire pipeline so we can fit it under the Tuner-SearchSpace paradigm. In that symbolize system, two operators are provided: `+` for `OneOf` and `*` for `Concatenate`, where `OneOf` means one of estimators will be picked up as a pipe stage when constructing pipeline, and `Concatenate` means concatenate two pipe stage together. `Concatenate` has a higher priority than `OneOf` when parsing symbols. And for estimator candidates, they will have their own id in symbolize system, which usually start with `E` and connected with an index, like `E0` or `E1`.

With that symbol system, we can represent nearly any pipelines that has a tree-like structure. For example, pipeline `E0 * E1 * E2` represents a pipeline with three stages, and each stage comes with a single candidate. While `E0 * (E1 + E2) * (E3 + E4 + E5)` represents the case for a pipeline with three stages, with the second stage has two candidates and the third stage with three candidates.

Another interesting feature of that symbol system is the symbol for estimator is recursively defined, similarly like how exression is defined in a programing language, so you can have pipeline as an estimator candidate which adds tons of flexibility when defining a pipeline. For example, suppose you have `P1: E0 * E1 * E2`, `P2: E0 * (E1 + E2) * (E3 + E4 + E5)` and `P3: E0 + E1 + E2 + E3`, to construct a pipeline that uses either `P1` or `P2` in the first stage, and direct result to `P3`, you simply need to put `P4: (P1 + P2) * P3` instead of constructing from very beginning.

In practice, you don't actually need to play with schema directly, instead `SweepablePipeline` provides `.Append(e1, e2, ...)` to simplify that whole process. Therefore, in coding, for pipeline `P1: E0 * E1 * E2`, it will look like
```csharp
var p1
p1 = p1.Append(e0).Append(e1).Append(e2);
```
And for `P2:E0 * (E1 + E2) * (E3 + E4 + E5)`, it will be
```csharp
var p2
p2 = p2.Append(e0).Append(e1, e2).Append(e3, e4, e5);
```
And for P3, P4
```csharp
var p3
p2 = p4.Append(p1, p2).Append(p3);
```

## Seach space in sweepable pipeline
Similarly on search space in sweepable estimator, sweepable pipeline also has search space. And it consists by two parts: search space from its estimators, and search space from its schema. In practice, the two search space will be saved together as nested search space. The key for estimators' search spaces is simply estimators' ids, and the key for schema's search space is hard-coded as `_SCHEMA_`.

For example, for `p2`, its search space will be `{_SCHEMA_: schema_search_space, e0: e0_search_space, e1: e1_search_space, e2: e2_search_space ... e5: e5_search_space}`, where `e_N_search_space` is simply search space from `e_N`, `schema_search_space` is deducted from its schema.

The deducting rule for schema is nothing different than expanding a term. It generates a choice option, where each choice represent a possible way of constructing a trainable pipeline from sweepable pipeline.
For `p1`, its search space will be a choice option with only one choice: `choice:[e0 * e1 * e2]`, for `p2`, it is `choice: [e0 * e1 * e3, e0 * e1 * e4, e0 * e1 * e5,,, e0 * e2 * e5]`, which contains `1 * 2 * 3 = 6` options.

The advantage of using this strategy to represent a sweepable pipeline's search space is it hides the difference for search spaces between sweepable pipeline and estimator and therefore, greatly simplify the developing work for tuners.

Another advantage of putting estimators and schema search spaces altogether is this strategy making it possible to predict model score via hyper-parameter sampling only, which is critial for the future one-shot automl implementation.