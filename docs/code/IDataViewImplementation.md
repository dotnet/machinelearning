# `IDataView` Implementation

This document is intended as an essay on the best practices for `IDataView`
implementations. As a prerequisite, we suppose that someone has read, and
mostly understood, the following documents:

* [Design principles](IDataViewDesignPrinciples.md) and
* [Type system](IDataViewTypeSystem.md).

and has also read and understood the code documentation for the `IDataView`
and its attendant interfaces. Given that background, we will expand on best
practices and common patterns that go into a successful implementation of
`IDataView`, and motivate them with real examples, and historical learnings.

Put another way: There are now within the ML.NET codebase many implementations
of `IDataView` and many others in other related code bases that interface with
ML.NET. The corresponding PRs and discussions have resulted in the
accumulation of some information, stuff that is not and perhaps should not be
covered in the specification or XML code documentation, but that is
nonetheless quite valuable to know. That is, not the `IDataView` spec itself,
but many of the logical implications of that spec.

We will here start with the idioms and practices for `IDataView` generally,
before launching into specific *types* of data views: right now there are two
types of data views that have risen to the dignity of being "general": loaders
and transforms. (There are many "specific" non-general data views: "array"
data views, cache data views, join data views, data views for taking other
abstractions for representing data and phrasing it in a way our code can
understand, but these do not follow any more general pattern as loaders and
transforms do.)

# Urgency in Adhering to Invariants

The point of `IDataView` is that it enables composable data pipelines. But
what does that composability, practically, entail?

There are many implementations of `IDataView` and `IDataTransform` in the
ML.NET codebase. There are, further, many instances of `ITrainer` that consume
those data views. There are more implementations of these currently outside of
this codebase, totaling some hundreds. Astonishingly, they all actually work
well together. The reason why so many transforms can work well with so many
different dataviews as potential inputs, chained in arbitrary and strange ways
we can hardly imagine, and feed well into so many instances of `ITrainer` is
not of course because we wrote code to accommodate the Cartesian product of
all possible inputs, but merely because we assume that any given
implementation of `IDataView` obeys the invariants and principles it must.

This is a general principal of software engineering, or indeed any
engineering: it is nearly impossible to build any complex system of multiple
parts unless those subcomponents adhere to whatever specifications they're
supposed to, and fulfill their requirements.

We can to some extent tolerate divergence from the invariants in *some*
components, if they are isolated: we have some losses that behave strangely,
even trainers behave somewhat strangely, sort of. Yet `IDataView` is the
center of our data pipeline, and divergences are more potentially harmful.
There is, for every requirement listed here, actually *something* that is
relying on it.

The inverse is also true: not only must `IDataView` conform to invariants,
code that consumes `IDataView` should be robust to situations other than the
"happy path." It needn't succeed, but it should at least be able to detect if
data is not in the expected form and throw an error message to the user
telling them how they misused it.

To give the most common example of what I have seen in PRs: often one designs
a transform or learner whose anticipated usage is that it will be used in
conjunction with another transform "upstream" to prepare the data. (Again,
this is very common: a `KeyToVector` transform for example assumes there's
*something* upstream producing key values.) What happens sometimes is people
forget to check that the input data actually *does* conform to that, with the
result that if a pipeline was composed in some other fashion, there would be
some error.

The only thing you can really assume is that an `IDataView` behaves "sanely"
according to the contracts of the `IDataView` interface, so that future ML.NET
developers can form some reasonable expectations of how your code behaves, and
also have a prayer of knowing how to maintain the code. It is hard enough to
write software correctly even when the code you're working with actually does
what it is supposed to, and impossible when it doesn't. Anyway, not to belabor
the point: hidden undocumented implicit requirements on the usage

# Design Decisions

Presumably you are motivated to read this document because you have some
problem of how to get some data into ML.NET, or process data using ML.NET, or
something along these lines. There is a decision to be made about how to even
engineer a solution. Sometimes it's quite obvious: text featurization
obviously belongs as a transform. But other cases are *less* obvious. We will
talk here about how we think about these things.

One crucial question is whether something should be a data view at all: Often
there is ambiguity. To give some examples of previously contentious points:
should clustering be *transform* or a *trainer*? What about PCA? What about
LDA? In the end, we decided clustering was a *trainer* and both PCA and LDA
are *transforms*, but this decision was hardly unambiguous. Indeed, what
purpose is served by considering trainers and transforms fundamentally
different things, at all?

Even once we decide whether something *should* be an `IDataView` of some sort,
the question remains what type of data view. We have some canonical types of
data views:

If it involves taking data from a stream, like a file, or some sort of stream
of data from a network, or other such thing, we might consider this a
*loader*, that is, it should perhaps implement `IDataLoader`.

If it involves taking a *single* data view, and transmuting it in some
fashion, **and** the intent is this same transmutation might be applied to
novel data, then it should perhaps implement `IDataTransform`, and be a
transform.

Now then, consider that not everything should be a loader, or a transform,
even when data could be considered to be read from a stream, or when there is
a data view based on another single data view. The essential purpose of loader
and transforms is that they can exist as part of the data model, that is, they
should be serializable and applicable to new data. A nice rule of thumb is: if
when designing some you can imagine a scenario where you want to apply some
logic to *both* a training set as well as a test set, then it might make sense
to make it a loader or a transform. If not, it probably does not make sense.

1. Often data comes from some programmatic source, as a starting point for an
   ML.NET pipeline. Despite being at the head of the data pipe, it is *not* a
   loader, because the data source is not a stream (though it is stream*ing*):
   it is a `RowSetDataView`.

2. During training, data is sometimes cached. the structure that handles the
   data caching is a `CacheDataView`. It is absolutely not a transform,
   despite taking a single input and being itself an `IDataView`. There is no
   reason to make it a transform, because there is no plausible rationale to
   make it part of the data model: the decision of whether you want to cache
   data during *training* has nothing at all to do with whether you want to
   cache data during *scoring*, so there is no point in saving it to the data
   model.

3. The ML.NET API for prediction uses a scheme that phrases input data
   programmatically as coming from an enumerable of typed objects: the
   underlying programmatic `IDataView` that is constructed to wrap this is
   *not* a loader, because it is not part of the data model. It is merely the
   entry point to the data model, at least, in typical usage.

# Why `GetGetter`?

Let us address something fairly conspicuous. The question almost everyone
asks, when they first start using `IDataView`: what is up with these getters?

One does not fetch values directly from an `IRow` implementation (including
`IRowCursor`). Rather, one retains a delegate that can be used to fetch
objects, through the `GetGetter` method on `IRow`. This delegate is:

```csharp
public delegate void ValueGetter<TValue>(ref TValue value);
```

If you are unfamiliar with delegates, [read
this](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/delegates/).
Anyway: you open a row cursor, you get the delegate through this `GetGetter`
method, and you use this delegate multiple times to fetch the actual column
values as you `MoveNext` through the cursor.

Some history to motivate this: In the first version of `IDataView` the
`IRowCursor` implementation did not actually have these "getters" but rather
had a method, `GetColumnValue<TValue>(int col, ref TValue val)`. However, this
has the following problems:

* **Every** call had to verify that the column was active,
* **Every** call had to verify that `TValue` was of the right type,
* When these were part of, say, a transform in a chain (as they often are,
  considering how common transforms are used by ML.NET's users) each access
  would be accompanied by a virtual method call to the upstream cursor's
  `GetColumnValue`.

In contrast, consider the situation with these getter delegates. The
verification of whether the column is active happens *exactly* once. The
verification of types happens *exactly* once. Rather than *every* access being
passed up through a chain of dozens of transform cursors, you merely get a
getter from whatever cursor is serving it up, and do every access directly
without having to pass through umpteen virtual method calls (each, naturally,
accompanied by their own checks!). With these preliminaries done, a getter on
every iteration, when called, merely has to just fill in the value: all this
verification work is already taken care of. The practical result of this is
that, for some workloads where the getters merely amounted to assigning
values, the "getter" method became an order of magnitude faster. So: we got
rid of this `GetColumnValue` method, and now work with `GetGetter`.

# Repeatability

A single `IDataView` instance should be considered a consistent view onto
data. So: if you open a cursor on the same `IDatView` instance, and access
values for the same columns, it will apparently be a "consistent" view. It is
probably obvious what this mean, but specifically:

The cursor as returned through `GetRowCursor` (with perhaps an identically
constructed `IRandom` instance) in any iteration should return the same number
of rows on all calls, and with the same values at each row.

Why is this important? Many machine learning algorithms require multiple
passes over the dataset. Most stochastic methods wouldn't really care if the
data changed, but others are *very* sensitive to changes in the data. For
example, how could an L-BFGS or OWL-QN algorithm effectively compute its
approximation to a Hessian, if the examples from which the per-pass history
are computed were not consistent? How could a dual algorithm like SDCA
function with any accuracy, if the examples associated with any given dual
variable were to change? Consider even a relatively simple transform, like a
forward looking windowed averager, or anything relating to time series. The
implementation of those `ICursor` interfaces often open *two* cursors on the
underlying `IDataView`, one "look ahead" cursor used to gather and calculate
necessary statistics, and another cursor for any data: how could the column
constructed out of that transform be meaningful of the look ahead cursor was
consuming different data from the contemporaneous cursor? There are many
examples of this throughout the codebase.

Nevertheless: in very specific circumstances we have relaxed this. For
example, some ML.NET API code serves up corrupt `IDataView` implementations
that have their underlying data change, since reconstituting a data pipeline
on fresh data is at the present moment too resource intensive. Nonetheless,
this is wrong: for example, the `TrainingCursorBase` and related subclasses
rely upon the data not changing. Since, however, that is used for *training*
and the prediction engines of the API as used for *scoring*, we accept these.
However this is not, strictly speaking, correct, and this sort of corruption
of `IDataView` should only be considered as a last resort, and only when some
great good can be accomplished through this. We certainly did not accept this
corruption lightly!

# Norms for the Data Model

In a similar vein for repeatability and consistency is the notion of the data
model. Unlike repeatability, this topic is a bit specialized: `IDataView`
specifically is not serializable, but both `IDataLoader` and `IDataTransform`
are serializable. Nonetheless those are the two most important types of data
views, so we will treat on them here.

From a user's perspective, when they run ML.NET and specify a loader or set of
transforms, what they are doing is composing a data pipe. For example, perhaps
they specify a way to load data from, say, a text file, apply some
normalization, some categorical handling, some text, some this, some that,
some everything, and it all just works, and is consistent whether we're
applying that to the training data on which the transforms were defined, or
some other test set, whether we programmatically load the model in the API and
apply it to some production setting, whether we are running in a distributed
environment and want to make sure *all* worker nodes are featurizing data in
exactly the same way, etc. etc.

The way in which this consistency is accomplished is by having certain
requirements on the essential parts of the data model: loaders and transforms.
The essential reason these things exist is so that they can be applied to new
data in a consistent way.

Let us formalize this somewhat. We consider two data views to be functionally
identical if there is absolutely no way to distinguish them: they return the
same values, have the same types, same number of rows, they shuffle
identically given identically constructed `IRandom` when row cursors are
constructed, return the same ID for rows from the ID getter, etc. Obviously
this concept is transitive. (Of course, `Batch` in a cursor might be different
between the two, but that is the case even with two cursors constructed on the
same data view.) So some rules:

1. If you have an `IDataLoader`, then saving/loading the associated data model
   on the same data should result in a functionally identical `IDataLoader`.

2. If you have an `IDataTransform`, then saving/loading the associated data
   model for the transforms on functionally identical `IDataView`s, should
   itself result in functionally identical `IDataView`s.

## Versioning

This requirement for consistency of a data model often has implications across
versions of ML.NET, and our requirements for data model backwards
compatibility. As time has passed, we often feel like it would make sense if a
transform behaved *differently*, that is, if it organized or calculated its
output in a different way than it currently does. For example, suppose we
wanted to switch the hash transform to something a bit more efficient than
murmur hashes, for example. If we did so, presumably the same input values
would map to different outputs. We are free to do so, of course, yet: when we
deserialize a hash transform from before we made this change, that hash
transform should continue to output values as it did, before we made that
change. (This, of course, assuming that the transform was released as part of
a "blessed" non-preview point release of ML.NET. We can, and have, broken
backwards compatibility for something that has not yet been incorporated in
any sort of blessed release, though we prefer to not.)

## What is Not Functionally Identical

Note that identically *constructed* data views are not necessarily
*functionally* identical. Consider this usage of the train and score transform
with `xf=trainScore{tr=ap}`, where we first train averaged perceptron, then
copy its score and probability columns out of the way, then construct the
same basic transform again.

```maml
maml.exe showdata saver=md seed=1 data=breast-cancer.txt xf=trainScore{tr=ap}
    xf=copy{col=ScoreA:Score col=ProbA:Probability} xf=trainScore{tr=ap}
```

The result is this.

Label | Features                     | PredictedLabel | Score  | Probability  | ScoreA | ProbA
------|------------------------------|----------------|--------|--------------|--------|-------
0     | 5, 1, 1, 1, 2, 1, 3, 1, 1    | 0              | -62.07 | 0.0117       | -75.28 | 0.0107
0     | 5, 4, 4, 5, 7, 10, 3, 2, 1   | 1              |  88.41 | 0.8173       |  92.04 | 0.8349
0     | 3, 1, 1, 1, 2, 2, 3, 1, 1    | 0              | -40.53 | 0.0269       | -44.23 | 0.0329
0     | 6, 8, 8, 1, 3, 4, 3, 7, 1    | 1              | 201.21 | 0.9973       | 208.07 | 0.9972
0     | 4, 1, 1, 3, 2, 1, 3, 1, 1    | 0              | -43.11 | 0.0243       | -55.32 | 0.0221
1     | 8, 10, 10, 8, 7, 10, 9, 7, 1 | 1              | 259.22 | 0.9997       | 257.43 | 0.9995
0     | 1, 1, 1, 1, 2, 10, 3, 1, 1   | 1              |  71.10 | 0.6933       |  89.52 | 0.8218
0     | 2, 1, 2, 1, 2, 1, 3, 1, 1    | 0              | -38.94 | 0.0286       | -39.59 | 0.0388
0     | 2, 1, 1, 1, 2, 1, 1, 1, 5    | 0              | -32.87 | 0.0360       | -41.52 | 0.0362
0     | 4, 2, 1, 1, 2, 1, 2, 1, 1    | 0              | -31.76 | 0.0376       | -41.68 | 0.0360

One could argue it's not *really* identically constructed, exactly, since both
of those transforms (including the underlying averaged perceptron learner!)
are initialized using the pseudo-random number generator in an `IHost` that
changes from one to another. But, that's a bit nit-picky.

Note also: when we say functionally identical we include everything about it:
not just the data, but the schema, its metadata, the implementation of
shuffling, etc. For this reason, while serializing the data *model* has
guarantees of consistency, serializing the *data* has no such guarantee: if
you serialize data using the text saver, practically all metadata (except slot
names) will be completely lost, which can have implications on how some
transforms and downstream processes work. Or: if you serialize data using the
binary saver, suddenly it may become shufflable whereas it may not have been
before.

The inevitable caveat to all this stuff about "consistency" is that it is
ultimately limited by hardware and other runtime environment factors: the
truth is, certain machines will, with identical programs with seemingly
identical flows of execution result, *sometimes*, in subtly different answers
where floating point values are concerned. Even on the same machine there are
runtime considerations, e.g., when .NET's RyuJIT was introduced in VS2015, we
had lots of test failures around our model consistency tests because the JIT
was compiling the CLI just *slightly* differently. But, this sort of thing
aside (which we can hardly help), we expect the models to be the same.

# On Loaders, Data Models, and Empty `IMultiStreamSource`s

When you create a loader you have the option of specifying not only *one* data
input, but any number of data input files, including zero. But there's also a
more general principle at work here with zero files: when deserializing a data
loader from a data model with an `IMultiStreamSource` with `Count == 0` (e.g.,
as would be constructed with `new MultiFileSource(null)`), we have a protocol
that *every* `IDataLoader` should work in that circumstance, and merely be a
data view with no rows, but the same schema as it had when it was serialized.
The purpose of this is that we often have circumstances where we need to
understand the schema of the data (what columns were produced, what the
feature names are, etc.) when all we have is the data model. (E.g., the
`savemodel` command, and other things.)

# Getters Must Fail for Invalid Types

For a given `IRow`, we must expect that `GetGetter<TValue>(col)` will throw if
either `IsColumnActive(col)` is `false`, or `typeof(TValue) !=
Schema.GetColumnType(col).RawType`, as indicated in the code documentation.
But why? It might seem reasonable to add seemingly "harmless" flexibility to
this interface. So let's imagine your type should be `float`, because the
corresponding column's type's `RawType` is `typeof(float)`. Now: if you
*happen* to call `GetGetter<double>(col)` instead of `GetGetter<float>(col)`,
it would actually be a fairly easy matter for `GetGetter` to actually
accommodate it, by doing the necessary transformations under the hood, and
*not* fail. This type of thinking is actually insidiously and massively
harmful to the codebase, as I will remark.

The danger of writing code is that there's a chance someone might find it
useful. Imagine a consumer of your dataview actually relies on your
"tolerance." What that means, of course, is that this consuming code cannot
function effectively on any *other* dataview. The consuming code is by
definition *buggy*: it is requesting data of a type we've explicitly claimed,
through the schema, that we do not support. And the developer, through a well
intentioned but misguided design decision, has allowed buggy code to pass a
test it should have failed, thus making the codebase more fragile when, if we
had simply maintained requirements, would have otherwise detected the bug.

Moreover: it is a solution to a problem that does not exist. `IDataView`s are
fundamentally composable structures already, and one of the most fundamental
operations you can do is transform columns into different types. So, there is
no need for you to do the conversion yourself. Indeed, it is harmful for you
to try: if we have the conversion capability in one place, including the logic
of what can be converted and *how* these things are to be converted, is it
reasonable to suppose we should have it in *every implementation of
`IDataView`?* Certainly not. At best the situation will be needless complexity
in the code: more realistically it will lead to inconsistency, and from
inconsistency, surprises and bugs for users and developers.

# Thread Safety

Any `IDataView` implementation, as well as the `ISchema`, *must* be thread
safe. There is a lot of code that depends on this. For example, cross
validation works by operating over the same dataset (just, of course, filtered
to different subsets of the data). That amounts to multiple cursors being
opened, simultaneously, over the same data.

So: `IDataView` and `ISchema` must be thread safe. However, `IRowCursor`,
being a stateful object, we assume is accessed from exactly one thread at a
time. The `IRowCursor`s returned through a `GetRowCursorSet`, however, which
each single one must be accessed by a single thread at a time, multiple
threads can access this set of cursors simultaneously: that's why we have that
method in the first place.

# Exceptions and Errors

There is one non-obvious implication of the lazy evaluation while cursoring
over an `IDataView`: while cursoring, you should almost certainly not throw
exceptions.

Imagine you have a `TextLoader`. You might expect that if you have a parse
error, e.g., you have a column of floats, and one of the rows has a value
like, `"hi!"` or something otherwise uninterpretable, you would throw. Yet,
consider the implications of lazy evaluation. If that column were not
selected, the cursoring would *succeed*, because it would not look at that
`"hi!"` token *at all*, much less detect that it was not parsable as a float.

If we were to throw, the effect is that *sometimes* the cursoring will succeed
(if the column is not selected), and *sometimes* will fail (if not selected).
These failures are explainable, ultimately, of course, in the sense that
anything is explainable, but a user knows nothing about lazy evaluation or
anything like this: correspondingly this is enormously confusing.

The implication is that we should not throw an exception in this case. We
instead consider this value "missing," and we *may* register a warning using
an `IChannel.Warning`, but we cannot fail.

So: If you could reasonably catch the exception on *any* cursoring over your
`IDataView`, you can throw. If, however, detecting the condition on which you
could throw the exception requires that a certain column be made active, then
you should not throw. Of course, there are extreme circumstances: for example,
one cannot help but throw on a cursoring if, say, there is some weird system
event, and if one somehow detects in a subsequent iteration that something is
fundamentally broken then you can throw: e.g., the binary loader will throw if
it detects the file it is reading is corrupted, even if that corruption may
not have been obvious immediately.

# `GetGetter` Returning the Same Delegate

On a single instance of `IRowCursor`, since each `IRowCursor` instance has no
requirement to be thread safe, it is entirely legal for a call to `GetGetter`
on a single column to just return the same getting delegate. It has come to
pass that the majority of implementations of `IRowCursor` actually do that,
since it is in some ways easier to write the code that way.

This practice has inadvertently enabled a fairly attractive tool for analysis
of data pipelines: by returning the same delegate each time, we can check in a
data pipeline what data is being passed through by seeing whether the
references to getter delegates are being passed through. Now this is
imperfect, because some transforms that could use the same delegate each time
do not, but the vast majority do.

# Class Structuring

The essential attendant classes of an `IDataView` are its schema, as returned
through the `Schema` property, as well as the `IRowCursor` implementation(s),
as returned through the `GetRowCursor` and `GetRowCursorSet` methods. The
implementations for those two interfaces are typically nested within the
`IDataView` implementation itself. The cursor implementation is almost always
at the bottom of the data view class.

# `IRow` and `ICursor` vs. `IRowCursor`

We have `IRowCursor` which descends from both `IRow` and `ICursor`. Why do
these other interfaces exist?

Firstly, there are implementations of `IRow` or `ICursor` that are not
`IRowCursor`s. We have occasionally found it useful to have something
resembling a key-value store, but that is strongly, dynamically typed in some
fashion. Why not simply represent this using the same idioms of `IDataView`?
So we put them in an `IRow`. Similarly: we have several things that behave
*like* cursors, but that are in no way *row* cursors.

However, more than that, there are a number of utility functions where we want
to operate over something like an `IRowCursor`, but we want to have some
indication that this function will not move the cursor (in which case `IRow`
is helpful), or that will not access any values (in which case `ICursor` is
helpful).

# Schema

The schema contains information about the columns. As we see in [the design
principles](IDataViewDesignPrinciples.md), it has index, data type, and
optional metadata.

While *programmatically* accesses to an `IDataView` are by index, from a
user's perspective the indices are by name; most training algorithms
conceptually train on the `Features` column (under default settings). For this
reason nearly all usages of an `IDataView` will be prefixed with a call to the
schema's `TryGetColumnIndex`.

Regarding name hiding, the principles mention that when multiple columns have
the same name, other columns are "hidden." The convention all implementations
of `ISchema` obey is that the column with the *largest* index. Note however
that this is merely convention, not part of the definition of `ISchema`.

Implementations of `TryGetColumnIndex` should be O(1), that is, practically,
this mapping ought to be backed with a dictionary in most cases. (There are
obvious exceptions like, say, things like `LineLoader` which produce exactly
one column. There, a simple equality test suffices.)

It is best if `GetColumnType` returns the *same* object every time. That is,
things like key-types and vector-types, when returned, should not be created
in the function itself (thereby creating a new object every time), but rather
stored somewhere and returned.

## Metadata

Since metadata is *optional*, one is not obligated to necessarily produce it,
or conform to any particular schemas for any particular kinds (beyond, say,
the obvious things like making sure that the types and values are consistent).
However, the flip side of that freedom given to *producers*, is that
*consumers* are obligated, when processing a data view input, to react
gracefully when metadata of a certain kind is absent, or not in a form that
one expects. One should *never* fail when input metadata is in a form one does
not expect.

To give a practical example of this: many transforms, learners, or other
components that process `IDataView`s will do something with the slot names,
but when the `SlotNames` metadata kind for a given column is either absent,
*or* not of the right type (vectors of strings), *or* not of the right size
(same length vectors as the input), the behavior is not to throw or yield
errors or do anything of the kind, but to simply say, "oh, I don't really have
slot names," and proceed as if the slot names hadn't been present at all.
