# IDataView Design Principles

## Overview

### Brief Introduction to IDataView

The *IDataView system* is a set of interfaces and components that provide
efficient, compositional processing of schematized data for machine learning
and advanced analytics applications. It is designed to gracefully and
efficiently handle high dimensional data and large data sets. It does not
directly address distributed data and computation, but is suitable for single
node processing of data partitions belonging to larger distributed data sets.

IDataView is the data pipeline machinery for ML.NET. Microsoft teams consuming
this library have implemented libraries of IDataView related components
(loaders, transforms, savers, trainers, predictors, etc.) and have validated
the performance, scalability and task flexibility benefits.

The name IDataView was inspired from the database world, where the term table
typically indicates a mutable body of data, while a view is the result of a
query on one or more tables or views, and is generally immutable. Note that
both tables and views are schematized, being organized into typed columns and
rows conforming to the column types. Views differ from tables in several ways:

* Views are *composable*. New views are formed by applying transformations
  (queries) to other views. In contrast, forming a new table from an existing
  table involves copying data, making the tables decoupled; the new table is
  not linked to the original table in any way.

* Views are *virtual*; tables are fully realized/persisted. In other words, a
  table contains the values in the rows while a view computes values from
  other views or tables, so does not contain or own the values.

* Views are *immutable*; tables are mutable. Since a view does not contain
  values, but merely computes values from its source views, there is no
  mechanism for modifying the values.

Note that immutability and compositionality are critical enablers of
technologies that require reasoning over transformation, like query
optimization and remoting. Immutability is also key for concurrency and thread
safety. Views being virtual minimizes I/O, memory allocation, and computation.
Information is accessed, memory is allocated, and computation is performed,
only when needed to satisfy a local request for information.

### Design Requirements

The IDataView design fulfills the following design requirements:

* **General schema**: Each view carries schema information, which specifies
  the names and types of the view's columns, together with metadata associated
  with the columns. The system is optimized for a reasonably small number of
  columns (hundreds). See [here](#basics).

* **Open type system**: The column type system is open, in the sense that new
  data types can be introduced at any time and in any assembly. There is a set
  of standard types (which may grow over time), but there is no registry of
  all supported types. See [here](#basics).

* **High dimensional data support**: The type system for columns includes
  homogeneous vector types, so a set of related primitive values can be
  grouped into a single vector-valued column. See [here](#vector-types).

* **Compositional**: The IDataView design supports components of various
  kinds, and supports composing multiple primitive components to achieve
  higher-level semantics. See [here](#components).

* **Open component system**: While the ML.NET code has a growing large library
  of IDataView components, additional components that interoperate with these
  may be implemented in other code bases. See [here](#components).

* **Cursoring**: The rows of a view are accessed sequentially via a row
  cursor. Multiple cursors can be active on the same view, both sequentially
  and in parallel. In particular, views support multiple iterations through
  the rows. Each cursor has a set of active columns, specified at cursor
  construction time. Shuffling is supported via an optional random number
  generator passed at cursor construction time. See [here](#cursoring).

* **Lazy computation**: When only a subset of columns or a subset of rows is
  requested, computation for other columns and rows can be, and generally is,
  avoided. Certain transforms, loaders, and caching scenarios may be
  speculative or eager in their computation, but the default is to perform
  only computation needed for the requested columns and rows. See
  [here](#lazy-computation-and-active-columns).

* **Immutability and repeatability**: The data served by a view is immutable
  and any computations performed are repeatable. In particular, multiple
  cursors on the view produce the same row values in the same order (when
  using the same shuffling). See [here](#immutability-and-repeatability).

* **Memory efficiency**: The IDataView design includes cooperative buffer
  sharing patterns that eliminate the need to allocate objects or buffers for
  each row when cursoring through a view. See [here](#memory-efficiency).

* **Batch-parallel computation**: The IDataView system includes the ability to
  get a set of cursors that can be executed in parallel, with each individual
  cursor serving up a subset of the rows. Splitting into multiple cursors can
  be done either at the loader level or at an arbitrary point in a pipeline.
  The component that performs splitting also provides the consolidation logic.
  This enables computation heavy pipelines to leverage multiple cores without
  complicating each individual transform implementation. See
  [here](#batch-parallel-cursoring).

* **Large data support**: Constructing views on data files and cursoring
  through the rows of a view does not require the entire data to fit in
  memory. Conversely, when the entire data fits, there is nothing preventing
  it from being loaded entirely in memory. See [here](#data-size).

### Design Non-requirements

The IDataView system design does *not* include the following:

* **Multi-view schema information**: There is no direct support for specifying
  cross-view schema information, for example, that certain columns are primary
  keys, and that there are foreign key relationships among tables. However,
  the column metadata support, together with conventions, may be used to
  represent such information.

* **Standard ML schema**: The IDataView system does not define, nor prescribe,
  standard ML schema representation. For example, it does not dictate
  representation of nor distinction between different semantic interpretations
  of columns, such as label, feature, score, weight, etc. However, the column
  metadata support, together with conventions, may be used to represent such
  interpretations.

* **Row count**: A view is not required to provide its row count. The
  `IDataView` interface has a `GetRowCount` method with type `Nullable<long>`.
  When this returns `null`, the row count is not available directly from the
  view.

* **Efficient indexed row access**: There is no standard way in the IDataView
  system to request the values for a specific row number. While the
  `IRowCursor` interface has a `MoveMany(long count)` method, it only supports
  moving forward `(count > 0)`, and is not necessarily more efficient than
  calling `MoveNext()` repeatedly. See [here](#row-cursor).

* **Data file formats**: The IDataView system does not dictate storage or
  transport formats. It *does* include interfaces for loader and saver
  components. The ML.NET code has implementations of loaders and savers for
  some binary and text file formats.

* **Multi-node computation over multiple data partitions**: The IDataView
  design is focused on single node computation. We expect that in multi-node
  applications, each node will be given its own data partition(s) to operate
  on, with aggregation happening outside an IDataView pipeline.

## Schema and Type System

### Basics

IDataView has general schema support, in that a view can have an arbitrary
number of columns, each having an associated name, index, data type, and
optional metadata.

Column names are case sensitive. Multiple columns can share the same name, in
which case, one of the columns hides the others, in the sense that the name
will map to one of the column indices, the visible one. All user interaction
with columns should be via name, not index, so the hidden columns are
generally invisible to the user. However, hidden columns are often useful for
diagnostic purposes.

The set of supported column data types forms an open type system, in the sense
that additional types can be added at any time and in any assembly. However,
there is a precisely defined set of standard types including:

* Text
* Boolean
* Single and Double precision floating point
* Signed integer values using 1, 2, 4, or 8 bytes
* Unsigned integer values using 1, 2, 4, or 8 bytes
* Unsigned 16 byte values for ids and probabilistically unique hashes
* Date time, date time zone, and timespan
* Key types
* Vector types

The set of standard types will likely be expanded over time.

The IDataView type system is specified in a separate document, *IDataView Type
System Specification*.

IDataView provides a general mechanism for associating semantic metadata with
columns, such as designating sets of score columns, names associated with the
individual slots of a vector-valued column, values associated with a key type
column, whether a column's data is normalized, etc.

While IDataView schema supports an arbitrary number of columns, it, like most
schematized data systems, is designed for a modest number of columns,
typically, limited to a few hundred. When a large number of *features* are
required, the features should be gathered into one or more vector-valued
columns, as discussed in the next section. This is important for both user
experience and performance.

### Vector Types

Machine learning and advanced analytics applications often involve high-
dimensional data. For example, a common technique for learning from text,
known as [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model),
represents each word in the text as a numeric feature containing the number of
occurrences of that word. Another technique is indicator or one-hot encoding
of categorical values, where, for example, a text-valued column containing a
person's last name is expanded to a set of features, one for each possible
name (Tesla, Lincoln, Gandhi, Zhang, etc.), with a value of one for the
feature corresponding to the name, and the remaining features having value
zero. Variations of these techniques use hashing in place of dictionary
lookup. With hashing, it is common to use 20 bits or more for the hash value,
producing `2^^20` (about a million) features or more.

These techniques typically generate an enormous number of features.
Representing each feature as an individual column is far from ideal, both from
the perspective of how the user interacts with the information and how the
information is managed in the schematized system. The solution is to represent
each set of features, whether indicator values, or bag-of-words counts, as a
single vector-valued column.

A vector type specifies an item type and optional dimensionality information.
The item type must be a primitive, non-vector, type. The optional
dimensionality information specifies, at the basic level, the number of items
in the corresponding vector values.

When the size is unspecified, the vector type is variable-length, and
corresponding vector values may have any length. A tokenization transform,
that maps a text value to the sequence of individual terms in that text,
naturally produces variable-length vectors of text. Then, a hashing ngram
transform may map the variable-length vectors of text to a bag-of-ngrams
representation, which naturally produces numeric vectors of length `2^^k`,
where `k` is the number of bits used in the hash function.

### Key Types

The IDataView system includes the concept of key types. Key types are used for
data that is represented numerically, but where the order and/or magnitude of
the values is not semantically meaningful. For example, hash values, social
security numbers, and the index of a term in a dictionary are all best modeled
with a key type.

## Components

The IDataView system includes several standard kinds of components and the
ability to compose them to produce efficient data pipelines. A loader
represents a data source as an `IDataView`. A transform is applied to an
`IDataView` to produce a derived `IDataView`. A saver serializes the data
produced by an `IDataView` to a stream, in some cases in a format that can be
read by a loader. There are other more specific kinds of components defined
and used by the ML.NET code base, for example, scorers, evaluators, joins, and
caches. While there are several standard kinds of components, the set of
component kinds is open.

### Transforms

Transforms are a foundational kind of IDataView component. Transforms take an
IDataView as input and produce an IDataView as output. Many transforms simply
"add" one or more computed columns to their input schema. More precisely,
their output schema includes all the columns of the input schema, plus some
additional columns, whose values are computed from some of the input column
values. It is common for an added column to have the same name as an input
column, in which case, the added column hides the input column. Both the
original column and new column are present in the output schema and available
for downstream components (in particular, savers and diagnostic tools) to
inspect. For example, a normalization transform may, for each slot of a
vector-valued column named Features, apply an offset and scale factor and
bundle the results in a new vector-valued column, also named Features. From
the user's perspective (which is entirely based on column names), the Features
column was "modified" by the transform, but the original values are available
downstream via the hidden column.

Some transforms require training, meaning that their precise behavior is
determined automatically from some training data. For example, normalizers and
dictionary-based mappers, such as the TermTransform, build their state from
training data. Training occurs when the transform is instantiated from user-
provided parameters. Typically, the transform behavior is later serialized.
When deserialized, the transform is not retrained; its behavior is entirely
determined by the serialized information.

### Composition Examples

Multiple primitive transforms may be applied to achieve higher-level
semantics. For example, ML.NET's `CategoricalTransform` is the composition of
two more primitive transforms, `TermTransform`, which maps each term to a key
value via a dictionary, and `KeyToVectorTransform`, which maps from key value
to indicator vector. Similarly, `CategoricalHashTransform` is the composition
of `HashTransform`, which maps each term to a key value via hashing, and
`KeyToVectorTransform`.

Similarly, `WordBagTransform` and `WordHashBagTransform` are each the
composition of three transforms. `WordBagTransform` consists of
`WordTokenizeTransform`, `TermTransform`, and `NgramTransform`, while
`WordHashBagTransform` consists of `WordTokenizeTransform`, `HashTransform`,
and `NgramHashTransform`.

## Cursoring

### Row Cursor

To access the data in a view, one gets a row cursor from the view by calling
the `GetRowCursor` method. The row cursor is a movable window onto a single
row of the view, known as the current row. The row cursor provides the column
values of the current row. The `MoveNext()` method of the cursor advances to
the next row. There is also a `MoveMany(long count)` method, which is
semantically equivalent to calling `MoveNext()` repeatedly, `count` times.

Note that a row cursor is not thread safe; it should be used in a single
execution thread. However, multiple cursors can be active simultaneously on
the same or different threads.

### Lazy Computation and Active Columns

It is common in a data pipeline for a down-stream component to only require a
small subset of the information produced by the pipeline. For example, code
that needs to build a dictionary of all terms used in a particular text column
does not need to iterate over any other columns. Similarly, code to display
the first 100 rows does not need to iterate through all rows. When up-stream
computations are lazy, meaning that they are only performed when needed, these
scenarios execute significantly faster than when the up-stream computation is
eager (always performing all computations).

The IDataView system enables and encourages components to be lazy in both
column and row directions.

A row cursor has a set of active columns, determined by arguments passed to
`GetRowCursor`. Generally, the cursor, and any upstream components, will only
perform computation or data movement necessary to provide values of the active
columns. For example, when `TermTransform` builds its term dictionary from its
input `IDataView`, it gets a row cursor from the input view with only the term
column active. Any data loading or computation not required to materialize the
term column is avoided. This is lazy computation in the column direction.

Generally, creating a row cursor is a very cheap operation. The expense is in
the data movement and computation required to iterate over the rows. If a
cursor is used to iterate over a small subset of the input rows, then
generally, only computation and data movement needed to materialize the
requested rows is performed. This is lazy computation in the row direction.

### Immutability and Repeatability

Cursoring through data does not modify input data in any way. The root data is
immutable, and the operations performed to materialize derived data are
repeatable. In particular, the values produced by two cursors constructed from
the same view with the same arguments to `GetRowCursor` will be identical.

Immutability and repeatability enable transparent caching. For example, when a
learning algorithm or other component requires multiple passes over an
IDataView pipeline that includes non-trivial computation, performance may be
enhanced by either caching to memory or caching to disk. Immutability and
repeatability ensure that inserting caching is transparent to the learning
algorithm.

Immutability also ensures that execution of a composed data pipeline graph is
safe for parallelism. Without the guarantee of immutability, nodes in a data
flow graph can produce side effects that are visible to other non-dependent
nodes. A system where multiple transforms worked by mutating data would be
impossible to predict or reason about, short of the gross inefficiency of
cloning of the source data to ensure consistency.

The IDataView system's immutability guarantees enable flexible scheduling
without the need to clone data.

### Batch Parallel Cursoring

The `GetRowCursor` method on `IDataView` includes options to allow or
encourage parallel execution. If the view is a transform that can benefit from
parallelism, it requests from its input view, not just a cursor, but a cursor
set. If that view is a transform, it typically requests from its input view a
cursor set, etc., on up the transformation chain. At some point in the chain
(perhaps at a loader), a component, called the splitter, determines how many
cursors should be active, creates those cursors, and returns them together
with a consolidator object. At the other end, the consolidator is invoked to
marshal the multiple cursors back into a single cursor. Intervening levels
simply create a cursor on each input cursor, return that set of cursors as
well as the consolidator.

The ML.NET code base includes transform base classes that implement the
minimal amount of code required to support this batch parallel cursoring
design. Consequently, most transform implementations do not have any special
code to support batch parallel cursoring.

### Memory Efficiency

Cursoring is inherently efficient from a memory allocation perspective.
Executing `MoveNext()` requires no memory allocation. Retrieving primitive
column values from a cursor also requires no memory allocation. To retrieve
vector column values from a cursor, the caller can optionally provide buffers
into which the values should be copied. When the provided buffers are
sufficiently large, no additional memory allocation is required. When the
buffers are not provided or are too small, the cursor allocates buffers of
sufficient size to hold the values. This cooperative buffer sharing protocol
eliminates the need to allocate separate buffers for each row. To avoid any
allocation while iterating, client code only need allocate sufficiently large
buffers up front, outside the iteration loop.

Note that IDataView allows algorithms that need to materialize data in memory
to do so. Nothing in the system prevents a component from cursoring through
the source data and building a complete in-memory representation of the
information needed, subject, of course, to available memory.

### Data Size

For large data scenarios, it is critical that the pipeline support efficient
multiple pass "streaming" from disk. IDataView naturally supports streaming
via cursoring through views. Typically, the root of a view is a loader that
pulls information from a file or other data source. We have implemented both
binary .idv and text-based loaders and savers. New loaders and savers can be
added at any time.

Note that when the data is small, and repeated passes over the data are
needed, the operating system disk cache transparently enhances performance.
Further, when the data is known to fit in memory, caching, as described above,
provides even better performance.

### Randomization

Some training algorithms benefit from randomizing the order of rows produced
by a cursor. An `IDataView` indicates via a property whether it supports
shuffling. If it does, a random number generator passed to its `GetRowCursor`
method indicates shuffling should happen, with seed information pulled from
the random number generator. Serving rows from disk in a random order is quite
difficult to do efficiently (without seeking for each row). The binary .idv
loader has some shuffling support, favoring performance over attempting to
provide a uniform distribution over the permutation space. This level of
support has been validated to be sufficient for machine learning goals (for example,
in recent work on SA-SDCA algorithm). When the data is all in memory, as it is
when cached, randomizing is trivial.

## Appendix: Comparison with LINQ

This section is intended for developers familiar with the .Net
`IEnumerable<T>` interface and the LINQ technologies.

The `IDataView` interface is, in some sense, similar to `IEnumerable<T>`, and
the IDataView system is similar to the LINQ eco-system. The comparisons below
refer to the `IDataView` and `IEnumerable<T>` interfaces as the core
interfaces of their respective worlds.

In both worlds, there is a cursoring interface associated with the core
interface. In the IEnumerable world, the cursoring interface is
`IEnumerator<T>`. In the IDataView world, the cursoring interface is
`IRowCursor`.

Both cursoring interfaces have `MoveNext()` methods for forward-only iteration
through the elements.

Both cursoring interfaces provide access to information about the current
item. For the IEnumerable world, the access is through the `Current` property
of the enumerator. Note that when `T` is a class type, this suggests that each
item served requires memory allocation. In the IDataView world, there is no
single object that represents the current row. Instead, the values of the
current row are directly accessible via methods on the cursor. This avoids
memory allocation for each row.

In both worlds, the item type information is carried by both the core
interface and the cursoring interface. In the IEnumerable world, this type
information is part of the .Net type, while in the IDataView world, the type
information is much richer and contained in the schema, rather than in the
.Net type.

In both worlds, many different classes implement the core interface. In the
IEnumerable world, developers explicitly write some of these classes, but many
more implementing classes are automatically generated by the C# compiler, and
returned from methods written using the C# iterator functionality (`yield
return`). In the IDataView world, developers explicitly write all of the
implementing classes, including all loaders and transforms. Unfortunately,
there is no equivalent `yield return` magic.

In both worlds, multiple cursors can be created and used.

In both worlds, computation is naturally lazy in the row direction. In the
IEnumerable world, laziness in the column direction would correspond to the
returned `Current` value of type `T` lazily computing some of its properties.

In both worlds, streaming from disk is naturally supported.

Neither world supports indexed item access, nor a guarantee that the number of
items is available without iterating and counting.
