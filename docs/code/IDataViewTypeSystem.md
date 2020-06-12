# `IDataView` Type System

## Overview

The *IDataView system* consists of a set of interfaces and classes that
provide efficient, compositional transformation of and cursoring through
schematized data, as required by many machine-learning and data analysis
applications. It is designed to gracefully and efficiently handle both
extremely high dimensional data and very large data sets. It does not directly
address distributed data, but is suitable for single node processing of data
partitions belonging to larger distributed data sets.

While `IDataView` is one interface in this system, colloquially, the term
IDataView is frequently used to refer to the entire system. In this document,
the specific interface is written using fixed pitch font as `IDataView`.

IDataView is the data pipeline machinery for ML.NET. The ML.NET codebase has
an extensive library of IDataView related components (loaders, transforms,
savers, trainers, predictors, etc.). More are being worked on.

The name IDataView was inspired from the database world, where the term table
typically indicates a mutable body of data, while a view is the result of a
query on one or more tables or views, and is generally immutable. Note that
both tables and views are schematized, being organized into typed columns and
rows conforming to the column types. Views differ from tables in several ways:

* Views are immutable; tables are mutable.

* Views are composable -- new views can be formed by applying transformations
  (queries) to other views. Forming a new table from an existing table
  involves copying data, making them decoupled—the new table is not linked to
  the original table in any way.

* Views are virtual; tables are fully realized/persisted.

Note that immutability and compositionality are critical enablers of
technologies that require reasoning over transformation, like query
optimization and remoting. Immutability is also key for concurrency and thread
safety.

This document includes a very brief introduction to some of the basic concepts
of IDataView, but then focuses primarily on the IDataView type system.

Why does IDataView need a special type system? The .NET type system is not
well suited to machine-learning and data analysis needs. For example, while
one could argue that `typeof(double[])` indicates a vector of double values,
it explicitly does not include the dimensionality of the vector/array.
Similarly, while enumeration over a set can be done with `enum`, there is no
concept of an enumeration over a set determined at runtime (for example, the
unique terms observed in a dataset). In short, there is no reasonable way to
encode complete range and dimensionality information in a `System.Type`.

In addition, a well-defined type system, including complete specification of
standard data types and conversions, enables separately authored components to
seamlessly work together without surprises.

### Basic Concepts

`IDataView`, in the narrow sense, is an interface implemented by many
components. At a high level, it is analogous to the .NET interface
`IEnumerable<T>`, with some very significant differences.

While `IEnumerable<T>` is a sequence of objects of type `T`, `IDataView` is a
sequence of rows. An `IDataView` object has an associated `DataViewSchema`
object that defines the `IDataView`'s columns, including their names, types,
indices, and associated annotations. Each row of the `IDataView` has a value
for each column defined by the schema.

Just as `IEnumerable<T>` has an associated enumerator interface, namely
`IEnumerator<T>`, `IDataView` has an associated cursor type, namely the
abstract class `DataViewRowCursor`. In the enumerable world, an enumerator
object implements a `Current` property that returns the current value of the
iteration as an object of type `T`. In the IDataView world, a
`DataViewRowCursor` object encapsulates the current row of the iteration.
There is no separate object that represents the current row. Instead, the
cursor implements methods that provide the values of the current row, when
requested. Additionally, the methods that serve up values do not require
memory allocation on each invocation, but use sharable buffers. This scheme
significantly reduces the memory allocations needed to cursor through data.

Both `IDataView` and `IEnumerable<T>` present a read-only view on data, in the
sense that a sequence presented by each is not directly mutable.
"Modifications" to the sequence are accomplished by additional operators or
transforms applied to the sequence, so do not modify any underlying data. For
example, to normalize a numeric column in an `IDataView` object, a
normalization transformer is applied using the `ITransformer.Transform` method
to form a new `IDataView` object representing the composition. In the new
view, the normalized values are contained in a new column. Often, the new
column has the same name as the original source column and "replaces" the
source column in the new view. Columns that are not involved in the
transformation are simply "passed through" from the source `IDataView` to the
new one.

Detailed specifications of the `IDataView`, `DataViewSchema`, and
`DataViewRowCursor` types are in other documents.

### IDataView Types

Each column in an `IDataView` has an associated type. The collection of types
in the IDataView system is open, in the sense that new code can introduce new
types without requiring modification of all `IDataView` related components.
While introducing new types is possible, we expect it will also be relatively
rare.

All column type implementations derive from the abstract class `DataViewType`.
Primitive column types are those whose implementation derives from the
abstract class `PrimitiveDataViewType`, which derives from `DataViewType`.

### Representation Type

A type in the IDataView system has an associated .NET type, known as its
representation type or raw type.

Note that a type in the IDataView system often contains much more information
than the associated .NET representation type. Moreover, many distinct types
can use the same representation type. Consequently, code should not assume
that a particular .NET type implies a particular type of the IDataView type
system. To give the most conspicuous example, `NumberDataViewType.UInt32` and
most instances of `KeyDataViewType` users will encounter, will both have
`ItemType == typeof(uint)`.

### Standard Column Types

There is a set of predefined standard types, divided into standard primitive
types and vector types. Note that there can be types that are neither
primitive nor vector types. These types are not standard types and may require
extra care when handling them. For example, a `ImageDataViewType` value might
require disposing when it is no longer needed.

Standard primitive types include the text type, the boolean type, numeric
types, and key types. Numeric types are further split into floating-point
types, signed integer types, and unsigned integer types.

A vector type has an associated item type that must be a primitive type, but
need not be a standard primitive type. Note that vector types are not
primitive types, so vectors of vectors are not supported. Note also that
vectors are homogeneous—all elements are of the same type. In addition to its
item type, a vector type contains dimensionality information. At the basic
level, this dimensionality information indicates the length of the vector
type. A length of zero means that the vector type is variable length, that is,
different values may have different lengths. Additional detail of vector types
is in a subsequent section. Vector types are instances of the sealed class
`VectorDataViewType`, which derives from `DataViewType`.

This document uses convenient shorthand for standard types:

* `TX`: text

* `BL`: boolean

* `R4`, `R8`: single and double precision floating-point

* `I1`, `I2`, `I4`, `I8`: signed integer types with the indicated number of
  bytes

* `U1`, `U2`, `U4`, `U8`: unsigned integer types with the indicated number of
  bytes

* `UG`: unsigned type with 16-bytes, typically used as a unique ID

* `TS`: timespan, a period of time

* `DT`: datetime, a date and time but no timezone

* `DZ`: datetime zone, a date and time with a timezone

* `U4[100]`: A key type based on `U4` representing logical non-missing values
  from 0 to 99, inclusive. It must be pointed out, that range is represented
  *physically* as 1 to 100, with the default value `0` corresponding to the
  missing value.

* `V<R4,3,2>`: A vector type with item type `R4` and dimensionality
  information `[3,2]`

See the sections on the specific types for more detail.

The IDataView system includes many standard conversions between standard
primitive types. A later section contains a full specification of these
conversions.

### Default Value

Each type in the IDataView system has an associated default value
corresponding to the default value of its representation type, as defined by
the .Net (C# and CLR) specifications.

The standard conversions map source default values to destination default
values. For example, the standard conversion from `TX` to `R8` maps the empty
text value to the value zero.

### Missing Value

Only floating-point types and key types have an internal representation of missing.
Other types don't support missing value. We follow R's lead and
denote the missing values as `NA`.

Unlike R, the floating-point types do not distinguish between missing and
invalid. For example, in floating-point arithmetic, computing zero divided by
zero, or infinity minus infinity, produces an invalid value known as a `NaN`
(for Not-a-Number). R uses a specific `NaN` value to represent its `NA` value,
with all other `NaN` values indicating invalid. The IDataView standard
floating-point types do not distinguish between the various `NaN` values,
treating them all as missing/invalid.

A standard conversion from a source type with `NA` to a destination type with
`NA` maps `NA` to `NA`. A standard conversion from a source type with `NA` to
a destination type without `NA` maps `NA` to the default value of the
destination type. For example, converting a `R4` to `R8` produces a `NaN`, but
converting an `R4` `NA` to `U4` results in zero. Note that this
specification does not address diagnostic user messages, so, in certain
environments, the latter situation may generate a warning to the user, or even
an exception.

Note that a vector type does not support a representation of missing, but may
contain `NA` values of its item type. Generally, there is no standard
mechanism faster than O(N) for determining whether a vector with N items
contains any missing values.

For further details on missing value representations, see the sections
detailing the particular standard primitive types.

### Vector Representations

Values of a vector type may be represented either sparsely or densely. A
vector type does not mandate denseness or sparsity, nor does it imply that one
is favored over the other. A sparse representation is semantically equivalent
to a dense representation having the suppressed entries filled in with the
*default* value of the item type. Note that the values of the suppressed
entries are emphatically *not* the missing/`NA` value of the item type, unless
the missing and default values are identical, as they are for key types.

### Annotations

A column in an `DataViewSchema` can have additional column-wide information,
known as annotations. For each string value, known as an annotation kind, a
column may have a value associated with that annotation kind. The value also
has an associated type, which is a compatible column type.

For example:

* A column may indicate that it is normalized, by providing a `BL` valued
  annotation named `IsNormalized`.

* A column whose type is `V<R4,17>`, meaning a vector of length 17 whose items
  are single-precision floating-point values, might have `SlotNames`
  annotation of type `V<TX,17>`, meaning a vector of length 17 whose items are
  text.

* A column produced by a scorer may have several pieces of associated
  annotations, indicating the "scoring column group id" that it belongs to,
  what kind of scorer produced the column (for example, binary
  classification), and the precise semantics of the column (for example,
  predicted label, raw score, probability).

The `DataViewSchema` class, including the annotations API, is fully specified in
another document.

## Text Type

The text type, denoted by the shorthand `TX`, represents text values. The
`TextDataViewType` class derives from `PrimitiveDataViewType` and has a single
instance, exposed as `TextDataViewType.Instance`. The representation type of
`TX` is an immutable `struct`, [`ReadOnlyMemory<char>`][ReadOnlyMemory]. Note
that the default value has a `ReadOnlyMemory<char>` represents an empty
sequence of characters.

[ReadOnlyMemory]: https://docs.microsoft.com/en-us/dotnet/api/system.readonlymemory-1?view=netstandard-2.1

In text processing transformations, it is very common to split text into
pieces. A key advantage of using `ReadOnlyMemory<char>` instead of
`System.String` for text values is that these splits require no memory
allocation -- the derived `ReadOnlyMemory<char>` references the same
underlying memory, most commonly backed by `System.String`, as the original
`ReadOnlyMemory<char>` does. Another reason that `System.String` is not ideal
for text is that we want the default value to be empty and not `null`. For
`System.String`, the default value is `null`, which would be a more natural
representation for `NA` than for empty text. There is no notion of `NA` text
for this type.

## Boolean Type

The standard boolean type, denoted by the shorthand `BL`, represents
true/false values. The `BooleanDataViewType` class derives from
`PrimitiveDataViewType` and has a single instance, exposed as
`BooleanDataViewType.Instance`. The representation type of `BL` is the
`System.Boolean` structure, and correspondingly takes values either `true` or
`false`.

The default value of `BL` is `false`, and it has no `NA` value.

There are standard conversions from `TX` to `BL`, and from `BL` to `TX`. There are standard
conversions from `BL` to all signed integer and floating point numeric types,
with `false` mapping to zero and `true` mapping to one.

## Number Types

The standard number types are all instances of the sealed class
`NumberDataViewType`, which is derived from `PrimitiveDataViewType`. There are
two standard floating-point types, four standard signed integer types, and
four standard unsigned integer types. Each of these is represented by a single
instance of `NumberDataViewType` and there are static properties of
`NumberDataViewType` to access each instance, whose names are the same as the
type names of their respresentation type in the BCL's `System` namespace. For
example, to test whether a variable type represents `I4` (which has the .NET
respresentation type `System.Int32`, or `int`), use the C# code `type ==
NumberType.Int32`.

Floating-point arithmetic has a well-deserved reputation for being
troublesome. This is primarily because it is imprecise, in the sense that the
result of most operations must be rounded to the nearest representable value.
This rounding means, among other side effects, that floating-point addition
and multiplication are not associate, nor satisfy the distributive property.

However, in many ways, floating-point arithmetic is the best-suited system for
arithmetic computation. For example, the IEEE 754 specification mandates
precise graceful overflow behavior—as results grow, they lose resolution in
the least significant digits, and eventually overflow to a special infinite
value. In contrast, when integer arithmetic overflows, the result is a non-
sense value. Trapping and handling integer overflow is expensive, both in
runtime and development costs.

The IDataView system supports integer numeric types mostly for data
interchange convenience, but we strongly discourage performing arithmetic on
those values without first converting to floating-point.

### Floating-point Types

The floating-point types, `R4` and `R8`, have representation types
`System.Single` and `System.Double`. Their default values are zero. Any `NaN`
is considered an `NA` value, with the specific `Single.NaN` and `Double.NaN`
values being the canonical `NA` values.

There are standard conversions from each floating-point type to the other
floating-point type. There are also standard conversions from text to each
floating-point type, from floating-point type to text types, and from each
integer type to each floating-point type.

### Signed Integer Types

The signed integer types, `I1`, `I2`, `I4`, and `I8`, have representation
types `Sytem.SByte`, `System.Int16`, `System.Int32`, and `System.Int64`. The
default value of each of these is zero.

There are standard conversions from each signed integer type to every other
signed integer type. There are also standard conversions from text to each
signed integer type, from each signed integer type to text, and from each
signed integer type to each floating-point type.

Note that we have not defined standard conversions from floating-point types
to signed integer types.

### Unsigned Integer Types

The unsigned integer types, `U1`, `U2`, `U4`, and `U8`, have representation
types `Sytem.Byte`, `System.UInt16`, `System.UInt32`, and `System.UInt64`,
respectively. The default value of each of these is zero. These types do not
have an `NA` value.

There are standard conversions from each unsigned integer type to every other
unsigned integer type. There are also standard conversions from text to each
unsigned integer type, each unsigned integer type to text, and from each unsigned
integer type to each floating-point type.

Note that we have not defined standard conversions from floating-point types
to unsigned integer types, or between signed integer types and unsigned
integer types.

## Key Types

Key types are used for data that is represented numerically, but that is
conceptually understood to be an enumeration into a known cardinality set. For
example, hash values, the index of a term in a dictionary, or other
categorical values, are all best modeled with a key type.

The representation type of a key type, also called its underlying type, must
be one of the standard four .Net unsigned integer types. The `NA` and default
values of a key type are the same value, namely the representational value
zero.

Key types are instances of the sealed class `KeyDataViewType`, which derives
from `PrimitiveDataViewType`. In addition to its underlying type, a key type
specifies a count value, from `1` to `ulong.MaxValue`, inclusive.

Regardless of the count value, the representational value zero always means
`NA`, and the representational value one is always the first valid value of
the key type.

Notes:

* The `Count` property returns the count, or cardinality of the key type. This
  is of type `ulong`. The legal representation values are from one up to and
  including `Count`. The `Count` is required to be representable in the
  underlying type, so, for example, the `Count` value of a key type based on
  `System.Byte` must not exceed `255`. As an example of the usefulness of the
  `Count` property, consider the `KeyToVectorMappingTransformer` transformer
  implemented as part of ML.NET. It maps from a key type value to an indicator
  vector. The length of the vector is the `Count` of the key type, which is
  required to be positive. For a key value of `k`, with `1 ≤ k ≤ Count`, the
  resulting vector has a value of one in the (`k-1`)th slot, and zero in all
  other slots. An `NA` value (with representation zero) is mapped to the all-
  zero vector of length `Count`.

* For a key type with positive `Count`, a representation value should be
  between `0` and `Count`, inclusive, with `0` meaning `NA`. When processing
  values from an untrusted source, it is best to guard against values bigger
  than `Count` and treat such values as equivalent to `NA`.

* The shorthand for a key type with representation type `U1`, and count `100`,
  is `U1[100]`. Recall that the representation values always start at 1 and
  extend up to `Count`, in this case `100`.

There are standard conversions from text to each key type. This conversion
parses the text as a standard non-negative integer value and honors the `Count`
value of the key type. If a parsed numeric value falls outside the range 
indicated by `Count`, or if the text is not parsable as a non-negative integer,
the result is `NA`.

There are standard conversions from one key type to another, provided:

* The source and destination key types have the same `Count` value.

* Either the number of bytes in the destination's underlying type is greater
  than the number of bytes in the source's underlying type, or the `Count`
  value is positive. In the latter case, the `Count` is necessarily less than
  2k, where k is the number of bits in the destination type's underlying type.
  For example, `U1[100]` and `U2[100]` can be converted in both directions.

## Vector Types

### Introduction

Vector types are one of the key innovations of the IDataView system and are
critical for high dimensional machine-learning applications.

For example, when processing text, it is common to hash all or parts of the
text and encode the resulting hash values, first as a key type, then as
indicator or bag vectors using the `KeyToVectorMappingTransformer`. Using a
`k`-bit hash produces a key type with `Count` equal to `2^^k`, and vectors of
the same length. It is common to use `20` or more hash bits, producing vectors
of length a million or more. The vectors are typically very sparse. In systems
that do not support vector-valued columns, each of these million or more
values is placed in a separate (sparse) column, leading to a massive explosion
of the column space. Most tabular systems are not designed to scale to
millions of columns, and the user experience also suffers when displaying such
data. Moreover, since the vectors are very sparse, placing each value in its
own column means that, when a row is being processed, each of those sparse
columns must be queried or scanned for its current value. Effectively the
sparse matrix of values has been needlessly transposed. This is very
inefficient when there are just a few (often one) non-zero entries among the
column values. Vector types solve these issues.

A vector type is an instance of the sealed `VectorDataViewType` class, which
derives from `DataViewType`. The vector type contains its `ItemType`, which
must be a `PrimitiveDataViewType`, and its dimensionality information. The
dimensionality information consists of one or more non-negative integer
values. The `VectorSize` is the product of the dimensions. A dimension value
of zero means that the true value of that dimension can vary from value to
value.

For example, tokenizing a text by splitting it into multiple terms generates a
vector of text of varying/unknown length. The result type shorthand is
`V<TX,*>`. Hashing this using `6` bits then produces the vector type
`V<U4[64],*>`. Applying the `KeyToVector` transform then produces the vector
type `V<R4,*,64>`. Each of these vector types has a `VectorSize` of zero,
indicating that the total number of slots varies, but the latter still has
potentially useful dimensionality information: the vector slots are
partitioned into an unknown number of runs of consecutive slots each of length
`64`.

As another example, consider an image data set. The data starts with a `TX`
column containing URLs for images. Applying an `ImageLoadingTransformer`
generates a column of a custom (non-standard) type, `Image<*,*,4>`, where
the asterisks indicate that the image dimensions are unknown. The last
dimension of `4` indicates that there are four channels in each pixel: the
three color components, plus the alpha channel. Applying an `ImageResizer`
transform scales and crops the images to a specified size, for example,
`100x100`, producing a type of `Image<100,100,4>`. Finally, applying a
`ImagePixelExtractor` transform (and specifying that the alpha channel should
be dropped), produces the vector type `V<R4,3,100,100>`. In this example, the
`ImagePixelExtractor` re-organized the color information into separate planes,
and divided each pixel value by 256 to get pixel values between zero and one.

### Equivalence

Note that two vector types are equivalent when they have equivalent item types
and have identical dimensionality information. We might however say that two
vector types with dimensions `{6}` and `{3,2}`, and the same item type, are
compatible, in the sense that `VBuffer` of length `6` are both legal values
for those types. To test for compatibility, a looser concept, one would check
the vector `Size` and `ItemType`s are the same.

### Representation Type

The representation type of a vector type is the struct `VBuffer<T>`, where `T`
is the representation type of the item type. For example, the representation
type of `V<R8,10>` is `VBuffer<double>`. When the vector type's `VectorSize`
is positive, each value of the type will have length equal to the
`VectorSize`.

The struct `VBuffer<T>` provides both dense and sparse representations and
encourages cooperative buffer sharing and reuse. A complete discussion of
`VBuffer<T>` and associated coding idioms is in the API documentation, a
document [dedicated to vectors specifically](VBufferCareFeeding.md), [as well
as the `IDataView` design principles](IDataViewDesignPrinciples.md).

These together include complete discussion of programming idioms, and
information on helper classes for building and manipulating vectors.

## Standard Conversions

The `IDataView` system includes the definition and implementation of many
standard conversions. Standard conversions are required to map source default
values to destination default values. When both the source type and
destination type have an `NA` value, the conversion must map `NA` to `NA`.
When the source type has an `NA` value, but the destination type does not, the
conversion must map `NA` to the default value of the destination type.

Most standard conversions are implemented by the singleton class `Conversions`
in the namespace `Microsoft.MachineLearning.Data.Conversion`. The standard
conversions are exposed by the `ConvertTransform`.

### From Text

There are standard conversions from `TX` to the standard primitive types,
`R4`, `R8`, `I1`, `I2`, `I4`, `I8`, `U1`, `U2`, `U4`, `U8`, and `BL`. For non-
empty `TX` values, these conversions use standard parsing of floating-point
and integer values. For `BL`, the mapping is case insensitive, maps text
values `{ true, yes, t, y, 1, +1, + }` to `true`, and maps the values
`{ false, no, f, n, 0, -1, - }` to `false`.

If parsing fails, the result is the `NA` value for floating-point, or an
exception being thrown.

These conversions are required to map empty text (the default value of `TX`)
to the default value of the destination, which is zero for all numeric types
and `false` for `BL`. This may seem unfortunate at first glance, but leads to
some nice invariants. For example, when loading a text file with sparse row
specifications, it's desirable for the result to be the same whether the row
is first processed entirely as `TX` values, then parsed, or processed directly
into numeric values, that is, parsing as the row is processed. In the latter
case, it is simple to map implicit items (suppressed due to sparsity) to zero.
In the former case, these items are first mapped to the empty text value. To
get the same result, we need empty text to map to zero. An exception to this
rule has been permitted in the TextLoader, where there's an option to load
empty `TX` fields as `NaN` for `R4` and `R8` fields, instead of using the default
conversion of empty `TX` to the numeric default `0`.

### To Text

There are standard conversions to `TX` from the standard primitive types,
`R4`, `R8`, `I1`, `I2`, `I4`, `I8`, `U1`, `U2`, `U4`, `U8`, `BL`, `TS`, `DT`, and `DZ`.
`R4` uses the G7 format and `R8` uses the G17 format. `BL` converts to "True" or "False".
`TS` uses the format "0:c". `DT` and `DZ` use the "0:o" format.
  
### Floating Point

There are standard conversions from `R4` to `R8` and from `R8` to `R4`. These
are the standard IEEE 754 conversions (using unbiased round-to-nearest in the
case of `R8` to `R4`).

### Signed Integer

There are standard conversions from each signed integer type to each other
signed integer type. These conversions map any other numeric value that fits
in the destination type to the corresponding value, and maps any numeric value
that does not fit in the destination type to `NA`. For example, when mapping
from `I1` to `I2`, the source `NA` value, namely 0x80, is mapped to the
destination `NA` value, namely 0x8000, and all other numeric values are mapped
as expected. When mapping from `I2` to `I1`, any value that is too large in
magnitude to fit in `I1`, such as 312, is mapped to `NA`, namely 0x80.

### Signed Integer to Floating Point

There are standard conversions from each signed integer type to each floating-
point type. These conversions map `NA` to `NA`, and map all other values
according to the IEEE 754 specification using unbiased round-to-nearest.

### Unsigned Integer

There are standard conversions from each unsigned integer type to each other
unsigned integer type. These conversions map any numeric value that fits in
the destination type to the corresponding value, and maps any numeric value
that does not fit in the destination type to zero. For example, when mapping
from `U2` to `U1`, any value that is too large in magnitude to fit in `U1`,
such as 312, is mapped to zero.

### Unsigned Integer to Floating Point

There are standard conversions from each unsigned integer type to each
floating-point type. These conversions map all values according to the IEEE
754 specification using unbiased round-to-nearest.

### Key Types

There are standard conversions from one key type to another, provided:

* The source and destination key types have the same `Count` values.

* Either the number of bytes in the destination's underlying type is greater
  than the number of bytes in the source's underlying type, or the `Count`
  value is positive. In the latter case, the `Count` is necessarily less than
  `2^^k`, where `k` is the number of bits in the destination type's underlying
  type. For example, for values key-typed as having `Count==100` can be
  converted to and from key-typed columns using both `byte` and `ushort`.

The conversion maps source representation values to the corresponding
destination representation values. There are no special cases, because of the
requirements above.

### Boolean to Numeric

There are standard conversions from `BL` to each of the signed integer and
floating point numeric. These map `DvBool.True` to one, `DvBool.False` to
zero, and `DvBool.NA` to the numeric type's `NA` value.

## Type Classes

This chapter contains information on the C# classes used to represent types in
the DataView system types. Since the IDataView type system is extensible this
list describes only the core data types.

### `DataViewType` Abstract Class

The IDataView system includes the abstract class `DataViewType`. This is the
base class for all types in the IDataView system. In the following notes, the
symbol `type` is a variable of type `DataViewType`. This abstract class has,
on itself, only one member:

* The `type.RawType` property indicates the representation type of the
  IDataView type. Its use should generally be restricted to constructing
  generic type and method instantiations. For example, `type.RawType ==
  typeof(uint)` is `true` for both `NumberDataViewType.UInt32` and some
  `KeyDataViewType` instances.

### `PrimitiveDataViewType` Abstract Class

The `PrimitiveDataViewType` abstract class derives from `DataViewType`
and is the base class of all primitive type implementations.

### `TextDataViewType` Sealed Class

The `TextDataViewType` sealed class derives from `PrimitiveDataViewType` and
is a singleton-class for the standard text type. The instance is exposed by
the static `TextDataViewType.Instance` property.

### `BooleanDataViewType` Sealed Class

The `BooleanDataViewType` sealed class derives from `PrimitiveDataViewType`
and is a singleton-class for the standard boolean type. The instance is
exposed by the static `BooleanDataViewType.Instance` property.

### `NumberDataViewType` Sealed Class

The `NumberDataViewType` sealed class derives from `PrimitiveDataViewType` and
exposes single instances of each of the standard numeric types, `Single`,
`Double`, `SByte`, `Int16`, `Int32`, `Int64`, `Byte`, `UInt16`, `UInt32`,
`UInt64`, each of which has the representation type from the .NET `System`
namespace as indicated by the name of the instance.

### `RowIdDataViewType` Sealed Class

The `RowIdDataViewType` sealed class derives from `PrimitiveDataViewType` and
is a singleton-class for a 16-byte row ID. The instance is exposed by the
static `BooleanType.Instance` property.

### `DateTimeDataViewType` Sealed Class

The `DateTimeDataViewType` sealed class derives from `PrimitiveDataViewType`
and is a singleton-class for the standard datetime type. The instance is
exposed by the static `DateTimeDataViewType.Instance` property.

### `DateTimeOffsetDataViewType` Sealed Class

The `DateTimeOffsetDataViewType` sealed class derives from
`PrimitiveDataViewType` and is a singleton-class for the standard datetime
timezone type. The instance is exposed by the static
`DateTimeOffsetDataViewType.Instance` property.

### `TimeSpanDataViewType` Sealed Class

The `TimeSpanDataViewType` sealed class derives from `PrimitiveDataViewType`
and is a singleton-class for the standard datetime timezone type. The instance
is exposed by the static `TimeSpanDataViewType.Instance` property.

### `KeyDataViewType` Sealed Class

The `KeyDataViewType` sealed class derives from `PrimitiveType` and instances
represent key types.

Note that two key types are considered equal iff their kind and count are the
same, though they may be different instances. (Unlike many of the
aforementioned types, which are based on singleton instances.)

### `VectorType` Sealed Class

The `VectorType` sealed class derives from `DataViewType` and instances
represent vector types. The item type is specified as the first parameter to
each constructor and the dimension information is inferred from the additional
parameters.

* The `Dimensions` property is an immutable array of `int` indicating the
  dimensions of the vector. This will be of length at least one. All dimension
  values are non-negative integers. A dimension value of zero indicates
  unknown (or variable) in that dimension.

* The `VectorSize` property returns the product of the dimensions.

* The inherited `Equals` method returns true if the two types have the same
  item type and the same dimension information.

* The inherited `SameSizeAndItemType(DataViewType other)` method returns true
  if `other` is a vector type with the same item type and the same
  `VectorSize` value.
