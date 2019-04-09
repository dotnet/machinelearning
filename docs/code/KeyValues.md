# Key Values

Most commonly, in key-valued data, each value takes one of a limited number of
distinct values. We might view them as being the enumeration into a set. They
are represented in memory using unsigned integers. Most commonly this is
`uint`, but `byte`, `ushort`, and `ulong` are possible values to use as well.

A more formal description of key values and types is
[here](IDataViewTypeSystem.md#key-types). *This* document's motivation is less
to describe what key types and values are, and more to instead describe why
key types are necessary and helpful things to have. Necessarily, this document,
is more anecdotal in its descriptions to motivate its content.

Let's take a few examples of transformers that produce keys:

* The `ValueToKeyMappingTransformer` has a dictionary of unique values
  obvserved when it was fit, each mapped to a key-value. The key type's count
  indicates the number of items in the set, and through the `KeyValue`
  annotation "remembers" what each key is representing.

* The `TokenizingByCharactersTransformer` will take input strings and produce
  key values representing the characters observed in the string. The
  `KeyValue` annotation "remembers" what each key is representing. (Note that
  unlike many other key-valued operations, this uses a representation type of
  `ushort` instead of `uint`.)

* The `HashingTransformer` performs a hash of input values, and produces a key
  value with count equal to the range of the hash function, which, if a `b`
  bit hash was used, will produce values with a key-type of count `2ᵇ` .

Note that in the first two cases, these are enumerating into a set with actual
specific values, whereas in the last case we are also enumerating into a set,
but one without values, since hashes don't intrinsically correspond to a
single item.

## Keys as Intermediate Values

Explicitly invoking transforms that produce key values, and using those key
values, is sometimes helpful. However, given that trainers typically expect
the feature vector to be a vector of floating point values and *not* keys, in
typical usage the majority of usages of keys is as some sort of intermediate
value on the way to that final feature vector. (Unless, say, doing something
like preparing labels for a multiclass trainer.)

So why not go directly to the feature vector from whatever the input was, and
forget this key stuff? Actually, to take text processing as the canonical
example, we used to. However, by structuring the transforms from, say, text to
key to vector, rather than text to vector *directly*, we were able to make a
more flexible pipeline, and re-use smaller, simpler components. Having
multiple composable transformers instead of one "omni-bus" transformer that
does everything makes the process easier to understand, maintain, and exploit
for novel purposes, while giving people greater visibility into the
composability of what actually happens.

So for example, the `TokenizingByCharactersTransformer` above might appear to
be a strange choice: *why* represent characters as keys? The reason is that
the ngram transform which often comes after it, is written to ingest keys, not
text, and so we can use the same transform for both the n-gram featurization
of words, as well as n-char grams.

Now, much of this complexity is hidden from the user: most users will just use
the text featurization transform, select some options for n-grams, and
chargrams, and not necessarily have to be aware of the usage of these internal
keys, at least. Similarly, this user can use the categorical or categorical
hash transforms, without knowing that internally it is just the term or hash
transform followed by a `KeyToVectorMappingTransformer`. But, keys are still
there, and it would be impossible to really understand ML.NET's featurization
pipeline without understanding keys. Any user that wants to debug how, say,
the text transform's multiple steps resulted in a particular featurization
will have to inspect the key values to get that understanding.

## The Representation of Keys

As an actual CLR data type, key values are stored as some form of unsigned
integer (most commonly `uint`, but the other unsigned integer types are legal
as well). One common confusion that arises from this is to ascribe too much
importance to the fact that it is a `uint`, and think these are somehow just
numbers. This is incorrect.

Most importantly, that the cardinality of the set they're enumerating is part
of the type is critical information. In an `IDataView`, these are represented
by the `KeyDataViewType` (or a vector of those types), with `RawType` being
one of the aforementioned .NET unsigned numeric types, and most critically
`Count` holding the cardinality of the set being represented. By encoding this
in the schema, one can tell in downstream `ITransformer`s.

For keys, the concept of order and difference has no inherent, real meaning as
it does for numbers, or at least, the meaning is different and highly domain
dependent. Consider a numeric `uint` type (specifically,
`NumberDataViewType.UInt32`), with values `0`, `1`, and `2`. The difference
between `0` and `1` is `1`, and the difference between `1` and `2` is `1`,
because they're numbers. Very well: now consider that you call
`ValueToKeyMappingEstimator.Fit` to get the transformer over the input tokens
`apple`, `pear`, and `orange`: this will also map to the keys physically
represented as the `uint`s `1`, `2`, and `3` respectively, which corresponds
to the logical ordinal indices of `0`, `1`, and `2`, again respectively.

Yet for a key, is the difference between the logical indices `0` and `1`, `1`?
No, the difference is `0` maps to `apple` and `1` to `pear`. Also order
doesn't mean one key is somehow "larger," it just sometimes means we saw one
before another -- or something else, if sorting by value happened to be
selected, or if the dictionary was constructed in some other fashion.

There's also the matter of default values. For key values, the default key
value should be the "missing" value for they key. So logically, `0` is the
missing value for any key type. The alternative is that the default value
would be whatever key value happened to correspond to the "first" key value,
which would be very strange and unnatural. Consider the `apple`, `pear`, and
`orange` example above -- it would be inappropriate for the default value to
be `apple`, since that's fairly arbitrary. Or, to extend this reasoning to
sparse `VBuffer`s, would it be appropriate for a sparse `VBuffer` of key to
have a value of `apple` for every implicit value? That doesn't make sense. So,
the default value is the missing value.

One of the more confusing consequences of this is that since, practically,
these key values are more often than not used as indices of one form or
another, and the first non-missing value is `1`, that in certain circumstances
like, say, writing out key values to text, that non-missing values will be
written out starting at `0`, even though physically they are stored starting
from the `1` value -- that is, the representation value for non-missing values
is written as the value minus `1`.

It may be tempting to think to avoid this by using nullables, for instance,
`uint?` instead of `uint`, since `default(uint?)` is `null`, a perfectly
intuitive missing value. However, since this has some performance and space
implications, and so many critical transformers use this as an intermediate
format for featurization, the decision was, that the performance gain we get
from not using nullables justified this modest bit of extra complexity. Note
however, that if you take a key-value with representation type `uint` and map
it to an `uint?` through operations like `MLContext.Data.CreateEnumerable`, it
will perform this more intuitive mapping.

## As an Enumeration of a Set: `KeyValues` Annotation

Since keys being an enumeration of some underlying set, there is often a
collection holding those items. This is expressed through the `KeyValues`
annotation kind. Note that this annotation is not part of the
`KeyDataViewType` structure itself, but rather the annotations of the column
with that type, as accessible through the `DataViewSchema.Column` extension
methods `HasKeyValues` and `GetKeyValues`.

Practically, the type of this is most often a vector of text. However, other
types are possible, and when `ValueToKeyMappingEstimator.Fit` is applied to an
input column with some item type, the resulting annotation type would be a
vector of that input item type. So if you were to apply it to a
`NumberDataViewType.Int32` column, you'd have a vector of
`NumberDataViewType.Int32` annotations.

How this annotation is used downstream depends on the purposes of who is
consuming it, but common uses are, in multiclass classification, for
determining the human readable class names, or if used in featurization,
determining the names of the features, or part of the names of the features.

Note that `KeyValues` kind annotation data is optional, since it is not always
sensible to have specific values in all cases where key values are
appropriate. For example, consider the output of the `k`-means clustering
algorithm. If there were five clusters, then the prediction would indicate the
cluster by a value with key-type of count five. Yet, there is no "value"
associated with each key.

Another example is hash based featurization: if you apply, say, a 10-bit hash,
you know you're enumerating into a set of 1024 values, so a key type is
appropriate. However, because it's a hash you don't have any particular
"original values" associated with it.
