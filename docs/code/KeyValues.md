# Key Values

Most commonly, key-values are used to encode items where it is convenient or
efficient to represent values using numbers, but you want to maintain the
logical "idea" that these numbers are keys indexing some underlying, implicit
set of values, in a way more explicit than simply mapping to a number would
allow you to do.

A more formal description of key values and types is
[here](IDataViewTypeSystem.md#key-types). *This* document's motivation is less
to describe what key types and values are, and more to instead describe why
key types are necessary and helpful things to have. Necessarily, this document,
is more anecdotal in its descriptions to motivate its content.

Let's take a few examples of transforms that produce keys:

* The `TermTransform` forms a dictionary of unique observed values to a key.
  The key type's count indicates the number of items in the set, and through
  the `KeyValue` metadata "remembers" what each key is representing.

* The `HashTransform` performs a hash of input values, and produces a key
  value with count equal to the range of the hash function, which, if a b bit
  hash was used, will produce a 2ᵇ hash.

* The `CharTokenizeTransform` will take input strings and produce key values
  representing the characters observed in the string.

## Keys as Intermediate Values

Explicitly invoking transforms that produce key values, and using those key
values, is sometimes helpful. However, given that most trainers expect the
feature vector to be a vector of floating point values and *not* keys, in
typical usage the majority of usages of keys is as some sort of intermediate
value on the way to that final feature vector. (Unless, say, doing something
like preparing labels for a multiclass learner or somesuch.)

So why not go directly to the feature vector, and forget this key stuff?
Actually, to take text as the canonical example, we used to. However, by
structuring the transforms from, say, text to key to vector, rather than text
to vector *directly*, we are able to simplify a lot of code on the
implementation side, which is both less for us to maintain, and also for users
gives consistency in behavior.

So for example, the `charTokenize` above might appear to be a strange choice:
*why* represent characters as keys? The reason is that the N-gram transform is
written to ingest keys, not text, and so we can use the same transform for
both the n-gram featurization of words, as well as n-char grams.

Now, much of this complexity is hidden from the user: most users will just use
the `text` transform, select some options for n-grams, and n-char grams, and
not be aware of these internal invisible keys. Similarly, use the categorical
or categorical hash transforms, without knowing that internally it is just the
term or hash transform followed by a `KeyToVector` transform. But, keys are
still there, and it would be impossible to really understand ML.NET's
featurization pipeline without understanding keys. Any user that wants to
understand how, say, the text transform resulted in a particular featurization
will have to inspect the key values to get that understanding.

## Keys are not Numbers

As an actual CLR data type, key values are stored as some form of unsigned
integer (most commonly `uint`). The most common confusion that arises from
this is to ascribe too much importance to the fact that it is a `uint`, and
think these are somehow just numbers. This is incorrect.

For keys, the concept of order and difference has no inherent, real meaning as
it does for numbers, or at least, the meaning is different and highly domain
dependent. Consider a numeric `U4` type, with values `0`, `1`, and `2`. The
difference between `0` and `1` is `1`, and the difference between `1` and `2`
is `1`, because they're numbers. Very well: now consider that you train a term
transform over the input tokens `apple`, `pear`, and `orange`: this will also
map to the keys logically represented as the numbers `0`, `1`, and `2`
respectively. Yet for a key, is the difference between keys `0` and `1`, `1`?
No, the difference is `0` maps to `apple` and `1` to `pear`. Also order
doesn't mean one key is somehow "larger," it just means we saw one before
another -- or something else, if sorting by value happened to be selected.

Also: ML.NET's vectors can be sparse. Implicit entries in a sparse vector are
assumed to have the `default` value for that type -- that is, implicit values
for numeric types will be zero. But what would be the implicit default value
for a key value be? Take the `apple`, `pear`, and `orange` example above -- it
would inappropriate for the default value to be `0`, because that means the
result is `apple`, would be appropriate. The only really appropriate "default"
choice is that the value is unknown, that is, missing.

An implication of this is that there is a distinction between the logical
value of a key-value, and the actual physical value of the value in the
underlying type. This will be covered more later.

## As an Enumeration of a Set: `KeyValues` Metadata

While keys can be used for many purposes, they are often used to enumerate
items from some underlying set. In order to map keys back to this original
set, many transform producing key values will also produce `KeyValues`
metadata associated with that output column.

Valid `KeyValues` metadata is a vector of length equal to the count of the
type of the column. This can be of varying types: it is often text, but does
not need to be. For example, a `term` applied to a column would have
`KeyValue` metadata of item type equal to the item type of the input data.

How this metadata is used downstream depends on the purposes of who is
consuming it, but common uses are: in multiclass classification, for
determining the human readable class names, or if used in featurization,
determining the names of the features.

Note that `KeyValues` data is optional, and sometimes is not even sensible.
For example, if we consider a clustering algorithm, the prediction of the
cluster of an example would. So for example, if there were five clusters, then
the prediction would indicate the cluster by `U4<0-4>`. Yet, these clusters
were found by the algorithm itself, and they have no natural descriptions.

## Actual Implementation

This may be of use only to writers or extenders of ML.NET, or users of our
API. How key values are presented *logically* to users of ML.NET, is distinct
from how they are actually stored *physically* in actual memory, both in
ML.NET source and through the API. For key values:

* All key values are stored in unsigned integers.
* The missing key values is always stored as `0`. See the note above about the
  default value, to see why this must be so.
* Valid non-missing key values are stored from `1`, onwards, irrespective of
whatever we claim in the key type that minimum value is.

So when, in the prior example, the term transform would map `apple`, `pear`,
and `orange` seemingly to `0`, `1`, and `2`, values of `U4<0-2>`, in reality,
if you were to fire up the debugger you would see that they were stored with
`1`, `2`, and `3`, with unrecognized values being mapped to the "default"
missing value of `0`.

Nevertheless, we almost never talk about this, no more than we would talk
about our "strings" really being implemented as string slices: this is purely
an implementation detail, relevant only to people working with key values at
the source level. To a regular non-API user of ML.NET, key values appear
*externally* to be simply values, just as strings appear to be simply strings,
and so forth.

There is another implication: a hypothetical type `U1<4000-4002>` is actually
a sensible type in this scheme. The `U1` indicates that is is stored in one
byte, which would on first glance seem to conflict with values like `4000`,
but remember that the first valid key-value is stored as `1`, and we've
identified the valid range as spanning the three values 4000 through 4002.
That is, `4000` would be represented physically as `1`.

The reality cannot be seen by any conventional means I am aware of, save for
viewing ML.NET's workings in the debugger or using the API and inspecting
these raw values yourself: that `4000` you would see is really stored as the
`byte` `1`, `4001` as `2`, `4002` as `3`, and the missing `�` stored as `0`.
`4001` as `2`.