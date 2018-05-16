# `VBuffer` Care and Feeding

The `VBuffer` is ML.NET's central vector type, used throughout our data
pipeline and many other places to represent vectors of values. For example,
nearly all trainers accept feature vectors as `VBuffer<float>`.

## Technical `VBuffers`

A `VBuffer<T>` is a generic type that supports both dense and sparse vectors
over items of type `T`. This is the representation type for all
[`VectorType`](../public/IDataViewTypeSystem.md#vector-representations)
instances in the `IDataView` ecosystem. When an instance of this is passed to
a row cursor getter, the callee is free to take ownership of and re-use the
arrays (`Values` and `Indices`).

A `VBuffer<T>` is a struct, and has the following `readonly` fields:

* `int Length`: The logical length of the buffer.

* `int Count`: The number of items explicitly represented. This equals `Length`
when the representation is dense and is less than `Length` when sparse.

* `T[] Values`: The values. Only the first `Count` of these are valid.

* `int[] Indices`: The indices. For a dense representation, this array is not
  used, and may be `null`. For a sparse representation it is parallel to
  values and specifies the logical indices for the corresponding values. Only
  the first `Count` of these are valid.

`Values` must have length equal to at least `Count`. If the representation is
sparse, that is, `Count < Length`, then `Indices` must have length also
greater than or equal to `Count`. If `Count == 0`, then it is entirely legal
for `Values` or `Indices` to be `null`, and if dense then `Indices` can always
be `null`.

On the subject of `Count == 0`, note that having no valid values in `Indices`
and `Values` merely means that no values are explicitly defined, and the
vector should be treated, logically, as being filled with `default(T)`.

For sparse vectors, `Indices` must have length equal to at least `Count`, and
the first `Count` indices must be increasing, with all indices between `0`
inclusive and `Length` exclusive.

Regarding the generic type parameter `T`, the only real assumption made about
this type is that assignment (that is, using `=`) is sufficient to create an
*independent* copy of that item. All representation types of the
[primitive types](../public/IDataViewTypeSystem.md#standard-column-types) have
this property (e.g., `DvText`, `DvInt4`, `Single`, `Double`, etc.), but for
example, `VBuffer<>` itself does not have this property. So, no `VBuffer` of
`VBuffer`s for you.

## Sparse Values as `default(T)`

Any implicit value in a sparse `VBuffer<T>` **must** logically be treated as
though it has value `default(T)`. For example, suppose we have the following
two declarations:

```csharp
var a = new VBuffer<float>(5, new float[] { 0, 1, 0, 0, 2 });
var b = new VBuffer<float>(5, 2, new float[] { 1, 2 }, new int[] { 1, 4 });
```

Here, `a` is dense, and `b` is sparse. However, any operations over either
must treat the logical indices `0`, `2`, and `3` as if they have value `0.0f`.
The two should be equivalent!

ML.NET throughout its codebase assumes in many places that sparse and dense
representations are interchangeable: if it is more efficient to consider
something sparse or dense, the code will have no qualms about making that
conversion. This does mean though, that we depend upon all code that deals
with `VBuffer` responding in the same fashion, and respecting this convention.

As a corollary to the above note about equivalence of sparse and dense
representations, since they are equivalent it follows that any code consuming
`VBuffer`s must work equally well with *both*. That is, there must never be a
condition where data is read and assumed to be either sparse, or dense, since
implementors of `IDataView` and related interfaces are perfectly free to
produce either.

The only "exception" to this rule is a necessary acknowledgment of the reality
of floating point mathematics: sometimes due to the way the JIT will optimize
code one code path or another, and due to the fact that floating point math is
not commutative, operations over sparse `VBuffer<float>` or `VBuffer<double>`
vectors can sometimes result in modestly different results than the "same"
operation over dense values.

## Why Buffer Reuse

The question is often asked by people new to this codebase: why bother with
buffer reuse at all? Without going into too many details, we used to not and
suffered for it. We had a far simpler system where examples were yielded
through an
[`IEnumerable<>`](https://msdn.microsoft.com/en-us/library/9eekhta0.aspx), and
our vector type at the time had `Indices` and `Values` arrays as well, but
their sizes were there actual sizes, and being returned through an
`IEnumerable<>` there was no plausible way to "recycle" the buffers.

Also: who "owned" a fetched example (the caller, or callee) was not clear.
Because it was not clear, code was inevitably written and checked in that made
*either* assumption, which meant, ultimately, that everything that touched
these would try to duplicate everything by default, because doing anything
else would fail in some case.

The reason why this becomes important is because [garbage
collection](https://msdn.microsoft.com/en-us/library/0xy59wtx.aspx) in the
.NET framework is not free. Creating and destroying these arrays *can* be
cheap, provided that they are sufficiently small, short lived, and only ever
exist in a single thread. But, violate any of these, there is a possibility
these arrays could be allocated on the large object heap, or promoted to gen-2
collection. The results could be disastrous: in one particularly memorable
incident regarding neural net training, the move to `IDataView` and its
`VBuffer`s resulted in a more than tenfold decrease in runtime performance,
because under the old regime the garbage collection of the feature vectors was
just taking so much time.

This is somewhat unfortunate: a joke-that's-not-really-a-joke on the team was
that we were writing C# as though it were C code. Be that as it may, buffer
reuse is essential to our performance, especially on larger problems.

This design requirement of buffer reuse has deeper implications for the
ecosystem merely than the type here. For example, it is one crucial reason why
so many value accessors in the `IDataView` ecosystem fill in values passed in
through a `ref` parameter, rather than, say, being a return value.

## Buffer Re-use as a User

Let's imagine we have an `IDataView` in a variable `dataview`, and we just so
happen to know that the column with index 5 has representation type
`VBuffer<float>`. (In real code, this would presumably we achieved through
more complicated involving an inspection of `dataview.Schema`, but we omit
such details here.)

```csharp
using (IRowCursor cursor = dataview.GetRowCursor(col => col == 5))
{
    ValueGetter<VBuffer<float>> getter = cursor.GetGetter<VBuffer<float>>(5);
    var value = default(VBuffer<float>);
    while (cursor.MoveNext())
    {
        getter(ref value);
        // Presumably something else is done with value.
    }
}
```

In this example, we open a cursor (telling it to make only column 5 active),
then get the "getter" over this column. What enables buffer re-use for this is
that, as we go row by row over the data with the `while` loop, we pass in the
same `value` variable in to the `getter` delegate, again and again. Presumably
the first time, or several, memory is allocated. Initially `value =
default(VBuffer<float>)`, that is, it has zero `Length` and `Count` and `null`
`Indices` and `Values`. Presumably at some point, probably the first call,
`value` is replaced with a `VBuffer<float>` that has actual values allocated.
In subsequent calls, perhaps these are judged as insufficiently large, and new
arrays are allocated, but we would expect at some point the arrays would
become "large enough" to accommodate many values, so reallocations would
become increasingly rare.

A common mistake made by first time users is to do something like move the
`var value` declaration inside the `while` loop, thus dooming `getter` to have
to allocate the arrays every single time, completely defeating the purpose of
buffer reuse.

## Buffer Re-use as a Developer

Nearly all methods in ML.NET that "return" a `VBuffer<T>` do not really return
a `VBuffer<T>` *at all*, but instead have a parameter `ref VBuffer<T> dst`,
where they are expected to put the result. See the above example, with the
`getter`. A `ValueGetter` is defined:

```csharp
public delegate void ValueGetter<TValue>(ref TValue value);
```

Let's describe the typical practice of "returning" a `VBuffer` in, say, a
`ref` parameter named `dst`: if `dst.Indices` and `dst.Values` are
sufficiently large to contain the result, they are used, and the value is
calculated, or sometimes copied, into them. If either is insufficiently large,
then a new array is allocated in its place. After all the calculation happens,
a *new* `VBuffer` is constructed and assigned to `dst`. (And possibly, if they
were large enough, using the same `Indices` and `Values` arrays as were passed
in, albeit with different values.)

`VBuffer`s can be either sparse or dense. However, even when returning a dense
`VBuffer`, you would not discard the `Indices` array of the passed in buffer,
assuming there was one. The `Indices` array was merely larger than necessary
to store *this* result: that you happened to not need it this call does not
justify throwing it away. We don't care about buffer re-use just for a single
call, after all! The dense constructor for the `VBuffer` accepts an `Indices`
array for precisely this reason!

Also note: when you return a `VBuffer` in this fashion, the caller is assumed
to *own* it at that point. This means they can do whatever they like to it,
like pass the same variable into some other getter, or modify its values.
Indeed, this is quite common: normalizers in ML.NET get values from their
source, then immediately scale the contents of `Values` appropriately. This
would hardly be possible if the callee was considered to have some stake in
that result.

There is a corollary on this point: because the caller owns any `VBuffer`,
then you shouldn't do anything that irrevocably destroys their usefulness to
the caller. For example, consider this method that takes a vector `src`, and
stores the scaled result in `dst`.

```csharp
VectorUtils.ScaleBy(ref VBuffer<float> src, ref VBuffer<float> dst, float c)
```

What this does is, copy the values from `src` to `dst`, while scaling each
value seen by `c`.

One possible alternate (wrong) implementation of this would be to just say
`dst=src` then scale all contents of `dst.Values` by `c`. But, then `dst` and
`src` would share references to their internal arrays, completely compromising
the callers ability to do anything useful with them: if the caller were to
pass `dst` into some other method that modified it, this could easily
(silently!) modify the contents of `src`. The point is: if you are writing
code *anywhere* whose end result is that two distinct `VBuffer` structs share
references to their internal arrays, you've almost certainly introduced a
**nasty** pernicious bug for your users.

## Utilities for Working with `VBuffer`s

ML.NET's runtime code has a number of utilities for operating over `VBuffer`s
that we have written to be generally useful. We will not treat on these in
detail here, but:

* `Microsoft.ML.Runtime.Data.VBuffer<T>` itself contains a few methods for
  accessing and iterating over its values.

* `Microsoft.ML.Runtime.Internal.Utilities.VBufferUtils` contains utilities
  mainly for non-numeric manipulation of `VBuffer`s.

* `Microsoft.ML.Runtime.Numeric.VectorUtils` contains math operations
  over `VBuffer<float>` and `float[]`, like computing norms, dot-products, and
  whatnot.

* `Microsoft.ML.Runtime.Data.BufferBuilder<T>` is an abstract class whose
  concrete implementations are used throughout ML.NET to build up `VBuffer<T>`
  instances. Note that if one *can* simply build a `VBuffer` oneself easily
  and do not need the niceties provided by the buffer builder, you should
  probably just do it yourself.

* `Microsoft.MachineLearning.Internal.Utilities.EnsureSize` is often useful to
ensure that the arrays are of the right size.

## Golden Rules

Here are some golden rules to remember:

Remember the conditions under which `Indices` and `Values` can be `null`! A
developer forgetting that `null` values for these fields are legal is probably
the most common error in our code. (And unfortunately one that sometimes takes
a while to pop up: most users don't feed in empty inputs to our trainers.)

In terms of accessing anything in `Values` or `Indices`, remember, treat
`Count` as the real length of these arrays, not the actual length of the
arrays.

If you write code that results in two distinct `VBuffer`s sharing references
to their internal arrays, (e.g., there are two `VBuffer`s `a` and `b`, with
`a.Indices == b.Indices` with `a.Indices != null`, or `a.Values == b.Values`
with `a.Values != null`) then you've almost certainly done something wrong.

Structure your code so that `VBuffer`s have their buffers re-used as much as
possible. If you have code called repeatedly where you are passing in some
`default(VBuffer<T>)`, there's almost certainly an opportunity there.

When re-using a `VBuffer` that's been passed to you, remember that even when
constructing a dense vector, you should still re-use the `Indices` array that
was passed in.