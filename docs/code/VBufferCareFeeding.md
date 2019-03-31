# `VBuffer` Care and Feeding

The `VBuffer` is ML.NET's central vector type, used throughout our data
pipeline and many other places to represent vectors of values. For example,
nearly all trainers accept feature vectors as `VBuffer<float>`.

## Technical `VBuffers`

A `VBuffer<T>` is a generic type that supports both dense and sparse vectors
over items of type `T`. This is the representation type for all
[`VectorType`](IDataViewTypeSystem.md#vector-representations) instances in the
`IDataView` ecosystem. When an instance of this is passed to a row cursor
getter, the callee is free to take ownership of and re-use the buffers, which
internally are arrays, and accessed externally as `ReadOnlySpan`s through the
`GetValues` and `GetIndices` methods.

A `VBuffer<T>` is a struct, and has the following most important members:

* `int Length`: The logical length of the buffer.

* `bool IsDense`: Whether the vector is dense (if `true`) or sparse (if
  `false`).

* `ReadOnlySpan<T> GetValues()`: The values. If `IsDense` is `true`, then this
  is of length exactly equal to `Length`. Otherwise, it will have length less
  than `Length`.

* `ReadOnlySpan<int> GetIndices()`: For a dense representation, this span will
  have length of `0`, but for a sparse representation, this will be of the
  same length as the span returned from `GetValues()`, and be parallel to it.
  All indices must be between `0` inclusive and `Length` exclusive, and all
  values in this span should be in strictly increasing order.

Regarding the generic type parameter `T`, the basic assumption made about this
type is that assignment (that is, using `=`) is sufficient to create an
*independent* copy of that  item. All representation types of the [primitive
types](IDataViewTypeSystem.md#standard-column-types) have this property (for
example, `ReadOnlyMemory<char>`, `int`, `float`, `double`, etc.), but for
example, `VBuffer<>` itself does not have this property. So, no `VBuffer` of
`VBuffer`s is possible.

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

There is also another point about sparsity, that is, about the increasing
indices. While it is a *requirement*, it is regrettably not one we can
practically enforce without a heavy performance cost. So, if you are creating
your own `VBuffer`s, be careful that you are creating indices that are in
strictly increasing order, and in the right range (non-negative and less than
`Length`). We cannot check, but we *do* rely on it!

# Buffer Reuse

The question is often asked by people new to this codebase: why bother with
buffer reuse at all? Without going into too many details, we used to not and
suffered for it. Long ago we had a far simpler system where examples were
yielded through an [`IEnumerable<>`][1], and our vector type at the time had
`Indices` and `Values`, but exposed as arrays, and their sizes were the actual
"logical" sizes, and being returned through an `IEnumerable<>` there was no
plausible way to reuse buffers.

Also: who "owned" a fetched example (the caller, or callee) was not clear;
code was at one time written under both assumptions, which was a mess. Because
it was not clear, code was inevitably written and checked in that made
*either* assumption, which meant, ultimately, that everything that touched
these would try to duplicate everything, because doing anything else would
fail in *some* case.

The reason why this becomes important is because [garbage collection][2] in
.NET is not free. Creating and destroying these arrays *can* be cheap,
provided that they are sufficiently small, short lived, and only ever exist in
a single thread. But, when that does not hold, there is a possibility these
arrays could be allocated on the large object heap, or promoted to gen-2
collection. The results then are sometimes disastrous: in one particularly
memorable incident regarding neural net training, the move to `IDataView` and
its `VBuffer`s resulted in a more than tenfold decrease in runtime
performance, because under the old regime the garbage collection of the
feature vectors was just taking so much time.

This design requirement of buffer reuse has deeper implications for the
ecosystem merely than the type here. For example, it is one crucial reason why
so many value accessors in the `IDataView` ecosystem fill in values passed in
through a `ref` parameter, rather than, say, being a return value.

[1]: https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.ienumerable-1?view=netstandard-2.0
[2]: https://docs.microsoft.com/en-us/dotnet/standard/garbage-collection/index

## Buffer Re-use as a User

Let's imagine we have an `IDataView` in a variable `data`, and we just so
happen to know that the column with index 5 has representation type
`VBuffer<float>`. (In real code, this would presumably we achieved through
more complicated involving an inspection of `data.Schema`, but we omit such
details here.)

```csharp
using (DataViewRowCursor cursor = data.GetRowCursor(data.Schema[5]))
{
    ValueGetter<VBuffer<float>> getter = cursor.GetGetter<VBuffer<float>>(5);
    VBuffer<float> value = default;
    while (cursor.MoveNext())
    {
        getter(ref value);
        // Presumably something else is done with value.
    }
}
```

To be explicit, a `ValueGetter` is defined this way:

```csharp
public delegate void ValueGetter<TValue>(ref TValue value);
```

Anyway, in this example, we open a cursor (telling it to make only column 5
active), then get the "getter" over this column. What enables buffer re-use
for this is that, as we go row by row over the data with the `while` loop, we
pass in the same `value` variable in to the `getter` delegate, again and
again. Presumably the first time, or several, memory is allocated (though,
internally, and invisibly to us). Initially `VBuffer<float> value = default`,
that is, it has zero `Length` and similarly empty spans. Presumably at some
point, probably the first call, `value` is replaced with a `VBuffer<float>`
that has actual values, stored in freshly allocated buffers. In subsequent
calls, *perhaps* these are judged as insufficiently large, and new arrays are
internally allocated, but we would expect at some point the arrays would
become "large enough," and we would no longer have to allocate new buffers and
garbage collect old ones after that point.

A common mistake made by first time users is to do something like move the
`var value` declaration inside the `while` loop, thus dooming `getter` to have
to allocate the arrays every single time, completely defeating the purpose of
buffer reuse.

##  Buffer Re-use as a Developer

So we've seen what it looks like from the user's point of view, but some words
on what a (well implemented) `getter` delegate is doing is also worthwhile, to
understand what is going on here.

`VBuffer<T>` by itself may appear to be a strictly immutable structure, but
"buffer reuse" implies it is in some way mutable. This mutability is primarily
accomplished with the `VBufferEditor`. The `VBufferEditor<T>` actually
strongly resembles a `VBuffer<T>`, *except* instead of being `ReadOnlySpan`
for values and indices they are `Span`, so, it is mutable.

First, one creates the `VBufferEditor<T>` structure using one of the
`VBufferEditor.Create` methods. You pass in a `VBuffer<T>` as a `ref`
parameter. (For this purpose, it is useful to remember that a
`default(VBuffer<T>)` is itself a completely valid, though completely empty,
`VBuffer<T>`.) The editor is, past that statement, considered to "own" the
internal structure of the `VBuffer<T>` you passed in. (So, correct code should
not continue to use the input `VBuffer<T>` structure from that point onwards.)

Then, one puts new values into `Span<T>` and, if in a sparse vector is the
desired result, puts new indices into `Span<int>`, on that editor, in whatever
way is appropriate and necessary for the task.

Last, one calls one of the `Commit` methods to get another `VBuffer`, with the
values and indices accessible through the `VBuffer<T>` methods the same as
those that had been set in the editor structure.

Similar to how creating a `VBufferEditor<T>` out of a `VBuffer<T>` renders the
passed in `VBuffer<T>` invalid, likewise, getting the `VBuffer<T>` out of the
editor out of the `VBufferEditor<T>` through one of the `Commit` methods
renders the *editor* invalid. In both cases, "ownership" of the internal
buffers is passed along to the successor structure, rendering the original
structure invalid, in some sense.

Internally, these buffers are backed by arrays that are reallocated *if
needed* by the editor upon its creation, but reused if they are large enough.

## Ownership of `VBuffer`

Nonetheless, this does not mean that bugs are impossible, because the concept
of buffer reuse does carry the implication that we must be quite clear about
who "owns" a `VBuffer` at any given time, for which we have developed this
convention:

Unless clearly specified in documentation, a caller is assumed to always own a
`VBuffer` as returned by `ref`. This means they can do whatever they like to
it, like pass the same variable into some other getter, or modify its values.
Indeed, this is quite common: normalizers in ML.NET get values from their
source, then immediately scale the contents of `Values` appropriately. This
would hardly be possible if the callee was considered to have some stake in
that result.

The `ValueGetter` indicated above is the most important example of this
principle. By passing in an existing `VBuffer<T>` into that delegate by
reference, they are giving the implementation control over it to use (or,
reallocate) as necessary to store the resulting value. But, once the delegate
returns with the value returned, the caller is now considered to own that
buffer.

There is a corollary on this point: because the caller owns any `VBuffer`,
then code should not do anything that irrevocably destroys its usefulness to
the caller. For example, consider this method that takes a vector `source`,
and stores the scaled result in `destination`.

```csharp
ScaleBy(in VBuffer<float> source, ref VBuffer<float> destination, float c)
```

What this does is, copy the values from `source` to `destination` while
scaling the values we are copying by the factor `c`.

One possible alternate (wrong) implementation of this would be to just say
`destination=source` then edit `destination` and scale each value by `c`. But,
then `destination` and `source` would share references to their internal
arrays, completely compromising the caller's ability to do anything useful
with `source`: if the caller were to pass `destination` into some other method
that modified it, this could easily (silently!) modify the contents of
`source`. The point is: if you are writing code *anywhere* whose end result is
that two distinct `VBuffer`s share references to their internal arrays, you've
almost certainly introduced a **nasty** pernicious bug.

# Internal Utilities for Working with `VBuffer`s

In addition to the public utilities around `VBuffer` and `VBufferEditor`
already mentioned, ML.NET's internal infrastructure and implementation code
has a number of utilities for operating over `VBuffer`s that we have written
to be generally useful. We will not treat on these in detail here since they
are internal and not part of the public API, but:

* `VBufferUtils` contains utilities
  mainly for non-numeric manipulation of `VBuffer`s.

* `VectorUtils` contains math operations over `VBuffer<float>` and `float[]`,
  like computing norms, dot-products, and whatnot.

* `BufferBuilder<T>` is a class for iteratively building up `VBuffer<T>`
  instances. Note that if one *can* simply build a `VBuffer` oneself easily
  and do not need the niceties provided by the buffer builder, you should
  probably just do it yourself, since there is a great deal of additional
  facilities here for combining multiple results at the same indices that many
  applications will not need.
