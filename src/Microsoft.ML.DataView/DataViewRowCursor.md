# `DataViewRowCursor` Notes

This document includes some more in depth notes on some expert topics for
`DataViewRow` and `DataViewRowCursor` derived classes.

## `Batch`

Multiple cursors can be returned through a method like
`IDataView.GetRowCursorSet`. Operations can happen on top of these cursors --
most commonly, transforms creating new cursors on top of them for  parallel
evaluation of a data pipeline. But the question is, if you need to "recombine"
them into a sequence again, how do to it? The `Batch` property is the
mechanism by which the data from these multiple cursors returned by
`IDataView.GetRowCursorSet` can be reconciled into a single, cohesive,
sequence.

The question might be, why recombine. This can be done for several reasons: we
may want repeatability and determinism in such a way that requires we view the
rows in a simple sequence, or the cursor may be stateful in some way that
precludes partitioning it, or some other consideration. And, since a core
`IDataView` design principle is repeatability, we now have a problem of how to
reconcile those separate partitioning.

Incidentally, for those working on the ML.NET codebase, there is an internal
method `DataViewUtils.ConsolidateGeneric` utility method to perform this
function. It may be helpful to understand how it works intuitively, so that we
can understand `Batch`'s requirements: when we reconcile the outputs of
multiple cursors, the consolidator will take the set of cursors. It will find
the one with the "lowest" `Batch` ID. (This must be uniquely determined: that
is, no two cursors should ever return the same `Batch` value.) It will iterate
on that cursor until the `Batch` ID changes. Whereupon, the consolidator will
find the next cursor with the next lowest batch ID (which should be greater,
of course, than the `Batch` value we were just iterating on).

Put another way: suppose we called `GetRowCursor` (with an optional `Random`
parameter), and we store all the values from the rows from that cursoring in
some list, in order. Now, imagine we create `GetRowCursorSet` (with an
identically constructed `Random` instance), and store the values from the rows
from the cursorings from all of them in a different list, in order,
accompanied by their `Batch` value. Then: if we were to perform a *stable*
sort on the second list keyed by the stored `Batch` value, it should have
content identical to the first list.

So: `Batch` is a `long` value associated with every `DataViewRow` instance.
This quantity must be non-decreasing as we call `MoveNext`. That is, it is
fine for the `Batch` to repeat the same batch value within the same cursor
(though not across cursors from the same set), but any change in the value
must be an increase.

The requirement of consistency is for one cursor or cursors from a *single*
call to `GetRowCursor` or `GetRowCursorSet`. It is not required that the
`Batch` be consistent among multiple independent cursorings, since which
"batch" a row belongs to is often an artifact of whatever thread worker
happens to be available to process that row, which is deliberately not
intended to be a deterministic process.

Note also that if there is only a single cursor, as returned from
`GetRowCursor`, or even `GetRowCursorSet` with an array of length one, it is
typical and perfectly fine for `Batch` to just be `0`.

## `MoveNext`

Once `MoveNext` returns `false`, naturally all subsequent calls to either of
that method should return `false`. It is important that they not throw, return
`true`, or have any other behavior.

## `GetIdGetter`

This treats on the requirements of a proper `GetIdGetter` implementation.

It is common for objects to serve multiple `DataViewRow` instances to iterate
over what is supposed to be the same data, for example, in an `IDataView` a
cursor set will produce the same data as a serial cursor, just partitioned,
and a shuffled cursor will produce the same data as a serial cursor or any
other shuffled cursor, only shuffled. The ID exists for applications that need
to reconcile which entry is actually which. Ideally this ID should be unique,
but for practical reasons, it suffices if collisions are simply extremely
improbable.

To be specific, the original case motivating this functionality was SDCA where
it is both simultaneously important that we see data in a "random-enough"
fashion (so shuffled), but each instance has an associated dual variable. The
ID is used to associate each instance with the corresponding dual variable
across multiple iterations of the data. (Note that in this specific
application collisions merely being improbable is sufficient, since if there
was hypothetically a collision it would not actually probably materially
affect the results anyway, though I'm making that claim without
justification).

Note that this ID, while it must be consistent for multiple streams according
to the semantics above, is not considered part of the data per se. So, to take
the example of a data view specifically, a single data view must render
consistent IDs across all cursorings, but there is no suggestion at all that
if the "same" data were presented in a different data view (as by, say, being
transformed, cached, saved, or whatever), that the IDs between the two
different data views would have any discernable relationship.

This ID is practically often derived from the IDs of some other `DataViewRow`.
For example, when applying `ITransformer.Transform`, the IDs of the output are
usually derived from the IDs of the input. It is not only necessary to claim
that the ID generated here is probabilistically unique, but also describe a
procedure or set of guidelines implementors of this method should attempt to
follow, in order to ensure that downstream components have a fair shake at
producing unique IDs themselves, which I will here attempt to do:

Duplicate IDs being improbable is practically accomplished with a
hashing-derived mechanism. For this we have the `DataViewRowId` methods
`Fork`, `Next`, and `Combine`. See their documentation for specifics, but they
all have in common that they treat the `DataViewRowId` as some sort of
intermediate hash state, then return a new hash state based on hashing of a
block of additional bits. (Since the additional bits hashed in `Fork` and
`Next` are specific, that is, effectively `0`, and `1`, this can be very
efficient.) The basic assumption underlying all of this is that collisions
between two different hash states on the same data, or hashes on the same hash
state on different data, are unlikely to collide.

Note that this is also the reason why `DataViewRowId` was introduced;
collisions become likely when we have the number of elements on the order of
the square root of the hash space. The square root of `UInt64.MaxValue` is
only several billion, a totally reasonable number of instances in a dataset,
whereas a collision in a 128-bit space is less likely.

Let's consider the IDs of a collection of entities, then, to be ideally an
"acceptable set." An "acceptable set" is one that is not especially or
perversely likely to contain collisions versus other sets, and also one
unlikely to result in an especially or perversely likely to collide set of
IDs, so long as the IDs are done according to the following operations that
operate on acceptable sets.

1. The simple enumeration of `DataViewRowId` numeric values from any number is
   an acceptable set. (This covers how most loaders generate IDs. Typically,
   we start from 0, but other choices, like -1, are acceptable.)

2. The subset of any acceptable set is an acceptable set. (For example, all
   filter transforms that map any input row to 0 or 1 output rows, can just
   pass through the input cursor's IDs.) Note that this covers the simplest
   and most common case of `ITransformer` row-to-row mapping transforms, as
   well as any one-to-one-or-zero filter.

3. Applying `Fork` to every element of an acceptable set exactly once will
   result in an acceptable set.

4. As a generalization of the above, if for each element of an acceptable set,
   you built the set comprised of the single application of `Fork` on that ID
   followed by the set of any number of application of `Next`, the union of
   all such sets would itself be an acceptable set. (This is useful, for
   example, for operations that produce multiple items per input item. So, if
   you produced two rows based on every single input row, if the input ID were
   _id_, then, the ID of the first row could be `Fork` of _id_, and the second
   row could have ID of `Fork` then `Next` of the same _id_.)

5. If you have potentially multiple acceptable sets, while the union of them
   obviously might not be acceptable, if you were to form a mapping from each
   set, to a different ID of some other acceptable set (each such ID should be
   different), and then for each such set/ID pairing, create the set created
   from `Combine` of the items of that set with that ID, and then union of
   those sets will be acceptable. (This is useful, for example, if you had
   something like a join, or a Cartesian product transform, or something like
   that.)

6. Moreover, similar to the note about the use of `Fork`, and `Next`, if
   during the creation of one of those sets describe above, you were to form
   for each item of that set, a set resulting from multiple applications of
   `Next`, the union of all those would also be an acceptable set.

This list is not exhaustive. Other operations I have not listed above might
result in an acceptable set as well, but one should not attempt other
operations without being absolutely certain of what one is doing. The general
idea is that one should structure the construction of IDs, so that it will
never arise that the same ID is hashed against the same data, and are
introduced as if we expect them to be two separate IDs.

Of course, with a malicious actor upstream, collisions are possible and can be
engineered quite trivially (for example, just by returning a constant ID for
all rows), but we're not supposing that the input `IDataView` is maliciously
engineering hash states, or applying the operations above in any strange way
to attempt to induce collisions. For example, you could take, operation 1,
define it to be the enumeration of all `DataViewRowId` values, then take
operation 2 to select out specifically those that are hash states that will
result in collisions. But I'm supposing this is not happening. If you are
running an implementation of a dataview in memory that you're supposing is
malicious, you probably have bigger problems than someone inducing collisions
to make SDCA converge suboptimally.

It should be noted that in *most* applications it is not nearly so complicated
as all this: the vast majority of `IDataView` implementations are either the
result of applying a simple row-to-row transformer (in which case passing
through the ID is perfectly acceptable), or the result of enumerating over
some sequential dataset (in which case the ordinal value is a perfectly fine
ID). The details above only become very important with things like, say,
combining multiple `IDataView`s together, performing one-to-many row
transformations, or other such things like this, in which case the details
above become important.

One common thought that comes up is the idea that we can have some "global
position" instead of ID.  This was actually the first idea by the original
implementor, and if if it *were* possible it would definitely make for a
cleaner, simpler solution, and multiple people have asked the question to the
point where it would probably be best to have a ready answer about where it
broke down, to undersatnd how it fails. It runs afoul of the earlier desire
with regard to data view cursor sets, that is, that `IDataView` cursors
should, if possible, present split cursors that can run independently on
"batches" of the data. But, let's imagine something like the operation for
filtering; if I have a batch `0` comprised of 64 rows, and a batch `1` with
another 64 rows, and one filters rows with missing values, the ML.NET code
could not provide the row sequence number for batch `1` until  had counted the
number of rows that passed in batch `0`, which compromises the whole point of
why we wanted to have cursor sets in the first place. The same is true also
for one-to-many `IDataView` implementations (for example, joins, or something
like that), where even a strictly increasing (but not necessarily contiguous)
value may not be possible, since you cannot even bound the number. So,
regrettably, that simpler solution would not work.