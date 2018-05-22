# `ICursor` Notes

This document includes some more in depth notes on some expert topics for
`ICursor` implementations.

## `Batch`

Some cursorable implementations, like `IDataView`, can through
`GetRowCursorSet` return a set of parallel cursors that partition the sequence
of rows as would have normally been returned through a plain old
`GetRowCursor`, just sharded into multiple cursors. These cursors can be
accessed across multiple threads to enable parallel evaluation of a data
pipeline. This is key for the data pipeline performance.

However, even though the data pipeline can perform this parallel evaluation,
at the end of this parallelization we usually ultimately want to recombine the
separate thread's streams back into a single stream. This is accomplished
through `Batch`.

So, to review what actually happens in ML.NET code: multiple cursors are
returned through a method like `IDataView.GetRowCursorSet`. Operations can
happen on top of these cursors -- most commonly, transforms creating new
cursors on top of them -- and the `IRowCursorConsolidator` implementation will
utilize this `Batch` field to "reconcile" the multiple cursors back down into
one cursor.

It may help to first understand this process intuitively, to understand
`Batch`'s requirements: when we reconcile the outputs of multiple cursors, the
consolidator will take the set of cursors. It will find the one with the
"lowest" `Batch` ID. (This must be uniquely determined: that is, no two
cursors should ever return the same `Batch` value.) It will iterate on that
cursor until the `Batch` ID changes. Whereupon, the consolidator will find the
next cursor with the next lowest batch ID (which should be greater, of course,
than the `Batch` value we were just iterating on).

Put another way: if we called `GetRowCursor` (possibly with an `IRandom`
instance), and we store all the values from the rows from that cursoring in
some list, in order. Now, imagine we create `GetRowCursorSet` (with an
identically constructed `IRandom` instance), and store the values from the
rows from the cursorings from all of them in a different list, in order,
accompanied by their `Batch` value. Then: if we were to perform a *stable*
sort on the second list keyed by the stored `Batch` value, it should have
content identical to the first list.

So: `Batch` is a `long` value associated with every `ICounted` implementation
(including implementations of `ICursor`). This quantity must be:

Non-decreasing as we call `MoveNext` or `MoveMany`. That is, it is fine for
the `Batch` to repeat the same batch value within the same cursor (though not
across cursors from the same set), but any change in the value must be an
increase.

The requirement of consistency is for one cursor or cursors from a *single*
call to `GetRowCursor` or `GetRowCursorSet`. It is not required that the
`Batch` be consistent among multiple independent cursorings.

## `MoveNext` and `MoveMany`

Once `MoveNext` or `MoveMany` returns `false`, naturally all subsequent calls
to either of these two methods should return `false`. It is important that
they not throw, return `true`, or have any other behavior.

## `GetIdGetter`

This treats on the requirements of a proper `GetIdGetter` implementation.

It is common for objects to serve multiple `ICounted` instances to iterate
over what is supposed to be the same data, e.g., in an `IDataView` a cursor
set will produce the same data as a serial cursor, just partitioned, and a
shuffled cursor will produce the same data as a serial cursor or any other
shuffled cursor, only shuffled. The ID exists for applications that need to
reconcile which entry is actually which. Ideally this ID should be unique, but
for practical reasons, it suffices if collisions are simply extremely
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

Since this ID is practically often derived from the IDs of some other
`ICounted` (e.g., for a transform, the IDs of the output are usually derived
from the IDs of the input), it is not only necessary to claim that the ID
generated here is probabilistically unique, but also describe a procedure or
set of guidelines implementors of this method should attempt to follow, in
order to ensure that downstream components have a fair shake at producing
unique IDs themselves.

Duplicate IDs being improbable is practically accomplished with a
hashing-derived mechanism. For this we have the `UInt128` methods `Fork`,
`Next`, and `Combine`. See their documentation for specifics, but they all
have in common that they treat the `UInt128` as some sort of intermediate hash
state, then return a new hash state based on hashing of a block of additional
'bits.' (Since the bits hashed may be fixed, depending on the operation, this
can be very efficient.) The basic assumption underlying all of that collisions
between two different hash states on the same data, or hashes on the same hash
state on different data, are unlikely to collide. Note that this is also the
reason why `UInt128` was introduced; collisions become likely when we have the
number of elements on the order of the square root of the hash space. The
square root of `UInt64.MaxValue` is only several billion, a totally reasonable
number of instances in a dataset, whereas a collision in a 128-bit space is
less likely.

Let's consider the IDs of a collection of entities, then, to be ideally an
"acceptable set." An "acceptable set" is one that is not especially or
perversely likely to contain collisions versus other sets, and also one
unlikely to result in an especially or perversely likely to collide set of
IDs, so long as the IDs are done according to the following operations that
operate on acceptable sets.

1. The simple enumeration of `UInt128` numeric values from any number is an
   acceptable set. (This covers how most loaders generate IDs. Typically, we
   start from 0, but other choices, like -1, are acceptable.)

2. The subset of any acceptable set is an acceptable set. (For example, all
   filter transforms that map any input row to 0 or 1 output rows, can just
   pass through the input cursor's IDs.)

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
engineered quite trivially (e.g., just by returning a constant ID for all
rows), but we're not supposing that the input `IDataView` is maliciously
engineering hash states, or applying the operations above in any strange way
to attempt to induce collisions. E.g., you could take, operation 1, define it
to be the enumeration of all `UInt128` values, then take operation 2 to select
out specifically those that are hash states that will result in collisions.
But I'm supposing this is not happening. If you are running an implementation
of a dataview in memory that you're supposing is malicious, you probably have
bigger problems than someone inducing collisions.