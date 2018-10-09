// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This is a base interface for an <see cref="ICursor"/> and <see cref="IRow"/>. It contains only the
    /// positional properties, no behavioral methods, and no data.
    /// </summary>
    public interface ICounted
    {
        /// <summary>
        /// This is incremented for ICursor when the underlying contents changes, giving clients a way to detect change.
        /// Generally it's -1 when the object is in an invalid state. In particular, for an <see cref="ICursor"/>, this is -1
        /// when the <see cref="ICursor.State"/> is <see cref="CursorState.NotStarted"/> or <see cref="CursorState.Done"/>.
        ///
        /// Note that this position is not position within the underlying data, but position of this cursor only.
        /// If one, for example, opened a set of parallel streaming cursors, or a shuffled cursor, each such cursor's
        /// first valid entry would always have position 0.
        /// </summary>
        long Position { get; }

        /// <summary>
        /// This provides a means for reconciling multiple streams of counted things. Generally, in each stream,
        /// batch numbers should be non-decreasing. Furthermore, any given batch number should only appear in one
        /// of the streams. Order is determined by batch number. The reconciler ensures that each stream (that is
        /// still active) has at least one item available, then takes the item with the smallest batch number.
        ///
        /// Note that there is no suggestion that the batches for a particular entry will be consistent from
        /// cursoring to cursoring, except for the consistency in resulting in the same overall ordering. The same
        /// entry could have different batch numbers from one cursoring to another. There is also no requirement
        /// that any given batch number must appear, at all.
        /// </summary>
        long Batch { get; }

        /// <summary>
        /// A getter for a 128-bit ID value. It is common for objects to serve multiple <see cref="ICounted"/>
        /// instances to iterate over what is supposed to be the same data, for example, in a <see cref="IDataView"/>
        /// a cursor set will produce the same data as a serial cursor, just partitioned, and a shuffled cursor
        /// will produce the same data as a serial cursor or any other shuffled cursor, only shuffled. The ID
        /// exists for applications that need to reconcile which entry is actually which. Ideally this ID should
        /// be unique, but for practical reasons, it suffices if collisions are simply extremely improbable.
        ///
        /// Note that this ID, while it must be consistent for multiple streams according to the semantics
        /// above, is not considered part of the data per se. So, to take the example of a data view specifically,
        /// a single data view must render consistent IDs across all cursorings, but there is no suggestion at
        /// all that if the "same" data were presented in a different data view (as by, say, being transformed,
        /// cached, saved, or whatever), that the IDs between the two different data views would have any
        /// discernable relationship.</summary>
        ValueGetter<UInt128> GetIdGetter();
    }

    /// <summary>
    /// Defines the possible states of a cursor.
    /// </summary>
    public enum CursorState
    {
        NotStarted,
        Good,
        Done
    }

    /// <summary>
    /// The basic cursor interface. <see cref="ICounted.Position"/> is incremented by <see cref="MoveNext"/>
    /// and <see cref="MoveMany"/>. When the cursor state is <see cref="CursorState.NotStarted"/> or
    /// <see cref="CursorState.Done"/>, <see cref="ICounted.Position"/> is -1. Otherwise,
    /// <see cref="ICounted.Position"/> >= 0.
    /// </summary>
    public interface ICursor : ICounted, IDisposable
    {
        /// <summary>
        /// Returns the state of the cursor. Before the first call to <see cref="MoveNext"/> or
        /// <see cref="MoveMany(long)"/> this should be <see cref="CursorState.NotStarted"/>. After
        /// any call those move functions that returns <c>true</c>, this should return
        /// <see cref="CursorState.Good"/>,
        /// </summary>
        CursorState State { get; }

        /// <summary>
        /// Advance to the next row. When the cursor is first created, this method should be called to
        /// move to the first row. Returns <c>false</c> if there are no more rows.
        /// </summary>
        bool MoveNext();

        /// <summary>
        /// Logically equivalent to calling <see cref="MoveNext"/> the given number of times. The
        /// <paramref name="count"/> parameter must be positive. Note that cursor implementations may be
        /// able to optimize this.
        /// </summary>
        bool MoveMany(long count);

        /// <summary>
        /// Returns a cursor that can be used for invoking <see cref="ICounted.Position"/>, <see cref="State"/>,
        /// <see cref="MoveNext"/>, and <see cref="MoveMany"/>, with results identical to calling those
        /// on this cursor. Generally, if the root cursor is not the same as this cursor, using the
        /// root cursor will be faster. As an aside, note that this is not necessarily the case of
        /// values from <see cref="ICounted.GetIdGetter"/>.
        /// </summary>
        ICursor GetRootCursor();
    }
}