// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Data
{
    // REVIEW: Since each cursor will create a channel, it would be great that the RootCursorBase takes
    // ownership of the channel so the derived classes don't have to.

    /// <summary>
    /// Base class for creating a cursor with default tracking of <see cref="Position"/> and <see cref="State"/>
    /// with a default implementation of <see cref="MoveManyCore(long)"/> (call <see cref="MoveNextCore"/> repeatedly).
    /// This cursor base class returns "this" from <see cref="GetRootCursor"/>. That is, all
    /// <see cref="MoveNext"/>/<see cref="MoveMany(long)"/> calls will be seen by this cursor. For a cursor
    /// that has an input cursor and does NOT need notification on <see cref="MoveNext"/>/<see cref="MoveMany(long)"/>,
    /// use <see cref="SynchronizedCursorBase{TBase}"/> .
    /// </summary>
    public abstract class RootCursorBase : ICursor
    {
        protected readonly IChannel Ch;

        /// <summary>
        /// Zero-based position of the cursor.
        /// </summary>
        public long Position { get; private set; }

        public abstract long Batch { get; }

        public abstract ValueGetter<UInt128> GetIdGetter();

        public CursorState State { get; private set; }

        /// <summary>
        /// Convenience property for checking whether the current state of the cursor is <see cref="CursorState.Good"/>.
        /// </summary>
        protected bool IsGood => State == CursorState.Good;

        /// <summary>
        /// Creates an instance of the RootCursorBase class
        /// </summary>
        /// <param name="provider">Channel provider</param>
        protected RootCursorBase(IChannelProvider provider)
        {
            Contracts.CheckValue(provider, nameof(provider));
            Ch = provider.Start("Cursor");

            Position = -1;
            State = CursorState.NotStarted;
        }

        public virtual void Dispose()
        {
            if (State != CursorState.Done)
            {
                Ch.Done();
                Ch.Dispose();
                Position = -1;
                State = CursorState.Done;
            }
        }

        public bool MoveNext()
        {
            if (State == CursorState.Done)
                return false;

            Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);
            if (MoveNextCore())
            {
                Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);

                Position++;
                State = CursorState.Good;
                return true;
            }

            Dispose();
            return false;
        }

        public bool MoveMany(long count)
        {
            // Note: If we decide to allow count == 0, then we need to special case
            // that MoveNext() has never been called. It's not entirely clear what the return
            // result would be in that case.
            Ch.CheckParam(count > 0, nameof(count));

            if (State == CursorState.Done)
                return false;

            Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);
            if (MoveManyCore(count))
            {
                Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);

                Position += count;
                State = CursorState.Good;
                return true;
            }

            Dispose();
            return false;
        }

        /// <summary>
        /// Default implementation is to simply call MoveNextCore repeatedly. Derived classes should
        /// override if they can do better.
        /// </summary>
        /// <param name="count">The number of rows to move forward.</param>
        /// <returns>Whether the move forward is on a valid row</returns>
        protected virtual bool MoveManyCore(long count)
        {
            Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);
            Ch.Assert(count > 0);

            while (MoveNextCore())
            {
                Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);
                if (--count <= 0)
                    return true;
            }

            return false;
        }

        /// <summary>
        /// Core implementation of <see cref="MoveNext"/>, called if the cursor state is not
        /// <see cref="CursorState.Done"/>.
        /// </summary>
        protected abstract bool MoveNextCore();

        /// <summary>
        /// Returns a cursor that can be used for invoking <see cref="Position"/>, <see cref="State"/>,
        /// <see cref="MoveNext"/>, and <see cref="MoveMany(long)"/>, with results identical to calling
        /// those on this cursor. Generally, if the root cursor is not the same as this cursor, using
        /// the root cursor will be faster.
        /// </summary>
        public ICursor GetRootCursor() => this;
    }
}