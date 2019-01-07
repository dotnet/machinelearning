// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    // REVIEW: Since each cursor will create a channel, it would be great that the RootCursorBase takes
    // ownership of the channel so the derived classes don't have to.

    /// <summary>
    /// Base class for creating a cursor with default tracking of <see cref="Position"/> and <see cref="State"/>.
    /// All calls to <see cref="MoveNext"/> calls will be seen by subclasses of this cursor. For a cursor that has
    /// an input cursor and does not need notification on <see cref="MoveNext"/>, use <see cref="SynchronizedCursorBase"/>.
    /// </summary>
    [BestFriend]
    internal abstract class RootCursorBase : RowCursor
    {
        protected readonly IChannel Ch;
        private CursorState _state;
        private long _position;

        /// <summary>
        /// Zero-based position of the cursor.
        /// </summary>
        public sealed override long Position => _position;

        public sealed override CursorState State => _state;

        /// <summary>
        /// Convenience property for checking whether the current state of the cursor is <see cref="CursorState.Good"/>.
        /// </summary>
        protected bool IsGood => State == CursorState.Good;

        /// <summary>
        /// Creates an instance of the <see cref="RootCursorBase"/> class
        /// </summary>
        /// <param name="provider">Channel provider</param>
        protected RootCursorBase(IChannelProvider provider)
        {
            Contracts.CheckValue(provider, nameof(provider));
            Ch = provider.Start("Cursor");

            _position = -1;
            _state = CursorState.NotStarted;
        }

        protected override void Dispose(bool disposing)
        {
            if (State == CursorState.Done)
                return;
            if (disposing)
                Ch.Dispose();
            _position = -1;
            _state = CursorState.Done;
        }

        public sealed override bool MoveNext()
        {
            if (State == CursorState.Done)
                return false;

            Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);
            if (MoveNextCore())
            {
                Ch.Assert(State == CursorState.NotStarted || State == CursorState.Good);

                _position++;
                _state = CursorState.Good;
                return true;
            }

            Dispose();
            return false;
        }

        /// <summary>
        /// Core implementation of <see cref="MoveNext"/>, called if the cursor state is not
        /// <see cref="CursorState.Done"/>.
        /// </summary>
        protected abstract bool MoveNextCore();
    }
}