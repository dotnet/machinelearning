// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    // REVIEW: Since each cursor will create a channel, it would be great that the RootCursorBase takes
    // ownership of the channel so the derived classes don't have to.

    /// <summary>
    /// Base class for creating a cursor with default tracking of <see cref="Position"/>. All calls to <see cref="MoveNext"/>
    /// will be seen by subclasses of this cursor. For a cursor that has an input cursor and does not need notification on
    /// <see cref="MoveNext"/>, use <see cref="SynchronizedCursorBase"/> instead.
    /// </summary>
    [BestFriend]
    internal abstract class RootCursorBase : DataViewRowCursor
    {
        protected readonly IChannel Ch;
        private long _position;
        private bool _disposed;

        /// <summary>
        /// Zero-based position of the cursor.
        /// </summary>
        public sealed override long Position => _position;

        /// <summary>
        /// Convenience property for checking whether the current state of the cursor is one where data can be fetched.
        /// </summary>
        protected bool IsGood => _position >= 0;

        /// <summary>
        /// Creates an instance of the <see cref="RootCursorBase"/> class
        /// </summary>
        /// <param name="provider">Channel provider</param>
        protected RootCursorBase(IChannelProvider provider)
        {
            Contracts.CheckValue(provider, nameof(provider));
            Ch = provider.Start("Cursor");

            _position = -1;
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed)
                return;
            if (disposing)
            {
                Ch.Dispose();
                _position = -1;
            }
            _disposed = true;
            base.Dispose(disposing);

        }

        public sealed override bool MoveNext()
        {
            if (_disposed)
                return false;

            if (MoveNextCore())
            {
                _position++;
                return true;
            }

            Dispose();
            return false;
        }

        /// <summary>
        /// Core implementation of <see cref="MoveNext"/>, called if no prior call to this method
        /// has returned <see langword="false"/>.
        /// </summary>
        protected abstract bool MoveNextCore();
    }
}