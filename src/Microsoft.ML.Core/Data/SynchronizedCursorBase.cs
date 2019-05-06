// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Base class for creating a cursor on top of another cursor that does not add or remove rows.
    /// It forces one-to-one correspondence between items in the input cursor and this cursor.
    /// It delegates all ICursor functionality except Dispose() to the root cursor.
    /// Dispose is virtual with the default implementation delegating to the input cursor.
    /// </summary>
    [BestFriend]
    internal abstract class SynchronizedCursorBase : DataViewRowCursor
    {
        protected readonly IChannel Ch;

        /// <summary>
        /// The synchronized cursor base, as it merely passes through requests for all "positional" calls (including
        /// <see cref="MoveNext"/>, <see cref="Position"/>, <see cref="Batch"/>, and so forth), offers an opportunity
        /// for optimization for "wrapping" cursors (which are themselves often <see cref="SynchronizedCursorBase"/>
        /// implementors) to get this root cursor. But, this can only be done by exposing this root cursor, as we do here.
        /// Internal code should be quite careful in using this as the potential for misuse is quite high.
        /// </summary>
        internal readonly DataViewRowCursor Root;
        private bool _disposed;

        protected DataViewRowCursor Input { get; }

        public sealed override long Position => Root.Position;

        public sealed override long Batch => Root.Batch;

        /// <summary>
        /// Convenience property for checking whether the cursor is in a good state where values
        /// can be retrieved, that is, whenever <see cref="Position"/> is non-negative.
        /// </summary>
        protected bool IsGood => Position >= 0;

        protected SynchronizedCursorBase(IChannelProvider provider, DataViewRowCursor input)
        {
            Contracts.AssertValue(provider);
            Ch = provider.Start("Cursor");

            Ch.AssertValue(input);
            Input = input;
            // If this thing happens to be itself an instance of this class (which, practically, it will
            // be in the majority of situations), we can treat the input as likewise being a passthrough,
            // thereby saving lots of "nested" calls on the stack when doing common operations like movement.
            Root = Input is SynchronizedCursorBase syncInput ? syncInput.Root : input;
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed)
                return;
            if (disposing)
            {
                Input.Dispose();
                Ch.Dispose();
            }
            base.Dispose(disposing);
            _disposed = true;
        }

        public sealed override bool MoveNext() => Root.MoveNext();

        public sealed override ValueGetter<DataViewRowId> GetIdGetter() => Input.GetIdGetter();
    }
}
