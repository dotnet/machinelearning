// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Base class for creating a cursor on top of another cursor that does not add or remove rows.
    /// It forces one-to-one correspondence between items in the input cursor and this cursor.
    /// It delegates all ICursor functionality except Dispose() to the root cursor.
    /// Dispose is virtual with the default implementation delegating to the input cursor.
    /// </summary>
    [BestFriend]
    internal abstract class SynchronizedCursorBase : RowCursor
    {
        protected readonly IChannel Ch;

        private readonly RowCursor _root;
        private bool _disposed;

        protected RowCursor Input { get; }

        public sealed override long Position => _root.Position;

        public sealed override long Batch => _root.Batch;

        public sealed override CursorState State => _root.State;

        /// <summary>
        /// Convenience property for checking whether the current state is CursorState.Good.
        /// </summary>
        protected bool IsGood => _root.State == CursorState.Good;

        protected SynchronizedCursorBase(IChannelProvider provider, RowCursor input)
        {
            Contracts.AssertValue(provider, "provider");
            Ch = provider.Start("Cursor");

            Ch.AssertValue(input, "input");
            Input = input;
            _root = Input.GetRootCursor();
        }

        public override void Dispose()
        {
            if (!_disposed)
            {
                Input.Dispose();
                Ch.Dispose();
                _disposed = true;
            }
        }

        public sealed override bool MoveNext() => _root.MoveNext();

        public sealed override bool MoveMany(long count) => _root.MoveMany(count);

        public sealed override RowCursor GetRootCursor() => _root;

        public sealed override ValueGetter<UInt128> GetIdGetter() => Input.GetIdGetter();
    }
}
