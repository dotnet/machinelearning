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
    internal abstract class SynchronizedCursorBase<TBase> : IRowCursor
        where TBase : class, IRowCursor
    {
        protected readonly IChannel Ch;

        private readonly IRowCursor _root;
        private bool _disposed;

        protected TBase Input { get; }

        public long Position => _root.Position;

        public long Batch => _root.Batch;

        public CursorState State => _root.State;

        /// <summary>
        /// Convenience property for checking whether the current state is CursorState.Good.
        /// </summary>
        protected bool IsGood => _root.State == CursorState.Good;

        public abstract Schema Schema { get; }

        protected SynchronizedCursorBase(IChannelProvider provider, TBase input)
        {
            Contracts.AssertValue(provider, "provider");
            Ch = provider.Start("Cursor");

            Ch.AssertValue(input, "input");
            Input = input;
            _root = Input.GetRootCursor();
        }

        public virtual void Dispose()
        {
            if (!_disposed)
            {
                Input.Dispose();
                Ch.Dispose();
                _disposed = true;
            }
        }

        public bool MoveNext() => _root.MoveNext();

        public bool MoveMany(long count) => _root.MoveMany(count);

        public IRowCursor GetRootCursor() => _root;

        public ValueGetter<UInt128> GetIdGetter() => Input.GetIdGetter();

        public abstract bool IsColumnActive(int col);

        public abstract ValueGetter<TValue> GetGetter<TValue>(int col);
    }
}
