// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A cursor that allows slot-by-slot access of data. This is to <see cref="ITransposeDataView"/>
    /// what <see cref="DataViewRowCursor"/> is to <see cref="IDataView"/>.
    /// </summary>
    [BestFriend]
    internal abstract class SlotCursor : IDisposable
    {
        [BestFriend]
        private protected readonly IChannel Ch;
        private bool _started;
        protected bool Disposed { get; private set; }

        /// <summary>
        /// Whether the cursor is in a state where it can serve up data, that is, <see cref="MoveNext"/>
        /// has been called and returned <see langword="true"/>.
        /// </summary>
        [BestFriend]
        private protected bool IsGood => _started && !Disposed;

        [BestFriend]
        private protected SlotCursor(IChannelProvider provider)
        {
            Contracts.AssertValue(provider);
            Ch = provider.Start("Slot Cursor");
        }

        /// <summary>
        /// The slot index. Incremented by one when <see cref="MoveNext"/> is called and returns <see langword="true"/>.
        /// When initially created, or after <see cref="MoveNext"/> returns <see langword="false"/>, this will be <c>-1</c>.
        /// </summary>
        public abstract int SlotIndex { get; }

        /// <summary>
        /// Advance to the next slot. When the cursor is first created, this method should be called to
        /// move to the first slot. Returns <see langword="false"/> if there are no more slots.
        /// </summary>
        public abstract bool MoveNext();

        /// <summary>
        /// The slot type for this cursor. Note that this should equal the
        /// <see cref="ITransposeDataView.GetSlotType"/> for the column from which this slot cursor
        /// was created.
        /// </summary>
        public abstract VectorType GetSlotType();

        /// <summary>
        /// A getter delegate for the slot values. The type <typeparamref name="TValue"/> must correspond
        /// to the item type from <see cref="ITransposeDataView.GetSlotType"/>.
        /// </summary>
        public abstract ValueGetter<VBuffer<TValue>> GetGetter<TValue>();

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (Disposed)
                return;
            if (disposing)
                Ch.Dispose();
            Disposed = true;
        }

        /// <summary>
        /// For wrapping another slot cursor from which we get <see cref="SlotIndex"/> and <see cref="MoveNext"/>,
        /// but not the data or type accesors. Somewhat analogous to the <see cref="SynchronizedCursorBase"/>
        /// for <see cref="DataViewRowCursor"/>s.
        /// </summary>
        [BestFriend]
        internal abstract class SynchronizedSlotCursor : SlotCursor
        {
            private readonly SlotCursor _root;

            public SynchronizedSlotCursor(IChannelProvider provider, SlotCursor cursor)
                : base(provider)
            {
                Contracts.AssertValue(cursor);
                // If the input is itself a sync-base, we can walk up the chain to get its root,
                // thereby making things more efficient.
                _root = cursor is SynchronizedSlotCursor sync ? sync._root : cursor;
            }

            public override bool MoveNext()
                => _root.MoveNext();

            public override int SlotIndex => _root.SlotIndex;
        }

        /// <summary>
        /// A useful base class for common <see cref="SlotCursor"/> implementations, somewhat
        /// analogous to the <see cref="RootCursorBase"/> for <see cref="DataViewRowCursor"/>s.
        /// </summary>
        [BestFriend]
        internal abstract class RootSlotCursor : SlotCursor
        {
            private int _slotIndex;

            public RootSlotCursor(IChannelProvider provider)
                : base(provider)
            {
                _slotIndex = -1;
            }

            public override int SlotIndex => _slotIndex;

            protected sealed override void Dispose(bool disposing)
            {
                if (Disposed)
                    return;
                if (disposing)
                    _slotIndex = -1;
                base.Dispose(disposing);
            }

            public override bool MoveNext()
            {
                if (Disposed)
                    return true;

                if (MoveNextCore())
                {
                    _slotIndex++;
                    Ch.Assert(_slotIndex >= 0);
                    _started = true;
                    return true;
                }

                Dispose();
                Ch.Assert(_slotIndex < 0);
                return false;
            }

            /// <summary>
            /// Core implementation of <see cref="MoveNext"/>. This is called only if this method
            /// has not yet previously returned <see langword="false"/>.
            /// </summary>
            protected abstract bool MoveNextCore();
        }
    }
}
