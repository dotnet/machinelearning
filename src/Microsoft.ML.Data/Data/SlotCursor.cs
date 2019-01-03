// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A cursor that allows slot-by-slot access of data. This is to <see cref="ITransposeDataView"/>
    /// what <see cref="RowCursor"/> is to <see cref="IDataView"/>.
    /// </summary>
    public abstract class SlotCursor : IDisposable
    {
        [BestFriend]
        private protected readonly IChannel Ch;
        private CursorState _state;

        /// <summary>
        /// Whether the cursor is in a state where it can serve up data, that is, <see cref="MoveNext"/>
        /// has been called and returned <see langword="true"/>.
        /// </summary>
        [BestFriend]
        private protected bool IsGood => _state == CursorState.Good;

        [BestFriend]
        private protected SlotCursor(IChannelProvider provider)
        {
            Contracts.AssertValue(provider);
            Ch = provider.Start("Slot Cursor");
            _state = CursorState.NotStarted;
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
        /// <see cref="ITransposeSchema.GetSlotType"/> for the column from which this slot cursor
        /// was created.
        /// </summary>
        public abstract VectorType GetSlotType();

        /// <summary>
        /// A getter delegate for the slot values. The type <typeparamref name="TValue"/> must correspond
        /// to the item type from <see cref="ITransposeSchema.GetSlotType"/>.
        /// </summary>
        public abstract ValueGetter<VBuffer<TValue>> GetGetter<TValue>();

        public virtual void Dispose()
        {
            if (_state != CursorState.Done)
            {
                Ch.Dispose();
                _state = CursorState.Done;
            }
        }

        /// <summary>
        /// For wrapping another slot cursor from which we get <see cref="SlotIndex"/> and <see cref="MoveNext"/>,
        /// but not the data or type accesors. Somewhat analogous to the <see cref="SynchronizedCursorBase"/>
        /// for <see cref="RowCursor"/>s.
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
        /// analogous to the <see cref="RootCursorBase"/> for <see cref="RowCursor"/>s.
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

            public override void Dispose()
            {
                base.Dispose();
                _slotIndex = -1;
            }

            public override bool MoveNext()
            {
                if (_state == CursorState.Done)
                    return false;

                Ch.Assert(_state == CursorState.NotStarted || _state == CursorState.Good);
                if (MoveNextCore())
                {
                    Ch.Assert(_state == CursorState.NotStarted || _state == CursorState.Good);

                    _slotIndex++;
                    _state = CursorState.Good;
                    return true;
                }

                Dispose();
                return false;
            }

            /// <summary>
            /// Core implementation of <see cref="MoveNext"/>. Called only if this method
            /// has not yet previously returned <see langword="false"/>.
            /// </summary>
            protected abstract bool MoveNextCore();
        }
    }
}
