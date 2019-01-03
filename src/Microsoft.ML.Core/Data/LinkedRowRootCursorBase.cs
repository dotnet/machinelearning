// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A base class for a <see cref="RowCursor"/> that has an input cursor, but still needs
    /// to do work on <see cref="RowCursor.MoveNext"/>/<see cref="RowCursor.MoveMany(long)"/>. Note
    /// that the default <see cref="LinkedRowRootCursorBase.GetGetter{TValue}(int)"/> assumes
    /// that each input column is exposed as an output column with the same column index.
    /// </summary>
    [BestFriend]
    internal abstract class LinkedRowRootCursorBase : LinkedRootCursorBase
    {
        private readonly bool[] _active;

        /// <summary>Gets row's schema.</summary>
        public sealed override Schema Schema { get; }

        protected LinkedRowRootCursorBase(IChannelProvider provider, RowCursor input, Schema schema, bool[] active)
            : base(provider, input)
        {
            Ch.CheckValue(schema, nameof(schema));
            Ch.Check(active == null || active.Length == schema.Count);
            _active = active;
            Schema = schema;
        }

        public sealed override bool IsColumnActive(int col)
        {
            Ch.Check(0 <= col && col < Schema.Count);
            return _active == null || _active[col];
        }

        public override ValueGetter<TValue> GetGetter<TValue>(int col)
        {
            return Input.GetGetter<TValue>(col);
        }
    }
}