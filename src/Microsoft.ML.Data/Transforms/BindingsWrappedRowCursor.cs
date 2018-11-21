// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A class for mapping an input to an output cursor assuming no output columns
    /// are requested, given a bindings object. This can be useful for transforms
    /// utilizing the <see cref="ColumnBindingsBase"/>, but for which it is
    /// inconvenient or inefficient to handle the "no output selected" case in their
    /// own implementation.
    /// </summary>
    public sealed class BindingsWrappedRowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
    {
        private readonly ColumnBindingsBase _bindings;

        public Schema Schema => _bindings.AsSchema;

        /// <summary>
        /// Creates a wrapped version of the cursor
        /// </summary>
        /// <param name="provider">Channel provider</param>
        /// <param name="input">The input cursor</param>
        /// <param name="bindings">The bindings object, </param>
        public BindingsWrappedRowCursor(IChannelProvider provider, IRowCursor input, ColumnBindingsBase bindings)
            : base(provider, input)
        {
            Ch.CheckValue(input, nameof(input));
            Ch.CheckValue(bindings, nameof(bindings));

            _bindings = bindings;
        }

        public bool IsColumnActive(int col)
        {
            Ch.Check(0 <= col & col < _bindings.ColumnCount, "col");
            bool isSrc;
            col = _bindings.MapColumnIndex(out isSrc, col);
            return isSrc && Input.IsColumnActive(col);
        }

        public ValueGetter<TValue> GetGetter<TValue>(int col)
        {
            Ch.Check(IsColumnActive(col), "col");
            bool isSrc;
            col = _bindings.MapColumnIndex(out isSrc, col);
            Ch.Assert(isSrc);
            return Input.GetGetter<TValue>(col);
        }
    }
}
