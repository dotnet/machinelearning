// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Base class for a cursor has an input cursor, but still needs to do work on
    /// <see cref="RowCursor.MoveNext"/> / <see cref="RowCursor.MoveMany(long)"/>.
    /// </summary>
    [BestFriend]
    internal abstract class LinkedRootCursorBase : RootCursorBase
    {

        /// <summary>Gets the input cursor.</summary>
        protected RowCursor Input { get; }

        /// <summary>
        /// Returns the root cursor of the input. It should be used to perform MoveNext or MoveMany operations.
        /// Note that <see cref="RowCursor.GetRootCursor"/> returns <see langword="this"/>, not <see cref="Root"/>.
        /// <see cref="Root"/> is used to advance our input, not for clients of this cursor. That is why it is
        /// protected, not public.
        /// </summary>
        protected RowCursor Root { get; }

        protected LinkedRootCursorBase(IChannelProvider provider, RowCursor input)
            : base(provider)
        {
            Ch.AssertValue(input, nameof(input));

            Input = input;
            Root = Input.GetRootCursor();
        }

        public override void Dispose()
        {
            if (State != CursorState.Done)
            {
                Input.Dispose();
                base.Dispose();
            }
        }
    }
}