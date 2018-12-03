// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Base class for a cursor has an input cursor, but still needs to do work on
    /// MoveNext/MoveMany.
    /// </summary>
    [BestFriend]
    internal abstract class LinkedRootCursorBase<TInput> : RootCursorBase
        where TInput : class, IRowCursor
    {

        /// <summary>Gets the input cursor.</summary>
        protected TInput Input { get; }

        /// <summary>
        /// Returns the root cursor of the input. It should be used to perform MoveNext or MoveMany operations.
        /// Note that <see cref="IRowCursor.GetRootCursor"/> returns <see langword="this"/>, not <see cref="Root"/>.
        /// <see cref="Root"/> is used to advance our input, not for clients of this cursor. That is why it is
        /// protected, not public.
        /// </summary>
        protected IRowCursor Root { get; }

        protected LinkedRootCursorBase(IChannelProvider provider, TInput input)
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