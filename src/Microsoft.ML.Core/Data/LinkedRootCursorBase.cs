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
        where TInput : class, ICursor
    {
        private readonly ICursor _root;

        /// <summary>Gets the input cursor.</summary>
        protected TInput Input { get; }

        /// <summary>
        /// Returns the root cursor of the input. It should be used to perform MoveNext or MoveMany operations.
        /// Note that GetRootCursor() returns "this", NOT Root. Root is used to advance our input, not for
        /// clients of this cursor. That's why it is protected, not public.
        /// </summary>
        protected ICursor Root { get { return _root; } }

        protected LinkedRootCursorBase(IChannelProvider provider, TInput input)
            : base(provider)
        {
            Ch.AssertValue(input, nameof(input));

            Input = input;
            _root = Input.GetRootCursor();
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