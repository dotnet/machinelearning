// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Base class for a cursor has an input cursor, but still needs to do work on <see cref="RowCursor.MoveNext"/>.
    /// </summary>
    [BestFriend]
    internal abstract class LinkedRootCursorBase : RootCursorBase
    {

        /// <summary>Gets the input cursor.</summary>
        protected RowCursor Input { get; }

        /// <summary>
        /// Returns the root cursor of the input. It should be used to perform <see cref="RowCursor.MoveNext"/>
        /// operations, but with the distinction, as compared to <see cref="SynchronizedCursorBase"/>, that this is not
        /// a simple passthrough, but rather very implementation specific. For example, a common usage of this class is
        /// on filter cursor implemetnations, where how that input cursor is consumed is very implementation specific.
        /// That is why this is <see langword="protected"/>, not <see langword="private"/>.
        /// </summary>
        protected RowCursor Root { get; }

        private bool _disposed;

        protected LinkedRootCursorBase(IChannelProvider provider, RowCursor input)
            : base(provider)
        {
            Ch.AssertValue(input, nameof(input));

            Input = input;
            Root = Input is SynchronizedCursorBase snycInput ? snycInput.Root : input;
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed)
                return;
            if (disposing)
            {
                Input.Dispose();
                // The base class should set the state to done under these circumstances.

            }
            _disposed = true;
            base.Dispose(disposing);
        }
    }
}