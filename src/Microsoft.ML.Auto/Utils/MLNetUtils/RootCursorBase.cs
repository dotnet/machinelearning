// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;

namespace Microsoft.ML.Auto
{
    internal abstract class RootCursorBase : DataViewRowCursor
    {
        protected readonly IChannel Ch;

        private long _position;

        private bool _disposed;

        /// <summary>
        /// Zero-based position of the cursor.
        /// </summary>
        public sealed override long Position => _position;

        /// <summary>
        /// Convenience property for checking whether the current state of the cursor is one where data can be fetched.
        /// </summary>
        protected bool IsGood => _position >= 0;

        /// <summary>
        /// Creates an instance of the <see cref="T:Microsoft.ML.Data.RootCursorBase" /> class
        /// </summary>
        /// <param name="provider">Channel provider</param>
        protected RootCursorBase(IChannelProvider provider)
        {
            Contracts.CheckValue(provider, "provider");
            Ch = provider.Start("Cursor");
            _position = -1L;
        }

        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    Ch.Dispose();
                    _position = -1L;
                }
                _disposed = true;
                base.Dispose(disposing);
            }
        }

        public sealed override bool MoveNext()
        {
            if (_disposed)
            {
                return false;
            }
            if (MoveNextCore())
            {
                _position += 1L;
                return true;
            }
            base.Dispose();
            return false;
        }

        /// <summary>
        /// Core implementation of <see cref="M:Microsoft.ML.Data.RootCursorBase.MoveNext" />, called if no prior call to this method
        /// has returned <see langword="false" />.
        /// </summary>
        protected abstract bool MoveNextCore();
    }
}