// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Convenient base class for <see cref="DataViewRow"/> implementors that wrap a single <see cref="DataViewRow"/>
    /// as their input. The <see cref="DataViewRow.Position"/>, <see cref="DataViewRow.Batch"/>, and <see cref="DataViewRow.GetIdGetter"/>
    /// are taken from this <see cref="Input"/>.
    /// </summary>
    [BestFriend]
    internal abstract class WrappingRow : DataViewRow
    {
        private bool _disposed;

        /// <summary>
        /// The wrapped input row.
        /// </summary>
        protected DataViewRow Input { get; }

        public sealed override long Batch => Input.Batch;
        public sealed override long Position => Input.Position;
        public override ValueGetter<DataViewRowId> GetIdGetter() => Input.GetIdGetter();

        [BestFriend]
        private protected WrappingRow(DataViewRow input)
        {
            Contracts.AssertValue(input);
            Input = input;
        }

        /// <summary>
        /// This override of the dispose method by default only calls <see cref="Input"/>'s
        /// <see cref="IDisposable.Dispose"/> method, but subclasses can enable additional functionality
        /// via the <see cref="DisposeCore(bool)"/> functionality.
        /// </summary>
        /// <param name="disposing"></param>
        protected sealed override void Dispose(bool disposing)
        {
            if (_disposed)
                return;
            // Since the input was created first, and this instance may depend on it, we should
            // dispose local resources first before potentially disposing the input row resources.
            DisposeCore(disposing);
            if (disposing)
                Input.Dispose();
            _disposed = true;
        }

        /// <summary>
        /// Called from <see cref="Dispose(bool)"/> with <see langword="true"/> in the case where
        /// that method has never been called before, and right after <see cref="Input"/> has been
        /// disposed. The default implementation does nothing.
        /// </summary>
        /// <param name="disposing">Whether this was called through the dispose path, as opposed
        /// to the finalizer path.</param>
        protected virtual void DisposeCore(bool disposing)
        {
        }
    }
}
