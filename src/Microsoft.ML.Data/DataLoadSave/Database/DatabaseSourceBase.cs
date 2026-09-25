// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Data.Common;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    internal abstract class DatabaseSourceBase : IDisposable
    {
        private bool _disposed;

        protected DatabaseSourceBase(string commandText, int commandTimeoutInSeconds)
        {
            Contracts.CheckValue(commandText, nameof(commandText));
            Contracts.CheckUserArg(commandTimeoutInSeconds >= 0, nameof(commandTimeoutInSeconds));

            CommandText = commandText;
            CommandTimeoutInSeconds = commandTimeoutInSeconds;
        }

        public abstract DbConnection Connection { get; }
        public string CommandText { get; }
        public int CommandTimeoutInSeconds { get; }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;
            if (disposing)
            {
                // We don't know whether we own the connection, so we don't dispose it here.
            }
            _disposed = true;
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
