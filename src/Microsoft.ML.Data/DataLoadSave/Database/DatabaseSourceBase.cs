// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data.Common;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    internal abstract class DatabaseSourceBase
    {
        protected DatabaseSourceBase(string commandText, int commandTimeoutInSeconds)
        {
            Contracts.CheckValue(commandText, nameof(commandText));
            Contracts.CheckUserArg(commandTimeoutInSeconds >= 0, nameof(commandTimeoutInSeconds));

            CommandText = commandText;
            CommandTimeoutInSeconds = commandTimeoutInSeconds;
        }

        /// <summary>The caller-supplied connection, or <see langword="null"/> when each cursor opens its own.</summary>
        public virtual DbConnection Connection => null;

        /// <summary>The factory used to create a connection, or <see langword="null"/> when <see cref="Connection"/> is supplied.</summary>
        public virtual DbProviderFactory ProviderFactory => null;

        /// <summary>The connection string used with <see cref="ProviderFactory"/>, or <see langword="null"/> when <see cref="Connection"/> is supplied.</summary>
        public virtual string ConnectionString => null;

        public string CommandText { get; }
        public int CommandTimeoutInSeconds { get; }
    }
}
