// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data.Common;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>Exposes the data required for opening a database for reading.</summary>
    public sealed class DatabaseSource
    {
        private const int DefaultCommandTimeoutInSeconds = 30;

        /// <summary>Creates a new instance of the <see cref="DatabaseSource" /> class.</summary>
        /// <param name="providerFactory">The factory used to create the <see cref="DbConnection"/>..</param>
        /// <param name="connectionString">The string used to open the connection.</param>
        /// <param name="commandText">The text command to run against the data source.</param>
        public DatabaseSource(DbProviderFactory providerFactory, string connectionString, string commandText) :
            this(providerFactory, connectionString, commandText, DefaultCommandTimeoutInSeconds)
        {
        }

        /// <summary>Creates a new instance of the <see cref="DatabaseSource" /> class.</summary>
        /// <param name="providerFactory">The factory used to create the <see cref="DbConnection"/>..</param>
        /// <param name="connectionString">The string used to open the connection.</param>
        /// <param name="commandText">The text command to run against the data source.</param>
        /// <param name="commandTimeoutInSeconds">The timeout(in seconds) for database command.</param>
        public DatabaseSource(DbProviderFactory providerFactory, string connectionString, string commandText, int commandTimeoutInSeconds)
        {
            Contracts.CheckValue(providerFactory, nameof(providerFactory));
            Contracts.CheckValue(connectionString, nameof(connectionString));
            Contracts.CheckValue(commandText, nameof(commandText));
            Contracts.CheckUserArg(commandTimeoutInSeconds >= 0, nameof(commandTimeoutInSeconds));

            ProviderFactory = providerFactory;
            ConnectionString = connectionString;
            CommandText = commandText;
            CommandTimeoutInSeconds = commandTimeoutInSeconds;
        }

        /// <summary>Gets the timeout for database command.</summary>
        public int CommandTimeoutInSeconds { get; }

        /// <summary>Gets the text command to run against the data source.</summary>
        public string CommandText { get; }

        /// <summary>Gets the string used to open the connection.</summary>
        public string ConnectionString { get; }

        /// <summary>Gets the factory used to create the <see cref="DbConnection"/>.</summary>
        public DbProviderFactory ProviderFactory { get; }
    }
}
