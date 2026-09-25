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
        private enum DatabaseSourceType
        {
            UsingProviderFactory,
            UsingConnection
        }

        private readonly DatabaseSourceType _sourceType;
        private readonly DbProviderFactory _providerFactory;
        private readonly string _connectionString;
        private readonly DbConnection _connection;
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
            : this(commandText, commandTimeoutInSeconds)
        {
            Contracts.CheckValue(providerFactory, nameof(providerFactory));
            Contracts.CheckNonEmpty(connectionString, nameof(connectionString));

            _sourceType = DatabaseSourceType.UsingProviderFactory;
            _providerFactory = providerFactory;
            _connectionString = connectionString;
        }

        /// <summary>Creates a new instance of the <see cref="DatabaseSource" /> class.</summary>
        /// <param name="connection">The database connection. The loader does not dispose this connection.</param>
        /// <param name="commandText">The text command to run against the data source.</param>
        public DatabaseSource(DbConnection connection, string commandText) :
            this(connection, commandText, DefaultCommandTimeoutInSeconds)
        {
        }

        /// <summary>Creates a new instance of the <see cref="DatabaseSource" /> class.</summary>
        /// <param name="connection">The database connection. The loader does not dispose this connection.</param>
        /// <param name="commandText">The text command to run against the data source.</param>
        /// <param name="commandTimeoutInSeconds">The timeout(in seconds) for database command.</param>
        public DatabaseSource(DbConnection connection, string commandText, int commandTimeoutInSeconds)
            : this(commandText, commandTimeoutInSeconds)
        {
            Contracts.CheckValue(connection, nameof(connection));

            _sourceType = DatabaseSourceType.UsingConnection;
            _connection = connection;
        }

        private DatabaseSource(string commandText, int commandTimeoutInSeconds)
        {
            Contracts.CheckValue(commandText, nameof(commandText));
            Contracts.CheckUserArg(commandTimeoutInSeconds >= 0, nameof(commandTimeoutInSeconds));

            CommandText = commandText;
            CommandTimeoutInSeconds = commandTimeoutInSeconds;
        }

        /// <summary>Gets the timeout for database command.</summary>
        public int CommandTimeoutInSeconds { get; }

        /// <summary>Gets the text command to run against the data source.</summary>
        public string CommandText { get; }

        /// <summary>Gets the string used to open the connection.</summary>
        public string ConnectionString => _sourceType == DatabaseSourceType.UsingProviderFactory ? _connectionString : null;

        /// <summary>Gets the factory used to create the <see cref="DbConnection"/>.</summary>
        public DbProviderFactory ProviderFactory => _sourceType == DatabaseSourceType.UsingProviderFactory ? _providerFactory : null;

        /// <summary>Gets the caller-supplied database connection.</summary>
        public DbConnection Connection => _sourceType == DatabaseSourceType.UsingConnection ? _connection : null;
    }
}
