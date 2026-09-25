// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data.Common;

namespace Microsoft.ML.Data
{
    /// <summary>Exposes the data required for opening a database for reading.</summary>
    public sealed class DatabaseSource
    {
        private readonly DatabaseSourceBase _innerSource;
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
            _innerSource = new DatabaseSourceUsingProviderFactory(providerFactory, connectionString, commandText, commandTimeoutInSeconds);
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
        {
            _innerSource = new DatabaseSourceUsingConnection(connection, commandText, commandTimeoutInSeconds);
        }

        /// <summary>Gets the timeout for database command.</summary>
        public int CommandTimeoutInSeconds => _innerSource.CommandTimeoutInSeconds;

        /// <summary>Gets the text command to run against the data source.</summary>
        public string CommandText => _innerSource.CommandText;

        /// <summary>Gets the string used to open the connection.</summary>
        /// <remarks>This is <see langword="null"/> when the source was created from an existing <see cref="DbConnection"/>.</remarks>
        public string ConnectionString => _innerSource.ConnectionString;

        /// <summary>Gets the factory used to create the <see cref="DbConnection"/>.</summary>
        /// <remarks>This is <see langword="null"/> when the source was created from an existing <see cref="DbConnection"/>.</remarks>
        public DbProviderFactory ProviderFactory => _innerSource.ProviderFactory;

        /// <summary>Gets the caller-supplied database connection.</summary>
        /// <remarks>
        /// This is <see langword="null"/> when the source was created from a <see cref="ProviderFactory"/>.
        /// In that case each cursor opens and disposes its own connection.
        /// </remarks>
        public DbConnection Connection => _innerSource.Connection;
    }
}
