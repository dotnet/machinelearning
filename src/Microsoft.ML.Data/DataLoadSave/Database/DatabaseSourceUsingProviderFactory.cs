// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data.Common;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    internal sealed class DatabaseSourceUsingProviderFactory : DatabaseSourceBase
    {
        private bool _disposed;

        private DbConnection _connection;

        public DatabaseSourceUsingProviderFactory(DbProviderFactory providerFactory, string connectionString, string commandText, int commandTimeoutInSeconds)
            : base(commandText, commandTimeoutInSeconds)
        {
            Contracts.CheckValue(providerFactory, nameof(providerFactory));
            Contracts.CheckNonEmpty(connectionString, nameof(connectionString));

            ProviderFactory = providerFactory;
            ConnectionString = connectionString;
        }

        public override DbConnection Connection
        {
            get
            {
                if (_connection is null)
                {
                    _connection = ProviderFactory.CreateConnection();
                    _connection.ConnectionString = ConnectionString;
                    _connection.Open();
                }
                return _connection;
            }
        }

        public DbProviderFactory ProviderFactory { get; }
        public string ConnectionString { get; }

        protected override void Dispose(bool disposing)
        {
            if (_disposed) return;
            if (disposing)
            {
                _connection?.Dispose();
            }
            _disposed = true;
            base.Dispose(disposing);
        }
    }
}
