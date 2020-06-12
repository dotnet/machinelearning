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
        /// <summary>Creates a new instance of the <see cref="DatabaseSource" /> class.</summary>
        /// <param name="providerFactory">The factory used to create the <see cref="DbConnection"/>..</param>
        /// <param name="connectionString">The string used to open the connection.</param>
        /// <param name="commandText">The text command to run against the data source.</param>
        public DatabaseSource(DbProviderFactory providerFactory, string connectionString, string commandText)
        {
            Contracts.CheckValue(providerFactory, nameof(providerFactory));
            Contracts.CheckValue(connectionString, nameof(connectionString));
            Contracts.CheckValue(commandText, nameof(commandText));

            ProviderFactory = providerFactory;
            ConnectionString = connectionString;
            CommandText = commandText;
        }

        /// <summary>Gets the text command to run against the data source.</summary>
        public string CommandText { get; }

        /// <summary>Gets the string used to open the connection.</summary>
        public string ConnectionString { get; }

        /// <summary>Gets the factory used to create the <see cref="DbConnection"/>.</summary>
        public DbProviderFactory ProviderFactory { get; }
    }
}
