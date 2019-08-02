// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data.Common;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    public sealed class DatabaseSource
    {
        public DatabaseSource(DbProviderFactory providerFactory, string connectionString, string commandText = "")
        {
            Contracts.CheckValue(providerFactory, nameof(providerFactory));
            Contracts.CheckValue(connectionString, nameof(connectionString));
            Contracts.CheckValue(commandText, nameof(commandText));

            ProviderFactory = providerFactory;
            ConnectionString = connectionString;
            CommandText = commandText;
        }

        public string CommandText { get; }

        public string ConnectionString { get; }

        public DbProviderFactory ProviderFactory { get; }
    }
}
