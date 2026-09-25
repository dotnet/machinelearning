// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data.Common;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    internal sealed class DatabaseSourceUsingProviderFactory : DatabaseSourceBase
    {
        public DatabaseSourceUsingProviderFactory(DbProviderFactory providerFactory, string connectionString, string commandText, int commandTimeoutInSeconds)
            : base(commandText, commandTimeoutInSeconds)
        {
            Contracts.CheckValue(providerFactory, nameof(providerFactory));
            Contracts.CheckNonEmpty(connectionString, nameof(connectionString));

            ProviderFactory = providerFactory;
            ConnectionString = connectionString;
        }

        public override DbProviderFactory ProviderFactory { get; }
        public override string ConnectionString { get; }
    }
}
