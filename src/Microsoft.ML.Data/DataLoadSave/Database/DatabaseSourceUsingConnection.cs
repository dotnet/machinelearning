// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data.Common;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    internal sealed class DatabaseSourceUsingConnection : DatabaseSourceBase
    {
        /// <summary>Creates a new instance of the <see cref="DatabaseSourceUsingConnection" /> class.</summary>
        /// <param name="connection">The database connection.</param>
        /// <param name="commandText">The text command to run against the data source.</param>
        /// <param name="commandTimeoutInSeconds">The timeout(in seconds) for database command.</param>
        public DatabaseSourceUsingConnection(DbConnection connection, string commandText, int commandTimeoutInSeconds)
            : base(commandText, commandTimeoutInSeconds)
        {
            Contracts.CheckValue(connection, nameof(connection));

            Connection = connection;
        }

        public override DbConnection Connection { get; }
    }
}
