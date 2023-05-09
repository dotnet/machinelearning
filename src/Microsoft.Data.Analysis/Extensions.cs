// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Common;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public static class Extensions
    {
        public static DbDataAdapter CreateDataAdapter(this DbProviderFactory factory, DbConnection connection, string tableName)
        {
            var query = connection.CreateCommand();
            query.CommandText = $"SELECT * FROM {tableName}";
            var res = factory.CreateDataAdapter();
            res.SelectCommand = query;
            return res;
        }

        public static bool TryOpen(this DbConnection connection)
        {
            if (connection.State == ConnectionState.Closed)
            {
                connection.Open();
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
