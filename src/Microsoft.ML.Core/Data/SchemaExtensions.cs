// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.Data.DataView;

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal static class SchemaExtensions
    {
        public static Schema MakeSchema(IEnumerable<Schema.DetachedColumn> columns)
        {
            var builder = new SchemaBuilder();
            builder.AddColumns(columns);
            return builder.GetSchema();
        }

        /// <summary>
        /// Legacy method to get the column index.
        /// DO NOT USE: use <see cref="Schema.GetColumnOrNull"/> instead.
        /// </summary>
        public static bool TryGetColumnIndex(this Schema schema, string name, out int col)
        {
            col = schema.GetColumnOrNull(name)?.Index ?? -1;
            return col >= 0;
        }
    }
}
