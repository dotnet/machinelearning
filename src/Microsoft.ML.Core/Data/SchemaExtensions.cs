// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal static class SchemaExtensions
    {
        public static DataViewSchema MakeSchema(IEnumerable<DataViewSchema.DetachedColumn> columns)
        {
            var builder = new DataViewSchema.Builder();
            builder.AddColumns(columns);
            return builder.ToSchema();
        }

        /// <summary>
        /// Legacy method to get the column index.
        /// DO NOT USE: use <see cref="DataViewSchema.GetColumnOrNull"/> instead.
        /// </summary>
        public static bool TryGetColumnIndex(this DataViewSchema schema, string name, out int col)
        {
            col = schema.GetColumnOrNull(name)?.Index ?? -1;
            return col >= 0;
        }
    }
}
