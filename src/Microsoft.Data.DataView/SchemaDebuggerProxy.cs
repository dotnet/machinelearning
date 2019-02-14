// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Microsoft.Data.DataView
{
    /// <summary>
    /// The debugger proxy for <see cref="DataViewSchema"/>.
    /// </summary>
    internal sealed class SchemaDebuggerProxy
    {
        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public DataViewSchema.Column[] Columns { get; }

        public SchemaDebuggerProxy(DataViewSchema schema)
        {
            Columns = Enumerable.Range(0, schema.Count).Select(x => schema[x]).ToArray();
        }
    }

    /// <summary>
    /// The debugger proxy for <see cref="DataViewSchema.Metadata"/>.
    /// </summary>
    internal sealed class MetadataDebuggerProxy
    {
        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public IReadOnlyList<KeyValuePair<string, object>> Values { get; }

        public MetadataDebuggerProxy(DataViewSchema.Metadata metadata)
        {
            Values = BuildValues(metadata);
        }

        private static List<KeyValuePair<string, object>> BuildValues(DataViewSchema.Metadata metadata)
        {
            var result = new List<KeyValuePair<string, object>>();
            foreach (var column in metadata.Schema)
            {
                var name = column.Name;
                var value = Utils.MarshalInvoke(GetValue<int>, column.Type.RawType, metadata, column.Index);
                result.Add(new KeyValuePair<string, object>(name, value));
            }
            return result;
        }

        private static object GetValue<T>(DataViewSchema.Metadata metadata, int columnIndex)
        {
            T value = default;
            metadata.GetGetter<T>(columnIndex)(ref value);
            return value;
        }
    }
}
