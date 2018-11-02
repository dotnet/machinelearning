// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Core.Data
{
    /// <summary>
    /// The debugger proxy for <see cref="Schema"/>.
    /// </summary>
    internal sealed class SchemaDebuggerProxy
    {
        [System.Diagnostics.DebuggerBrowsable(System.Diagnostics.DebuggerBrowsableState.RootHidden)]

        public Schema.Column[] Columns { get; }

        public SchemaDebuggerProxy(Schema schema)
        {
            Columns = Enumerable.Range(0, schema.ColumnCount).Select(x => schema[x]).ToArray();
        }
    }

    /// <summary>
    /// The debugger proxy for <see cref="Schema.Metadata"/>.
    /// </summary>
    internal sealed class MetadataDebuggerProxy
    {
        [System.Diagnostics.DebuggerBrowsable(System.Diagnostics.DebuggerBrowsableState.RootHidden)]
        public IReadOnlyList<KeyValuePair<string, object>> Values { get; }

        public MetadataDebuggerProxy(Schema.Metadata metadata)
        {
            Values = BuildValues(metadata);
        }

        private static List<KeyValuePair<string, object>> BuildValues(Schema.Metadata metadata)
        {
            var result = new List<KeyValuePair<string, object>>();
            foreach ((var index, var column) in metadata.Schema.GetColumns())
            {
                var name = column.Name;
                var value = Utils.MarshalInvoke(GetValue<int>, column.Type.RawType, metadata, index);
                result.Add(new KeyValuePair<string, object>(name, value));
            }
            return result;
        }

        private static object GetValue<T>(Schema.Metadata metadata, int columnIndex)
        {
            T value = default;
            metadata.GetGetter<T>(columnIndex)(ref value);
            return value;
        }
    }
}
