// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML
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
    /// The debugger proxy for <see cref="DataViewSchema.Annotations"/>.
    /// </summary>
    internal sealed class AnnotationsDebuggerProxy
    {
        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public IReadOnlyList<KeyValuePair<string, object>> Values { get; }

        public AnnotationsDebuggerProxy(DataViewSchema.Annotations annotations)
        {
            Values = BuildValues(annotations);
        }

        private static List<KeyValuePair<string, object>> BuildValues(DataViewSchema.Annotations annotations)
        {
            var result = new List<KeyValuePair<string, object>>();
            foreach (var column in annotations.Schema)
            {
                var name = column.Name;
                var value = Utils.MarshalInvoke(GetValue<DataViewSchema.Column>, column.Type.RawType, annotations, column);
                result.Add(new KeyValuePair<string, object>(name, value));
            }
            return result;
        }

        private static object GetValue<T>(DataViewSchema.Annotations annotations, DataViewSchema.Column column)
        {
            T value = default;
            annotations.GetGetter<T>(column)(ref value);
            return value;
        }
    }
}
