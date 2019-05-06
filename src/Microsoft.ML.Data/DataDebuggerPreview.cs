// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This class represents an eager 'preview' of a <see cref="IDataView"/>.
    /// </summary>
    public sealed class DataDebuggerPreview
    {
        internal static class Defaults
        {
            public const int MaxRows = 100;
        }

        public DataViewSchema Schema { get; }
        public ImmutableArray<ColumnInfo> ColumnView { get; }
        public ImmutableArray<RowInfo> RowView { get; }

        internal DataDebuggerPreview(IDataView data, int maxRows = Defaults.MaxRows)
        {
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckParam(maxRows >= 0, nameof(maxRows));
            Schema = data.Schema;

            int n = data.Schema.Count;

            var rows = new List<RowInfo>();
            var columns = new List<object>[n];
            for (int i = 0; i < columns.Length; i++)
                columns[i] = new List<object>();

            using (var cursor = data.GetRowCursorForAllColumns())
            {
                var setters = new Action<RowInfo, List<object>>[n];
                for (int i = 0; i < n; i++)
                    setters[i] = Utils.MarshalInvoke(MakeSetter<int>, data.Schema[i].Type.RawType, cursor, i);

                int count = 0;
                while (count < maxRows && cursor.MoveNext())
                {
                    var curRow = new RowInfo(n);
                    for (int i = 0; i < setters.Length; i++)
                        setters[i](curRow, columns[i]);

                    rows.Add(curRow);
                    count++;
                }
            }
            RowView = rows.ToImmutableArray();
            ColumnView = Enumerable.Range(0, n).Select(c => new ColumnInfo(data.Schema[c], columns[c].ToArray())).ToImmutableArray();
        }

        public override string ToString()
            => $"{Schema.Count} columns, {RowView.Length} rows";

        private Action<RowInfo, List<object>> MakeSetter<T>(DataViewRow row, int col)
        {
            var column = row.Schema[col];
            var getter = row.GetGetter<T>(column);
            Action<RowInfo, List<object>> result = (rowInfo, list) =>
            {
                T value = default;
                getter(ref value);
                rowInfo.Values[col] = new KeyValuePair<string, object>(column.Name, value);

                // Call getter again on another buffer, since we store it in two places.
                value = default;
                getter(ref value);
                list.Add(value);
            };
            return result;
        }

        public sealed class RowInfo
        {
            [System.Diagnostics.DebuggerBrowsable(System.Diagnostics.DebuggerBrowsableState.RootHidden)]
            public KeyValuePair<string, object>[] Values { get; }

            public override string ToString()
                => $"{Values.Length} columns";

            internal RowInfo(int n)
            {
                Values = new KeyValuePair<string, object>[n];
            }
        }

        public sealed class ColumnInfo
        {
            public DataViewSchema.Column Column { get; }
            public object[] Values { get; }

            public override string ToString() => $"{Column.Name}: {Column.Type}";

            internal ColumnInfo(DataViewSchema.Column column, object[] values)
            {
                Column = column;
                Values = values;
            }
        }
    }
}
