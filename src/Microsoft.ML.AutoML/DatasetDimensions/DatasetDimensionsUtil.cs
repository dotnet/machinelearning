// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.AutoML
{
    internal static class DatasetDimensionsUtil
    {
        public static int GetTextColumnCardinality(IDataView data, DataViewSchema.Column column)
        {
            var seen = new HashSet<string>();
            using (var cursor = data.GetRowCursor(new[] { column }))
            {
                var getter = cursor.GetGetter<ReadOnlyMemory<char>>(column);
                while (cursor.MoveNext())
                {
                    var value = default(ReadOnlyMemory<char>);
                    getter(ref value);
                    var valueStr = value.ToString();
                    seen.Add(valueStr);
                }
            }
            return seen.Count;
        }

        public static bool HasMissingNumericSingleValue(IDataView data, DataViewSchema.Column column)
        {
            using (var cursor = data.GetRowCursor(new[] { column }))
            {
                var getter = cursor.GetGetter<Single>(column);
                var value = default(Single);
                while (cursor.MoveNext())
                {
                    getter(ref value);
                    if (Single.IsNaN(value))
                    {
                        return true;
                    }
                }
                return false;
            }
        }

        public static bool HasMissingNumericVector(IDataView data, DataViewSchema.Column column)
        {
            using (var cursor = data.GetRowCursor(new[] { column }))
            {
                var getter = cursor.GetGetter<VBuffer<Single>>(column);
                var value = default(VBuffer<Single>);
                while (cursor.MoveNext())
                {
                    getter(ref value);
                    if (VBufferUtils.HasNaNs(value))
                    {
                        return true;
                    }
                }
                return false;
            }
        }

        public static ulong CountRows(IDataView data, ulong maxRows)
        {
            using (var cursor = data.GetRowCursor(new[] { data.Schema[0] }))
            {
                ulong rowCount = 0;
                while (cursor.MoveNext())
                {
                    if (++rowCount == maxRows)
                    {
                        break;
                    }
                }
                return rowCount;
            }
        }

        public static bool IsDataViewEmpty(IDataView data)
        {
            return CountRows(data, 1) == 0;
        }
    }
}
