// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class DatasetDimensionsUtil
    {
        public static int GetTextColumnCardinality(IDataView data, int colIndex)
        {
            var seen = new HashSet<string>();
            using (var cursor = data.GetRowCursor(x => x == colIndex))
            {
                var getter = cursor.GetGetter<ReadOnlyMemory<char>>(colIndex);
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

        public static bool HasMissingNumericSingleValue(IDataView data, int colIndex)
        {
            using (var cursor = data.GetRowCursor(x => x == colIndex))
            {
                var getter = cursor.GetGetter<Single>(colIndex);
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

        public static bool HasMissingNumericVector(IDataView data, int colIndex)
        {
            using (var cursor = data.GetRowCursor(x => x == colIndex))
            {
                var getter = cursor.GetGetter<VBuffer<Single>>(colIndex);
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
    }
}
