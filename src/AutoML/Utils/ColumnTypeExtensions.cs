// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class ColumnTypeExtensions
    {
        public static bool IsNumber(this ColumnType columnType)
        {
            return columnType is NumberType;
        }

        public static bool IsText(this ColumnType columnType)
        {
            return columnType is TextType;
        }

        public static bool IsBool(this ColumnType columnType)
        {
            return columnType is BoolType;
        }

        public static bool IsVector(this ColumnType columnType)
        {
            return columnType is VectorType;
        }

        public static bool IsKnownSizeVector(this ColumnType columnType)
        {
            var vector = columnType as VectorType;
            if(vector == null)
            {
                return false;
            }
            return vector.Size > 0;
        }

        public static ColumnType ItemType(this ColumnType columnType)
        {
            var vector = columnType as VectorType;
            if (vector == null)
            {
                return columnType;
            }
            return vector.ItemType;
        }

        public static DataKind RawKind(this ColumnType columnType)
        {
            columnType.RawType.TryGetDataKind(out var rawKind);
            return rawKind;
        }
    }
}
