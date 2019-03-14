// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class DataViewTypeExtensions
    {
        public static bool IsNumber(this DataViewType columnType)
        {
            return columnType is NumberDataViewType;
        }

        public static bool IsText(this DataViewType columnType)
        {
            return columnType is TextDataViewType;
        }

        public static bool IsBool(this DataViewType columnType)
        {
            return columnType is BooleanDataViewType;
        }

        public static bool IsVector(this DataViewType columnType)
        {
            return columnType is VectorType;
        }

        public static bool IsKey(this DataViewType columnType)
        {
            return columnType is KeyType;
        }

        public static bool IsKnownSizeVector(this DataViewType columnType)
        {
            var vector = columnType as VectorType;
            if (vector == null)
            {
                return false;
            }
            return vector.Size > 0;
        }

        public static DataViewType GetItemType(this DataViewType columnType)
        {
            var vector = columnType as VectorType;
            if (vector == null)
            {
                return columnType;
            }
            return vector.ItemType;
        }

        /// <summary>
        /// Zero return means either it's not a vector or the size is unknown.
        /// </summary>
        public static int GetVectorSize(this DataViewType columnType)
        {
            return (columnType as VectorType)?.Size ?? 0;
        }

        public static DataKind GetRawKind(this DataViewType columnType)
        {
            columnType.RawType.TryGetDataKind(out var rawKind);
            return rawKind;
        }

        /// <summary>
        /// Zero return means it's not a key type.
        /// </summary>
        public static ulong GetKeyCount(this DataViewType columnType)
        {
            return (columnType as KeyType)?.Count ?? 0;
        }

        /// <summary>
        /// Sometimes it is necessary to cast the Count to an int. This performs overflow check.
        /// Zero return means it's not a key type.
        /// </summary>
        public static int GetKeyCountAsInt32(this DataViewType columnType, IExceptionContext ectx = null)
        {
            ulong keyCount = columnType.GetKeyCount();
            return (int)keyCount;
        }
    }
}