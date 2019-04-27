// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
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
            return columnType is VectorDataViewType;
        }

        public static bool IsKey(this DataViewType columnType)
        {
            return columnType is KeyDataViewType;
        }

        public static bool IsKnownSizeVector(this DataViewType columnType)
        {
            var vector = columnType as VectorDataViewType;
            if (vector == null)
            {
                return false;
            }
            return vector.Size > 0;
        }

        public static DataViewType GetItemType(this DataViewType columnType)
        {
            var vector = columnType as VectorDataViewType;
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
            return (columnType as VectorDataViewType)?.Size ?? 0;
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
            return (columnType as KeyDataViewType)?.Count ?? 0;
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

        /// <summary>
        /// Equivalent to calling Equals(ColumnType) for non-vector types. For vector type,
        /// returns true if current and other vector types have the same size and item type.
        /// </summary>
        public static bool SameSizeAndItemType(this DataViewType columnType, DataViewType other)
        {
            if (other == null)
                return false;

            if (columnType.Equals(other))
                return true;

            // For vector types, we don't care about the factoring of the dimensions.
            if (!(columnType is VectorDataViewType vectorType) || !(other is VectorDataViewType otherVectorType))
                return false;
            if (!vectorType.ItemType.Equals(otherVectorType.ItemType))
                return false;
            return vectorType.Size == otherVectorType.Size;
        }
    }
}