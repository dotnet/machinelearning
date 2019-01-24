// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Extension methods related to the ColumnType class.
    /// </summary>
    [BestFriend]
    internal static class ColumnTypeExtensions
    {
        /// <summary>
        /// Whether this type is a standard scalar type completely determined by its <see cref="ColumnType.RawType"/>
        /// (not a <see cref="KeyType"/> or <see cref="StructuredType"/>, etc).
        /// </summary>
        public static bool IsStandardScalar(this ColumnType columnType) =>
            (columnType is NumberType) || (columnType is TextType) || (columnType is BoolType) ||
            (columnType is TimeSpanType) || (columnType is DateTimeType) || (columnType is DateTimeOffsetType);

        /// <summary>
        /// Zero return means either it's not a key type or the cardinality is unknown.
        /// </summary>
        public static int GetKeyCount(this ColumnType columnType) => (columnType as KeyType)?.Count ?? 0;

        /// <summary>
        /// For non-vector types, this returns the column type itself (i.e., return <paramref name="columnType"/>).
        /// For vector types, this returns the type of the items stored as values in vector.
        /// </summary>
        public static ColumnType GetItemType(this ColumnType columnType) => (columnType as VectorType)?.ItemType ?? columnType;

        /// <summary>
        /// Zero return means either it's not a vector or the size is unknown.
        /// </summary>
        public static int GetVectorSize(this ColumnType columnType) => (columnType as VectorType)?.Size ?? 0;

        /// <summary>
        /// For non-vectors, this returns one. For unknown size vectors, it returns zero.
        /// For known sized vectors, it returns size.
        /// </summary>
        public static int GetValueCount(this ColumnType columnType) => (columnType as VectorType)?.Size ?? 1;

        /// <summary>
        /// Whether this is a vector type with known size. Returns false for non-vector types.
        /// Equivalent to <c><see cref="GetVectorSize"/> &gt; 0</c>.
        /// </summary>
        public static bool IsKnownSizeVector(this ColumnType columnType) => columnType.GetVectorSize() > 0;

        /// <summary>
        /// Gets the equivalent <see cref="DataKind"/> for the <paramref name="columnType"/>'s RawType.
        /// This can return default(<see cref="DataKind"/>) if the RawType doesn't have a corresponding
        /// <see cref="DataKind"/>.
        /// </summary>
        public static DataKind GetRawKind(this ColumnType columnType)
        {
            columnType.RawType.TryGetDataKind(out DataKind result);
            return result;
        }

        /// <summary>
        /// Equivalent to calling Equals(ColumnType) for non-vector types. For vector type,
        /// returns true if current and other vector types have the same size and item type.
        /// </summary>
        public static bool SameSizeAndItemType(this ColumnType columnType, ColumnType other)
        {
            if (other == null)
                return false;

            if (columnType.Equals(other))
                return true;

            // For vector types, we don't care about the factoring of the dimensions.
            if (!(columnType is VectorType vectorType) || !(other is VectorType otherVectorType))
                return false;
            if (!vectorType.ItemType.Equals(otherVectorType.ItemType))
                return false;
            return vectorType.Size == otherVectorType.Size;
        }
    }
}
