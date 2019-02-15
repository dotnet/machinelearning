// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Extension methods related to the ColumnType class.
    /// </summary>
    [BestFriend]
    internal static class ColumnTypeExtensions
    {
        /// <summary>
        /// Whether this type is a standard scalar type completely determined by its <see cref="DataViewType.RawType"/>
        /// (not a <see cref="KeyType"/> or <see cref="StructuredDataViewType"/>, etc).
        /// </summary>
        public static bool IsStandardScalar(this DataViewType columnType) =>
            (columnType is NumberDataViewType) || (columnType is TextDataViewType) || (columnType is BooleanDataViewType) ||
            (columnType is TimeSpanDataViewType) || (columnType is DateTimeDataViewType) || (columnType is DateTimeOffsetDataViewType);

        /// <summary>
        /// Zero return means it's not a key type.
        /// </summary>
        public static ulong GetKeyCount(this DataViewType columnType) => (columnType as KeyType)?.Count ?? 0;

        /// <summary>
        /// Sometimes it is necessary to cast the Count to an int. This performs overflow check.
        /// Zero return means it's not a key type.
        /// </summary>
        public static int GetKeyCountAsInt32(this DataViewType columnType, IExceptionContext ectx = null)
        {
            ulong count = columnType.GetKeyCount();
            ectx.Check(count <= int.MaxValue, nameof(KeyType) + "." + nameof(KeyType.Count) + " exceeds int.MaxValue.");
            return (int)count;
        }

        /// <summary>
        /// For non-vector types, this returns the column type itself (i.e., return <paramref name="columnType"/>).
        /// For vector types, this returns the type of the items stored as values in vector.
        /// </summary>
        public static DataViewType GetItemType(this DataViewType columnType) => (columnType as VectorType)?.ItemType ?? columnType;

        /// <summary>
        /// Zero return means either it's not a vector or the size is unknown.
        /// </summary>
        public static int GetVectorSize(this DataViewType columnType) => (columnType as VectorType)?.Size ?? 0;

        /// <summary>
        /// For non-vectors, this returns one. For unknown size vectors, it returns zero.
        /// For known sized vectors, it returns size.
        /// </summary>
        public static int GetValueCount(this DataViewType columnType) => (columnType as VectorType)?.Size ?? 1;

        /// <summary>
        /// Whether this is a vector type with known size. Returns false for non-vector types.
        /// Equivalent to <c><see cref="GetVectorSize"/> &gt; 0</c>.
        /// </summary>
        public static bool IsKnownSizeVector(this DataViewType columnType) => columnType.GetVectorSize() > 0;

        /// <summary>
        /// Gets the equivalent <see cref="DataKind"/> for the <paramref name="columnType"/>'s RawType.
        /// This can return default(<see cref="DataKind"/>) if the RawType doesn't have a corresponding
        /// <see cref="DataKind"/>.
        /// </summary>
        public static DataKind GetRawKind(this DataViewType columnType)
        {
            columnType.RawType.TryGetDataKind(out DataKind result);
            return result;
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
            if (!(columnType is VectorType vectorType) || !(other is VectorType otherVectorType))
                return false;
            if (!vectorType.ItemType.Equals(otherVectorType.ItemType))
                return false;
            return vectorType.Size == otherVectorType.Size;
        }

        public static PrimitiveDataViewType PrimitiveTypeFromType(Type type)
        {
            if (type == typeof(ReadOnlyMemory<char>) || type == typeof(string))
                return TextDataViewType.Instance;
            if (type == typeof(bool))
                return BooleanDataViewType.Instance;
            if (type == typeof(TimeSpan))
                return TimeSpanDataViewType.Instance;
            if (type == typeof(DateTime))
                return DateTimeDataViewType.Instance;
            if (type == typeof(DateTimeOffset))
                return DateTimeOffsetDataViewType.Instance;
            return NumberTypeFromType(type);
        }

        public static PrimitiveDataViewType PrimitiveTypeFromKind(DataKind kind)
        {
            if (kind == DataKind.TX)
                return TextDataViewType.Instance;
            if (kind == DataKind.BL)
                return BooleanDataViewType.Instance;
            if (kind == DataKind.TS)
                return TimeSpanDataViewType.Instance;
            if (kind == DataKind.DT)
                return DateTimeDataViewType.Instance;
            if (kind == DataKind.DZ)
                return DateTimeOffsetDataViewType.Instance;
            return NumberTypeFromKind(kind);
        }

        public static NumberDataViewType NumberTypeFromType(Type type)
        {
            DataKind kind;
            if (type.TryGetDataKind(out kind))
                return NumberTypeFromKind(kind);

            Contracts.Assert(false);
            throw new InvalidOperationException($"Bad type in {nameof(ColumnTypeExtensions)}.{nameof(NumberTypeFromType)}: {type}");
        }

        public static NumberDataViewType NumberTypeFromKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.I1:
                    return NumberDataViewType.SByte;
                case DataKind.U1:
                    return NumberDataViewType.Byte;
                case DataKind.I2:
                    return NumberDataViewType.Int16;
                case DataKind.U2:
                    return NumberDataViewType.UInt16;
                case DataKind.I4:
                    return NumberDataViewType.Int32;
                case DataKind.U4:
                    return NumberDataViewType.UInt32;
                case DataKind.I8:
                    return NumberDataViewType.Int64;
                case DataKind.U8:
                    return NumberDataViewType.UInt64;
                case DataKind.R4:
                    return NumberDataViewType.Single;
                case DataKind.R8:
                    return NumberDataViewType.Double;
                case DataKind.UG:
                    return NumberDataViewType.DataViewRowId;
            }

            Contracts.Assert(false);
            throw new InvalidOperationException($"Bad data kind in {nameof(ColumnTypeExtensions)}.{nameof(NumberTypeFromKind)}: {kind}");
        }
    }
}
