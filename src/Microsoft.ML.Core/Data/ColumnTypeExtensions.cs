// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

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
            (columnType is RowIdDataViewType) || (columnType is TimeSpanDataViewType) ||
            (columnType is DateTimeDataViewType) || (columnType is DateTimeOffsetDataViewType);

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
        /// Gets the equivalent <see cref="InternalDataKind"/> for the <paramref name="columnType"/>'s RawType.
        /// This can return default(<see cref="InternalDataKind"/>) if the RawType doesn't have a corresponding
        /// <see cref="InternalDataKind"/>.
        /// </summary>
        public static InternalDataKind GetRawKind(this DataViewType columnType)
        {
            columnType.RawType.TryGetDataKind(out InternalDataKind result);
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
            if (type == typeof(DataViewRowId))
                return RowIdDataViewType.Instance;
            return NumberTypeFromType(type);
        }

        public static PrimitiveDataViewType PrimitiveTypeFromKind(InternalDataKind kind)
        {
            if (kind == InternalDataKind.TX)
                return TextDataViewType.Instance;
            if (kind == InternalDataKind.BL)
                return BooleanDataViewType.Instance;
            if (kind == InternalDataKind.TS)
                return TimeSpanDataViewType.Instance;
            if (kind == InternalDataKind.DT)
                return DateTimeDataViewType.Instance;
            if (kind == InternalDataKind.DZ)
                return DateTimeOffsetDataViewType.Instance;
            if (kind == InternalDataKind.UG)
                return RowIdDataViewType.Instance;
            return NumberTypeFromKind(kind);
        }

        public static NumberDataViewType NumberTypeFromType(Type type)
        {
            InternalDataKind kind;
            if (type.TryGetDataKind(out kind))
                return NumberTypeFromKind(kind);

            Contracts.Assert(false);
            throw new InvalidOperationException($"Bad type in {nameof(ColumnTypeExtensions)}.{nameof(NumberTypeFromType)}: {type}");
        }

        private static NumberDataViewType NumberTypeFromKind(InternalDataKind kind)
        {
            switch (kind)
            {
                case InternalDataKind.I1:
                    return NumberDataViewType.SByte;
                case InternalDataKind.U1:
                    return NumberDataViewType.Byte;
                case InternalDataKind.I2:
                    return NumberDataViewType.Int16;
                case InternalDataKind.U2:
                    return NumberDataViewType.UInt16;
                case InternalDataKind.I4:
                    return NumberDataViewType.Int32;
                case InternalDataKind.U4:
                    return NumberDataViewType.UInt32;
                case InternalDataKind.I8:
                    return NumberDataViewType.Int64;
                case InternalDataKind.U8:
                    return NumberDataViewType.UInt64;
                case InternalDataKind.R4:
                    return NumberDataViewType.Single;
                case InternalDataKind.R8:
                    return NumberDataViewType.Double;
            }

            Contracts.Assert(false);
            throw new InvalidOperationException($"Bad data kind in {nameof(ColumnTypeExtensions)}.{nameof(NumberTypeFromKind)}: {kind}");
        }
    }
}
