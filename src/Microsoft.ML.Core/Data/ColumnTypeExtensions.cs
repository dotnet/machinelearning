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
    }
}
