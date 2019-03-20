// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Extension methods to facilitate easy consumption of popular contents of <see cref="DataViewSchema.Column.Annotations"/>.
    /// </summary>
    public static class SchemaAnnotationsExtensions
    {
        /// <summary>
        /// Returns <see langword="true"/> if the input column is of <see cref="VectorType"/>, and that has
        /// <c>SlotNames</c> annotation of a <see cref="VectorType"/> whose <see cref="VectorType.ItemType"/>
        /// is of <see cref="TextDataViewType"/>, and further whose <see cref="VectorType.Size"/> matches
        /// this input vector size.
        /// </summary>
        /// <param name="column">The column whose <see cref="DataViewSchema.Column.Annotations"/> will be queried.</param>
        /// <seealso cref="GetSlotNames(DataViewSchema.Column, ref VBuffer{ReadOnlyMemory{char}})"/>
        public static bool HasSlotNames(this DataViewSchema.Column column)
            => column.Type is VectorType vectorType
                && vectorType.Size > 0
                && column.HasSlotNames(vectorType.Size);

        /// <summary>
        /// Stores the slots names of the input column into the provided buffer, if there are slot names.
        /// Otherwise it will throw an exception.
        /// </summary>
        /// <seealso cref="HasSlotNames(DataViewSchema.Column)"/>
        /// <param name="column">The column whose <see cref="DataViewSchema.Column.Annotations"/> will be queried.</param>
        /// <param name="slotNames">The <see cref="VBuffer{T}"/> into which the slot names will be stored.</param>
        public static void GetSlotNames(this DataViewSchema.Column column, ref VBuffer<ReadOnlyMemory<char>> slotNames)
            => column.Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref slotNames);

        /// <summary>
        /// Returns <see langword="true"/> if the input column is of <see cref="VectorType"/>, and that has
        /// <c>SlotNames</c> annotation of a <see cref="VectorType"/> whose <see cref="VectorType.ItemType"/>
        /// is of <see cref="TextDataViewType"/>, and further whose <see cref="VectorType.Size"/> matches
        /// this input vector size.
        /// </summary>
        /// <param name="column">The column whose <see cref="DataViewSchema.Column.Annotations"/> will be queried.</param>
        /// <param name="keyValueItemType">The type of the individual key-values to query. A common,
        /// though not universal, type to provide is <see cref="TextDataViewType.Instance"/>, so if left unspecified
        /// this will be assumed to have the value <see cref="TextDataViewType.Instance"/>.</param>
        /// <seealso cref="GetKeyValues{TValue}(DataViewSchema.Column, ref VBuffer{TValue})"/>
        public static bool HasKeyValues(this DataViewSchema.Column column, PrimitiveDataViewType keyValueItemType = null)
        {
            // False if type is neither a key type, or a vector of key types.
            if (!(column.Type.GetItemType() is KeyType keyType))
                return false;

            if (keyValueItemType == null)
                keyValueItemType = TextDataViewType.Instance;

            var metaColumn = column.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues);
            return
                metaColumn != null
                && metaColumn.Value.Type is VectorType vectorType
                && keyType.Count == (ulong)vectorType.Size
                && keyValueItemType.Equals(vectorType.ItemType);
        }

        /// <summary>
        /// Stores the key values of the input colum into the provided buffer, if this is of key type and whose
        /// key values are of <see cref="VectorType.ItemType"/> whose <see cref="DataViewType.RawType"/> matches
        /// <typeparamref name="TValue"/>. If there is no matching key valued annotation this will throw an exception.
        /// </summary>
        /// <typeparam name="TValue">The type of the key values.</typeparam>
        /// <param name="column">The column whose <see cref="DataViewSchema.Column.Annotations"/> will be queried.</param>
        /// <param name="keyValues">The <see cref="VBuffer{T}"/> into which the key values will be stored.</param>
        public static void GetKeyValues<TValue>(this DataViewSchema.Column column, ref VBuffer<TValue> keyValues)
            => column.Annotations.GetValue(AnnotationUtils.Kinds.KeyValues, ref keyValues);

        /// <summary>
        /// Returns <see langword="true"/> if and only if <paramref name="column"/> has <c>IsNormalized</c> annotation
        /// set to <see langword="true"/>.
        /// </summary>
        public static bool IsNormalized(this DataViewSchema.Column column)
        {
            var metaColumn = column.Annotations.Schema.GetColumnOrNull((AnnotationUtils.Kinds.IsNormalized));
            if (metaColumn == null || !(metaColumn.Value.Type is BooleanDataViewType))
                return false;

            bool value = default;
            column.Annotations.GetValue(AnnotationUtils.Kinds.IsNormalized, ref value);
            return value;
        }
    }
}
