// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Extension methods that allow to extract values of a single column of an <see cref="IDataView"/> as an
    /// <see cref="IEnumerable{T}"/>.
    /// </summary>
    public static class ColumnCursorExtensions
    {

        /// <summary>
        /// Extract all values of one column of the data view in a form of an <see cref="IEnumerable{T}"/>.
        /// </summary>
        /// <typeparam name="T">The type of the values. This must match the actual column type.</typeparam>
        /// <param name="data">The data view to get the column from.</param>
        /// <param name="columnName">The name of the column to be extracted.</param>

        public static IEnumerable<T> GetColumn<T>(this IDataView data, string columnName)
            => GetColumn<T>(data, data.Schema[columnName]);

        /// <summary>
        /// Extract all values of one column of the data view in a form of an <see cref="IEnumerable{T}"/>.
        /// </summary>
        /// <typeparam name="T">The type of the values. This must match the actual column type.</typeparam>
        /// <param name="data">The data view to get the column from.</param>
        /// <param name="column">The column to be extracted.</param>
        public static IEnumerable<T> GetColumn<T>(this IDataView data, DataViewSchema.Column column)
        {
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckNonEmpty(column.Name, nameof(column));

            var colIndex = column.Index;
            var colType = column.Type;
            var colName = column.Name;

            // Use column index as the principle address of the specified input column and check if that address in data contains
            // the column indicated.
            if (data.Schema[colIndex].Name != colName || data.Schema[colIndex].Type != colType)
                throw Contracts.ExceptParam(nameof(column), string.Format("column with name {0}, type {1}, and index {2} cannot be found in {3}",
                    colName, colType, colIndex, nameof(data)));

            // There are two decisions that we make here:
            // - Is the T an array type?
            //     - If yes, we need to map VBuffer to array and densify.
            //     - If no, this is not needed.
            // - Does T (or item type of T if it's an array) equal to the data view type?
            //     - If this is the same type, we can map directly.
            //     - Otherwise, we need a conversion delegate.

            if (colType.RawType == typeof(T))
            {
                // Direct mapping is possible.
                return GetColumnDirect<T>(data, colIndex);
            }
            else if (typeof(T) == typeof(string) && colType is TextDataViewType)
            {
                // Special case of ROM<char> to string conversion.
                Delegate convert = (Func<ReadOnlyMemory<char>, string>)((ReadOnlyMemory<char> txt) => txt.ToString());
                Func<IDataView, int, Func<int, T>, IEnumerable<T>> del = GetColumnConvert;
                var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(typeof(T), colType.RawType);
                return (IEnumerable<T>)(meth.Invoke(null, new object[] { data, colIndex, convert }));
            }
            else if (typeof(T).IsArray)
            {
                // Output is an array type.
                if (!(colType is VectorType colVectorType))
                    throw Contracts.ExceptParam(nameof(column), string.Format("Cannot load vector type, {0}, specified in {1} to the user-defined type, {2}.", column.Type, nameof(column), typeof(T)));
                var elementType = typeof(T).GetElementType();
                if (elementType == colVectorType.ItemType.RawType)
                {
                    // Direct mapping of items.
                    Func<IDataView, int, IEnumerable<int[]>> del = GetColumnArrayDirect<int>;
                    var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(elementType);
                    return (IEnumerable<T>)meth.Invoke(null, new object[] { data, colIndex });
                }
                else if (elementType == typeof(string) && colVectorType.ItemType is TextDataViewType)
                {
                    // Conversion of DvText items to string items.
                    Delegate convert = (Func<ReadOnlyMemory<char>, string>)((ReadOnlyMemory<char> txt) => txt.ToString());
                    Func<IDataView, int, Func<int, long>, IEnumerable<long[]>> del = GetColumnArrayConvert;
                    var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(elementType, colVectorType.ItemType.RawType);
                    return (IEnumerable<T>)meth.Invoke(null, new object[] { data, colIndex, convert });
                }
                // Fall through to the failure.
            }

            throw Contracts.ExceptParam(nameof(column), string.Format("Cannot map column (name: {0}, type: {1}) in {2} to the user-defined type, {3}.",
                column.Name, column.Type, nameof(data), typeof(T)));
        }

        private static IEnumerable<T> GetColumnDirect<T>(IDataView data, int col)
        {
            Contracts.AssertValue(data);
            Contracts.Assert(0 <= col && col < data.Schema.Count);

            var column = data.Schema[col];
            using (var cursor = data.GetRowCursor(column))
            {
                var getter = cursor.GetGetter<T>(column);
                T curValue = default;
                while (cursor.MoveNext())
                {
                    getter(ref curValue);
                    yield return curValue;
                }
            }
        }

        private static IEnumerable<TOut> GetColumnConvert<TOut, TData>(IDataView data, int col, Func<TData, TOut> convert)
        {
            Contracts.AssertValue(data);
            Contracts.Assert(0 <= col && col < data.Schema.Count);

            var column = data.Schema[col];
            using (var cursor = data.GetRowCursor(column))
            {
                var getter = cursor.GetGetter<TData>(column);
                TData curValue = default;
                while (cursor.MoveNext())
                {
                    getter(ref curValue);
                    yield return convert(curValue);
                }
            }
        }

        private static IEnumerable<T[]> GetColumnArrayDirect<T>(IDataView data, int col)
        {
            Contracts.AssertValue(data);
            Contracts.Assert(0 <= col && col < data.Schema.Count);

            var column = data.Schema[col];
            using (var cursor = data.GetRowCursor(column))
            {
                var getter = cursor.GetGetter<VBuffer<T>>(column);
                VBuffer<T> curValue = default;
                while (cursor.MoveNext())
                {
                    getter(ref curValue);
                    // REVIEW: should we introduce the 'reuse array' logic here?
                    // For now it re-creates the array and densifies.
                    var dst = new T[curValue.Length];
                    curValue.CopyTo(dst);
                    yield return dst;
                }
            }
        }

        private static IEnumerable<TOut[]> GetColumnArrayConvert<TOut, TData>(IDataView data, int col, Func<TData, TOut> convert)
        {
            Contracts.AssertValue(data);
            Contracts.Assert(0 <= col && col < data.Schema.Count);

            var column = data.Schema[col];
            using (var cursor = data.GetRowCursor(column))
            {
                var getter = cursor.GetGetter<VBuffer<TData>>(column);
                VBuffer<TData> curValue = default;
                while (cursor.MoveNext())
                {
                    getter(ref curValue);
                    // REVIEW: should we introduce the 'reuse array' logic here?
                    // For now it re-creates the array and densifies.
                    var dst = new TOut[curValue.Length];
                    foreach (var kvp in curValue.Items(all: false))
                        dst[kvp.Key] = convert(kvp.Value);
                    yield return dst;
                }
            }
        }
    }
}
