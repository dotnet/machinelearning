﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

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
        /// <param name="env">The current host environment.</param>
        /// <param name="columnName">The name of the column to extract.</param>
        public static IEnumerable<T> GetColumn<T>(this IDataView data, IHostEnvironment env, string columnName)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));
            env.CheckNonEmpty(columnName, nameof(columnName));

            if (!data.Schema.TryGetColumnIndex(columnName, out int col))
                throw env.ExceptSchemaMismatch(nameof(columnName), "input", columnName);

            // There are two decisions that we make here:
            // - Is the T an array type?
            //     - If yes, we need to map VBuffer to array and densify.
            //     - If no, this is not needed.
            // - Does T (or item type of T if it's an array) equal to the data view type?
            //     - If this is the same type, we can map directly.
            //     - Otherwise, we need a conversion delegate.

            var colType = data.Schema[col].Type;
            if (colType.RawType == typeof(T))
            {
                // Direct mapping is possible.
                return GetColumnDirect<T>(data, col);
            }
            else if (typeof(T) == typeof(string) && colType.IsText)
            {
                // Special case of ROM<char> to string conversion.
                Delegate convert = (Func<ReadOnlyMemory<char>, string>)((ReadOnlyMemory<char> txt) => txt.ToString());
                Func<IDataView, int, Func<int, T>, IEnumerable<T>> del = GetColumnConvert;
                var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(typeof(T), colType.RawType);
                return (IEnumerable<T>)(meth.Invoke(null, new object[] { data, col, convert }));
            }
            else if (typeof(T).IsArray)
            {
                // Output is an array type.
                if (!colType.IsVector)
                    throw env.ExceptSchemaMismatch(nameof(columnName), "input", columnName, "vector", "scalar");
                var elementType = typeof(T).GetElementType();
                if (elementType == colType.ItemType.RawType)
                {
                    // Direct mapping of items.
                    Func<IDataView, int, IEnumerable<int[]>> del = GetColumnArrayDirect<int>;
                    var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(elementType);
                    return (IEnumerable<T>)meth.Invoke(null, new object[] { data, col });
                }
                else if (elementType == typeof(string) && colType.ItemType.IsText)
                {
                    // Conversion of DvText items to string items.
                    Delegate convert = (Func<ReadOnlyMemory<char>, string>)((ReadOnlyMemory<char> txt) => txt.ToString());
                    Func<IDataView, int, Func<int, long>, IEnumerable<long[]>> del = GetColumnArrayConvert;
                    var meth = del.Method.GetGenericMethodDefinition().MakeGenericMethod(elementType, colType.ItemType.RawType);
                    return (IEnumerable<T>)meth.Invoke(null, new object[] { data, col, convert });
                }
                // Fall through to the failure.
            }
            throw env.Except($"Could not map a data view column '{columnName}' of type {colType} to {typeof(T)}.");
        }

        private static IEnumerable<T> GetColumnDirect<T>(IDataView data, int col)
        {
            Contracts.AssertValue(data);
            Contracts.Assert(0 <= col && col < data.Schema.Count);

            using (var cursor = data.GetRowCursor(col.Equals))
            {
                var getter = cursor.GetGetter<T>(col);
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

            using (var cursor = data.GetRowCursor(col.Equals))
            {
                var getter = cursor.GetGetter<TData>(col);
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

            using (var cursor = data.GetRowCursor(col.Equals))
            {
                var getter = cursor.GetGetter<VBuffer<T>>(col);
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

            using (var cursor = data.GetRowCursor(col.Equals))
            {
                var getter = cursor.GetGetter<VBuffer<TData>>(col);
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
