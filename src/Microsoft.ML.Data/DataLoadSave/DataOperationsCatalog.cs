﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// A catalog of operations over data that are not transformers or estimators.
    /// This includes data readers, saving, caching, filtering etc.
    /// </summary>
    public sealed class DataOperationsCatalog
    {
        [BestFriend]
        internal IHostEnvironment Environment { get; }

        internal DataOperationsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            Environment = env;
        }

        /// <summary>
        /// Creates a lazy in-memory cache of <paramref name="input"/>.
        /// Caching happens per-column. A column is only cached when it is first accessed.
        /// In addition, <paramref name="columnsToPrefetch"/> are considered 'always needed', so all of them
        /// will be cached whenever any data is requested.
        /// </summary>
        /// <param name="input">The data view to cache.</param>
        /// <param name="columnsToPrefetch">The columns that must be cached whenever anything is cached. Empty array or null
        /// is acceptable, it means that all columns are only cached at the first access.</param>
        public IDataView Cache(IDataView input, params string[] columnsToPrefetch)
        {
            Environment.CheckValue(input, nameof(input));
            Environment.CheckValueOrNull(columnsToPrefetch);

            int[] prefetch = new int[Utils.Size(columnsToPrefetch)];
            for (int i = 0; i < prefetch.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(columnsToPrefetch[i], out prefetch[i]))
                    throw Environment.ExceptSchemaMismatch(nameof(columnsToPrefetch), "prefetch", columnsToPrefetch[i]);
            }
            return new CacheDataView(Environment, input, prefetch);
        }

        /// <summary>
        /// Filter the dataset by the values of a numeric column.
        /// </summary>
        /// <remarks>
        /// Keep only those rows that satisfy the range condition: the value of column <paramref name="columnName"/>
        /// must be between <paramref name="lowerBound"/> (inclusive) and <paramref name="upperBound"/> (exclusive).
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="columnName">The name of a column to use for filtering.</param>
        /// <param name="lowerBound">The inclusive lower bound.</param>
        /// <param name="upperBound">The exclusive upper bound.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FilterByColumn](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/FilterByColumn.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView FilterByColumn(IDataView input, string columnName, double lowerBound = double.NegativeInfinity, double upperBound = double.PositiveInfinity)
        {
            Environment.CheckValue(input, nameof(input));
            Environment.CheckNonEmpty(columnName, nameof(columnName));
            Environment.CheckParam(lowerBound < upperBound, nameof(upperBound), "Must be less than lowerBound");

            var type = input.Schema[columnName].Type;
            if (!(type is NumberType))
                throw Environment.ExceptSchemaMismatch(nameof(columnName), "filter", columnName, "number", type.ToString());
            return new RangeFilter(Environment, input, columnName, lowerBound, upperBound, false);
        }

        /// <summary>
        /// Filter the dataset by the values of a <see cref="KeyType"/> column.
        /// </summary>
        /// <remarks>
        /// Keep only those rows that satisfy the range condition: the value of a key column <paramref name="columnName"/>
        /// (treated as a fraction of the entire key range) must be between <paramref name="lowerBound"/> (inclusive) and <paramref name="upperBound"/> (exclusive).
        /// This filtering is useful if the <paramref name="columnName"/> is a key column obtained by some 'stable randomization',
        /// for example, hashing.
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="columnName">The name of a column to use for filtering.</param>
        /// <param name="lowerBound">The inclusive lower bound.</param>
        /// <param name="upperBound">The exclusive upper bound.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FilterByKeyColumnFraction](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/FilterByKeyColumnFraction.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView FilterByKeyColumnFraction(IDataView input, string columnName, double lowerBound = 0, double upperBound = 1)
        {
            Environment.CheckValue(input, nameof(input));
            Environment.CheckNonEmpty(columnName, nameof(columnName));
            Environment.CheckParam(0 <= lowerBound && lowerBound <= 1, nameof(lowerBound), "Must be in [0, 1]");
            Environment.CheckParam(0 <= upperBound && upperBound <= 2, nameof(upperBound), "Must be in [0, 2]");
            Environment.CheckParam(lowerBound <= upperBound, nameof(upperBound), "Must be no less than lowerBound");

            var type = input.Schema[columnName].Type;
            if (type.GetKeyCount() == 0)
                throw Environment.ExceptSchemaMismatch(nameof(columnName), "filter", columnName, "KeyType", type.ToString());
            return new RangeFilter(Environment, input, columnName, lowerBound, upperBound, false);
        }
    }
}
