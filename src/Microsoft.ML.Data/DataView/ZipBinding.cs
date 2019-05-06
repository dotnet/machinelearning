// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A convenience class for concatenating several schemas together.
    /// This would be necessary when combining IDataViews through any type of combining operation, for example, zip.
    /// </summary>
    [BestFriend]
    internal sealed class ZipBinding
    {
        private readonly DataViewSchema[] _sources;

        public DataViewSchema OutputSchema { get; }

        // Zero followed by cumulative column counts. Zero being used for the empty case.
        private readonly int[] _cumulativeColCounts;

        public ZipBinding(DataViewSchema[] sources)
        {
            Contracts.AssertNonEmpty(sources);
            _sources = sources;
            _cumulativeColCounts = new int[_sources.Length + 1];
            _cumulativeColCounts[0] = 0;

            for (int i = 0; i < sources.Length; i++)
            {
                var schema = sources[i];
                _cumulativeColCounts[i + 1] = _cumulativeColCounts[i] + schema.Count;
            }

            var schemaBuilder = new DataViewSchema.Builder();
            foreach (var sourceSchema in sources)
                schemaBuilder.AddColumns(sourceSchema);
            OutputSchema = schemaBuilder.ToSchema();
        }

        public int ColumnCount => _cumulativeColCounts[_cumulativeColCounts.Length - 1];

        /// <summary>
        /// Returns an array of input predicated for sources, corresponding to the input predicate.
        /// The returned array size is equal to the number of sources, but if a given source is not needed at all,
        /// the corresponding predicate will be null.
        /// </summary>
        public Func<int, bool>[] GetInputPredicates(Func<int, bool> predicate)
        {
            Contracts.AssertValue(predicate);
            var result = new Func<int, bool>[_sources.Length];
            for (int i = 0; i < _sources.Length; i++)
            {
                var lastColCount = _cumulativeColCounts[i];
                result[i] = srcCol => predicate(srcCol + lastColCount);
            }

            return result;
        }

        /// <summary>
        /// Checks whether the column index is in range.
        /// </summary>
        public void CheckColumnInRange(int col)
        {
            Contracts.CheckParam(0 <= col && col < _cumulativeColCounts[_cumulativeColCounts.Length - 1], nameof(col), "Column index out of range");
        }

        public void GetColumnSource(int col, out int srcIndex, out int srcCol)
        {
            CheckColumnInRange(col);
            if (!Utils.TryFindIndexSorted(_cumulativeColCounts, 0, _cumulativeColCounts.Length, col, out srcIndex))
                srcIndex--;
            Contracts.Assert(0 <= srcIndex && srcIndex < _cumulativeColCounts.Length);
            srcCol = col - _cumulativeColCounts[srcIndex];
            Contracts.Assert(0 <= srcCol && srcCol < _sources[srcIndex].Count);
        }
    }
}
