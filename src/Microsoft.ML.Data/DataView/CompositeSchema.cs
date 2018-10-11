// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A convenience class for concatenating several schemas together.
    /// This would be necessary when combining IDataViews through any type of combining operation, for example, zip.
    /// </summary>
    internal sealed class CompositeSchema : ISchema
    {
        private readonly ISchema[] _sources;

        // Zero followed by cumulative column counts. Zero being used for the empty case.
        private readonly int[] _cumulativeColCounts;

        public CompositeSchema(ISchema[] sources)
        {
            Contracts.AssertNonEmpty(sources);
            _sources = sources;
            _cumulativeColCounts = new int[_sources.Length + 1];
            _cumulativeColCounts[0] = 0;

            for (int i = 0; i < sources.Length; i++)
            {
                var schema = sources[i];
                _cumulativeColCounts[i + 1] = _cumulativeColCounts[i] + schema.ColumnCount;
            }
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
            Contracts.Assert(0 <= srcCol && srcCol < _sources[srcIndex].ColumnCount);
        }

        public bool TryGetColumnIndex(string name, out int col)
        {
            for (int i = _sources.Length; --i >= 0;)
            {
                if (_sources[i].TryGetColumnIndex(name, out col))
                {
                    col += _cumulativeColCounts[i];
                    return true;
                }
            }

            col = -1;
            return false;
        }

        public string GetColumnName(int col)
        {
            GetColumnSource(col, out int dv, out int srcCol);
            return _sources[dv].GetColumnName(srcCol);
        }

        public ColumnType GetColumnType(int col)
        {
            GetColumnSource(col, out int dv, out int srcCol);
            return _sources[dv].GetColumnType(srcCol);
        }

        public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
        {
            GetColumnSource(col, out int dv, out int srcCol);
            return _sources[dv].GetMetadataTypes(srcCol);
        }

        public ColumnType GetMetadataTypeOrNull(string kind, int col)
        {
            GetColumnSource(col, out int dv, out int srcCol);
            return _sources[dv].GetMetadataTypeOrNull(kind, srcCol);
        }

        public void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            GetColumnSource(col, out int dv, out int srcCol);
            _sources[dv].GetMetadata(kind, srcCol, ref value);
        }
    }
}
