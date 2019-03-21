// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A row-to-row mapper that is the result of a chained application of multiple mappers.
    /// </summary>
    [BestFriend]
    internal sealed class CompositeRowToRowMapper : IRowToRowMapper
    {
        [BestFriend]
        internal IRowToRowMapper[] InnerMappers { get; }
        private static readonly IRowToRowMapper[] _empty = new IRowToRowMapper[0];

        public DataViewSchema InputSchema { get; }
        public DataViewSchema OutputSchema { get; }

        /// <summary>
        /// Out of a series of mappers, construct a seemingly unitary mapper that is able to apply them in sequence.
        /// </summary>
        /// <param name="inputSchema">The input schema.</param>
        /// <param name="mappers">The sequence of mappers to wrap. An empty or <c>null</c> argument
        /// is legal, and counts as being a no-op application.</param>
        public CompositeRowToRowMapper(DataViewSchema inputSchema, IRowToRowMapper[] mappers)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.CheckValueOrNull(mappers);
            InnerMappers = Utils.Size(mappers) > 0 ? mappers : _empty;
            InputSchema = inputSchema;
            OutputSchema = Utils.Size(mappers) > 0 ? mappers[mappers.Length - 1].OutputSchema : inputSchema;
        }

        /// <summary>
        /// Given a set of columns, return the input columns that are needed to generate those output columns.
        /// </summary>
        IEnumerable<DataViewSchema.Column> IRowToRowMapper.GetDependencies(IEnumerable<DataViewSchema.Column> columnsNeeded)
        {
            for (int i = InnerMappers.Length - 1; i >= 0; --i)
                columnsNeeded = InnerMappers[i].GetDependencies(columnsNeeded);

            return columnsNeeded;
        }

        DataViewRow IRowToRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(activeColumns, nameof(activeColumns));
            Contracts.CheckParam(input.Schema == InputSchema, nameof(input), "Schema did not match original schema");

            var activeIndices = activeColumns.Select(c => c.Index).ToArray();
            if (InnerMappers.Length == 0)
            {
                bool differentActive = false;
                for (int c = 0; c < input.Schema.Count; ++c)
                {
                    bool wantsActive = activeIndices.Contains(c);
                    bool isActive = input.IsColumnActive(input.Schema[c]);
                    differentActive |= wantsActive != isActive;

                    if (wantsActive && !isActive)
                        throw Contracts.ExceptParam(nameof(input), $"Mapper required column '{input.Schema[c].Name}' active but it was not.");
                }
                return input;
            }

            // For each of the inner mappers, we will be calling their GetRow method, but to do so we need to know
            // what we need from them. The last one will just have the input, but the rest will need to be
            // computed based on the dependencies of the next one in the chain.
            IEnumerable<DataViewSchema.Column>[] deps = new IEnumerable<DataViewSchema.Column>[InnerMappers.Length];
            deps[deps.Length - 1] = OutputSchema.Where(c => activeIndices.Contains(c.Index));
            for (int i = deps.Length - 1; i >= 1; --i)
                deps[i - 1] = InnerMappers[i].GetDependencies(deps[i]);

            DataViewRow result = input;
            for (int i = 0; i < InnerMappers.Length; ++i)
                result = InnerMappers[i].GetRow(result, deps[i]);

            return result;
        }

        private sealed class SubsetActive : DataViewRow
        {
            private readonly DataViewRow _row;
            private Func<int, bool> _pred;

            public SubsetActive(DataViewRow row, Func<int, bool> pred)
            {
                Contracts.AssertValue(row);
                Contracts.AssertValue(pred);
                _row = row;
                _pred = pred;
            }

            public override DataViewSchema Schema => _row.Schema;
            public override long Position => _row.Position;
            public override long Batch => _row.Batch;

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column) => _row.GetGetter<TValue>(column);
            public override ValueGetter<DataViewRowId> GetIdGetter() => _row.GetIdGetter();

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column) => _pred(column.Index);
        }
    }
}
