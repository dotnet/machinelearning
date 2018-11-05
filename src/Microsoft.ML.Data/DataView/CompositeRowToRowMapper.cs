// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A row-to-row mapper that is the result of a chained application of multiple mappers.
    /// </summary>
    public sealed class CompositeRowToRowMapper : IRowToRowMapper
    {
        private readonly IRowToRowMapper[] _innerMappers;
        private static readonly IRowToRowMapper[] _empty = new IRowToRowMapper[0];

        public Schema InputSchema { get; }
        public Schema Schema { get; }

        /// <summary>
        /// Out of a series of mappers, construct a seemingly unitary mapper that is able to apply them in sequence.
        /// </summary>
        /// <param name="inputSchema">The input schema.</param>
        /// <param name="mappers">The sequence of mappers to wrap. An empty or <c>null</c> argument
        /// is legal, and counts as being a no-op application.</param>
        public CompositeRowToRowMapper(Schema inputSchema, IRowToRowMapper[] mappers)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.CheckValueOrNull(mappers);
            _innerMappers = Utils.Size(mappers) > 0 ? mappers : _empty;
            InputSchema = inputSchema;
            Schema = Utils.Size(mappers) > 0 ? mappers[mappers.Length - 1].Schema : inputSchema;
        }

        public Func<int, bool> GetDependencies(Func<int, bool> predicate)
        {
            Func<int, bool> toReturn = predicate;
            for (int i = _innerMappers.Length - 1; i >= 0; --i)
                toReturn = _innerMappers[i].GetDependencies(toReturn);
            return toReturn;
        }

        public IRow GetRow(IRow input, Func<int, bool> active, out Action disposer)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(active, nameof(active));
            Contracts.CheckParam(input.Schema == InputSchema, nameof(input), "Schema did not match original schema");

            disposer = null;
            if (_innerMappers.Length == 0)
            {
                bool differentActive = false;
                for (int c = 0; c < input.Schema.ColumnCount; ++c)
                {
                    bool wantsActive = active(c);
                    bool isActive = input.IsColumnActive(c);
                    differentActive |= wantsActive != isActive;

                    if (wantsActive && !isActive)
                        throw Contracts.ExceptParam(nameof(input), $"Mapper required column '{input.Schema.GetColumnName(c)}' active but it was not.");
                }
                return input;
            }

            // For each of the inner mappers, we will be calling their GetRow method, but to do so we need to know
            // what we need from them. The last one will just have the input, but the rest will need to be
            // computed based on the dependencies of the next one in the chain.
            var deps = new Func<int, bool>[_innerMappers.Length];
            deps[deps.Length - 1] = active;
            for (int i = deps.Length - 1; i >= 1; --i)
                deps[i - 1] = _innerMappers[i].GetDependencies(deps[i]);

            IRow result = input;
            for (int i = 0; i < _innerMappers.Length; ++i)
            {
                result = _innerMappers[i].GetRow(result, deps[i], out var localDisp);
                if (localDisp != null)
                {
                    if (disposer == null)
                        disposer = localDisp;
                    else
                        disposer = localDisp + disposer;
                    // We want the last disposer to be called first, so the order of the addition here is important.
                }
            }

            return result;
        }

        private sealed class SubsetActive : IRow
        {
            private readonly IRow _row;
            private Func<int, bool> _pred;

            public SubsetActive(IRow row, Func<int, bool> pred)
            {
                Contracts.AssertValue(row);
                Contracts.AssertValue(pred);
                _row = row;
                _pred = pred;
            }

            public Schema Schema => _row.Schema;
            public long Position => _row.Position;
            public long Batch => _row.Batch;
            public ValueGetter<TValue> GetGetter<TValue>(int col) => _row.GetGetter<TValue>(col);
            public ValueGetter<UInt128> GetIdGetter() => _row.GetIdGetter();
            public bool IsColumnActive(int col) => _pred(col);
        }
    }
}
