using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    internal class StatefulTimeseriesTransformer : ITransformer
    {
        private ITransformer Transformer { get; set; }
        private IHost Host { get; set; }

        //private Action<PingerArgument> _pinger;
        //private long _rowPosition;

        public StatefulTimeseriesTransformer(IHostEnvironment env, ITransformer transformer)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register("StatefulTimeseriesTransformer");
            Host.AssertValue(transformer);
            Transformer = CloneTransformers(transformer);
        }

        public bool IsRowToRowMapper => IsRowToRowMapperTransformer(Transformer);

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => Transformer.GetOutputSchema(inputSchema);

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.Check(IsRowToRowMapperTransformer(Transformer), nameof(GetRowToRowMapper) +
                " method called despite " + nameof(IsRowToRowMapper) + " being false. or transformer not being " + nameof(IStatefulTransformer));

            if (!(Transformer is ITransformerChainAccessor))
                if (Transformer is IStatefulTransformer)
                    return ((IStatefulTransformer)Transformer).GetStatefulRowToRowMapper(inputSchema);
                else
                    return Transformer.GetRowToRowMapper(inputSchema);

            Contracts.Check(Transformer is ITransformerChainAccessor);

            var transformers = ((ITransformerChainAccessor)Transformer).Transformers;
            IRowToRowMapper[] mappers = new IRowToRowMapper[transformers.Length];
            DataViewSchema schema = inputSchema;
            for (int i = 0; i < mappers.Length; ++i)
            {
                if (transformers[i] is IStatefulTransformer)
                    mappers[i] = ((IStatefulTransformer)transformers[i]).GetStatefulRowToRowMapper(schema);
                else
                    mappers[i] = transformers[i].GetRowToRowMapper(schema);

                schema = mappers[i].OutputSchema;
            }
            return new CompositeRowToRowMapper(inputSchema, mappers);
        }

        public void Save(ModelSaveContext ctx) => Transformer.Save(ctx);

        public IDataView Transform(IDataView input)
        {
            var transformedView = Transformer.Transform(input);

            List<StatefulRow> rows = new List<StatefulRow>();
            var mapper = GetRowToRowMapper(transformedView.Schema);
            var inputRow = mapper.GetRow(input.GetRowCursorForAllColumns(), input.Schema.ToArray());
            GetStatefulRows(inputRow, mapper, mapper.OutputSchema, rows);
            return new DataView(this, transformedView, CreatePinger(rows));
        }

        private bool IsRowToRowMapperTransformer(ITransformer transformer)
        {
            if (transformer is ITransformerChainAccessor)
                return ((ITransformerChainAccessor)transformer).Transformers.All(t => t.IsRowToRowMapper || t is IStatefulTransformer);
            else
                return transformer.IsRowToRowMapper || transformer is IStatefulTransformer;
        }

        private Action<PingerArgument> CreatePinger(List<StatefulRow> rows)
        {
            if (rows.Count == 0)
                return position => { };
            Action<PingerArgument> pinger = null;
            foreach (var row in rows)
                pinger += row.GetPinger();
            return pinger;
        }

        private DataViewRow GetStatefulRows(DataViewRow input, IRowToRowMapper mapper, IEnumerable<DataViewSchema.Column> activeColumns, List<StatefulRow> rows)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(activeColumns, nameof(activeColumns));

            IRowToRowMapper[] innerMappers = new IRowToRowMapper[0];
            if (mapper is CompositeRowToRowMapper compositeMapper)
                innerMappers = compositeMapper.InnerMappers;

            var activeIndices = new HashSet<int>(activeColumns.Select(c => c.Index));
            if (innerMappers.Length == 0)
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

                var row = mapper.GetRow(input, activeColumns);
                if (row is StatefulRow statefulRow)
                    rows.Add(statefulRow);
                return row;
            }

            // For each of the inner mappers, we will be calling their GetRow method, but to do so we need to know
            // what we need from them. The last one will just have the input, but the rest will need to be
            // computed based on the dependencies of the next one in the chain.
            var deps = new IEnumerable<DataViewSchema.Column>[innerMappers.Length];
            deps[deps.Length - 1] = activeColumns;
            for (int i = deps.Length - 1; i >= 1; --i)
                deps[i - 1] = innerMappers[i].GetDependencies(deps[i]);

            DataViewRow result = input;
            for (int i = 0; i < innerMappers.Length; ++i)
            {
                result = GetStatefulRows(result, innerMappers[i], deps[i], rows);
                if (result is StatefulRow statefulResult)
                    rows.Add(statefulResult);
            }
            return result;
        }

        private static ITransformer CloneTransformers(ITransformer transformer)
        {
            ITransformer[] transformersClone = null;
            TransformerScope[] scopeClone = null;
            if (transformer is ITransformerChainAccessor)
            {
                ITransformerChainAccessor accessor = (ITransformerChainAccessor)transformer;
                transformersClone = accessor.Transformers.Select(x => x).ToArray();
                scopeClone = accessor.Scopes.Select(x => x).ToArray();
                int index = 0;
                foreach (var xf in transformersClone)
                    transformersClone[index++] = xf is IStatefulTransformer ? ((IStatefulTransformer)xf).Clone() : xf;

                return new TransformerChain<ITransformer>(transformersClone, scopeClone);
            }
            else
                return transformer is IStatefulTransformer ? ((IStatefulTransformer)transformer).Clone() : transformer;
        }

        private sealed class DataView : IDataView
        {
            private readonly StatefulTimeseriesTransformer _parent;
            private readonly IDataView _input;
            private readonly Action<PingerArgument> _pinger;

            public DataView(StatefulTimeseriesTransformer parent, IDataView input, Action<PingerArgument> pinger)
            {
                _parent = parent;
                _input = input;
                _pinger = pinger;
            }

            public bool CanShuffle => false;

            public DataViewSchema Schema => _input.Schema;

            public long? GetRowCount() => _input.GetRowCount();

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                return new Cursor(_parent, _input.GetRowCursor(columnsNeeded), columnsNeeded, _pinger);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                Contracts.CheckParam(n >= 0, nameof(n));
                Contracts.CheckValueOrNull(rand);
                return new[] { GetRowCursor(columnsNeeded, rand) };
            }
        }

        private sealed class Cursor : RootCursorBase
        {
            private readonly StatefulTimeseriesTransformer _parent;
            private readonly DataViewRowCursor _input;
            private readonly Action<PingerArgument> _pinger;

            private bool _disposed;

            public override long Batch => _input.Batch;

            public Cursor(StatefulTimeseriesTransformer parent, DataViewRowCursor input, IEnumerable<DataViewSchema.Column> columnsNeeded, Action<PingerArgument> pinger)
                : base(parent.Host)
            {
                Ch.AssertValue(input);
                Ch.AssertValue(columnsNeeded);

                _parent = parent;
                _input = input;
                _pinger = pinger;
            }

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    _input.Dispose();
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return _input.GetIdGetter();
            }

            protected override bool MoveNextCore()
            {
                // Update state.
                _pinger(new PingerArgument()
                {
                    RowPosition = Position + 1,
                });

                if (!_input.MoveNext())
                    return false;

                return true;
            }

            public override DataViewSchema Schema => _input.Schema;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column) => _input.IsColumnActive(column);

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column) => _input.GetGetter<TValue>(column);
        }

    }
}
