// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(typeof(RowToRowMapperTransform), null, typeof(SignatureLoadDataTransform),
    "", RowToRowMapperTransform.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This interface is used to create a <see cref="RowToRowMapperTransform"/>.
    /// Implementations should be given an <see cref="DataViewSchema"/> in their constructor, and should have a
    /// ctor or Create method with <see cref="SignatureLoadRowMapper"/>, along with a corresponding
    /// <see cref="LoadableClassAttribute"/>.
    /// </summary>
    [BestFriend]
    internal interface IRowMapper : ICanSaveModel
    {
        /// <summary>
        /// Returns the input columns needed for the requested output columns.
        /// </summary>
        Func<int, bool> GetDependencies(Func<int, bool> activeOutput);

        /// <summary>
        /// Returns the getters for the output columns given an active set of output columns. The length of the getters
        /// array should be equal to the number of columns added by the IRowMapper. It should contain the getter for the
        /// i'th output column if activeOutput(i) is true, and null otherwise. If creating a <see cref="DataViewRow"/> or
        /// <see cref="DataViewRowCursor"/> out of this, the <paramref name="disposer"/> delegate (if non-null) should be called
        /// from the dispose of either of those instances.
        /// </summary>
        Delegate[] CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer);

        /// <summary>
        /// Returns information about the output columns, including their name, type and any metadata information.
        /// </summary>
        DataViewSchema.DetachedColumn[] GetOutputColumns();

        /// <summary>
        /// DO NOT USE IT!
        /// Purpose of this method is to enable legacy loading and unwrapping of RowToRowTransform.
        /// It should be removed as soon as we get rid of <see cref="TrainedWrapperEstimatorBase"/>
        /// Returns parent transfomer which uses this mapper.
        /// </summary>
        ITransformer GetTransformer();
    }
    [BestFriend]
    internal delegate void SignatureLoadRowMapper(ModelLoadContext ctx, DataViewSchema schema);

    /// <summary>
    /// This class is a transform that can add any number of output columns, that depend on any number of input columns.
    /// It does so with the help of an <see cref="IRowMapper"/>, that is given a schema in its constructor, and has methods
    /// to get the dependencies on input columns and the getters for the output columns, given an active set of output columns.
    /// </summary>
    [BestFriend]
    internal sealed class RowToRowMapperTransform : RowToRowTransformBase, IRowToRowMapper,
        ITransformCanSaveOnnx, ITransformCanSavePfa, ITransformTemplate
    {
        private readonly IRowMapper _mapper;
        private readonly ColumnBindings _bindings;

        // If this is not null, the transform is re-appliable without save/load.
        private readonly Func<DataViewSchema, IRowMapper> _mapperFactory;

        public const string RegistrationName = "RowToRowMapperTransform";
        public const string LoaderSignature = "RowToRowMapper";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ROW MPPR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RowToRowMapperTransform).Assembly.FullName);
        }

        public override DataViewSchema OutputSchema => _bindings.Schema;

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => _mapper is ICanSaveOnnx onnxMapper ? onnxMapper.CanSaveOnnx(ctx) : false;

        bool ICanSavePfa.CanSavePfa => _mapper is ICanSavePfa pfaMapper ? pfaMapper.CanSavePfa : false;

        public RowToRowMapperTransform(IHostEnvironment env, IDataView input, IRowMapper mapper, Func<DataViewSchema, IRowMapper> mapperFactory)
            : base(env, RegistrationName, input)
        {
            Contracts.CheckValue(mapper, nameof(mapper));
            Contracts.CheckValueOrNull(mapperFactory);
            _mapper = mapper;
            _mapperFactory = mapperFactory;
            _bindings = new ColumnBindings(input.Schema, mapper.GetOutputColumns());
        }
        public static DataViewSchema GetOutputSchema(DataViewSchema inputSchema, IRowMapper mapper)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.CheckValue(mapper, nameof(mapper));
            return new ColumnBindings(inputSchema, mapper.GetOutputColumns()).Schema;
        }

        private RowToRowMapperTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            // *** Binary format ***
            // _mapper

            ctx.LoadModel<IRowMapper, SignatureLoadRowMapper>(host, out _mapper, "Mapper", input.Schema);
            _bindings = new ColumnBindings(input.Schema, _mapper.GetOutputColumns());
        }

        public static RowToRowMapperTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model", ch => new RowToRowMapperTransform(h, ctx, input));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // _mapper

            ctx.SaveModel(_mapper, "Mapper");
        }

        /// <summary>
        /// Produces the set of active columns for the data view (as a bool[] of length bindings.ColumnCount),
        /// and the needed active input columns, given a predicate for the needed active output columns.
        /// </summary>
        private bool[] GetActive(Func<int, bool> predicate, out IEnumerable<DataViewSchema.Column> inputColumns)
        {
            int n = _bindings.Schema.Count;
            var active = Utils.BuildArray(n, predicate);
            Contracts.Assert(active.Length == n);

            var activeInput = _bindings.GetActiveInput(predicate);
            Contracts.Assert(activeInput.Length == _bindings.InputSchema.Count);

            // Get a predicate that determines which outputs are active.
            var predicateOut = GetActiveOutputColumns(active);

            // Now map those to active input columns.
            var predicateIn = _mapper.GetDependencies(predicateOut);

            // Combine the two sets of input columns.
            inputColumns = _bindings.InputSchema.Where(col => activeInput[col.Index] || predicateIn(col.Index));

            return active;
        }

        private Func<int, bool> GetActiveOutputColumns(bool[] active)
        {
            Contracts.AssertValue(active);
            Contracts.Assert(active.Length == _bindings.Schema.Count);

            return
                col =>
                {
                    Contracts.Assert(0 <= col && col < _bindings.AddedColumnIndices.Count);
                    return 0 <= col && col < _bindings.AddedColumnIndices.Count && active[_bindings.AddedColumnIndices[col]];
                };
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");
            if (_bindings.AddedColumnIndices.Any(predicate))
                return true;
            return null;
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var active = GetActive(predicate, out IEnumerable<DataViewSchema.Column> inputCols);

            return new Cursor(Host, Source.GetRowCursor(inputCols, rand), this, active);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var active = GetActive(predicate, out IEnumerable<DataViewSchema.Column> inputCols);

            var inputs = Source.GetRowCursorSet(inputCols, n, rand);
            Host.AssertNonEmpty(inputs);

            if (inputs.Length == 1 && n > 1 && _bindings.AddedColumnIndices.Any(predicate))
                inputs = DataViewUtils.CreateSplitCursors(Host, inputs[0], n);
            Host.AssertNonEmpty(inputs);

            var cursors = new DataViewRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new Cursor(Host, inputs[i], this, active);
            return cursors;
        }

        void ISaveAsOnnx.SaveAsOnnx(OnnxContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            if (_mapper is ISaveAsOnnx onnx)
            {
                Host.Check(onnx.CanSaveOnnx(ctx), "Cannot be saved as ONNX.");
                onnx.SaveAsOnnx(ctx);
            }
        }

        void ISaveAsPfa.SaveAsPfa(BoundPfaContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            if (_mapper is ISaveAsPfa pfa)
            {
                Host.Check(pfa.CanSavePfa, "Cannot be saved as PFA.");
                pfa.SaveAsPfa(ctx);
            }
        }

        /// <summary>
        /// Given a set of output columns, return the input columns that are needed to generate those output columns.
        /// </summary>
        IEnumerable<DataViewSchema.Column> IRowToRowMapper.GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns)
        {
            var predicate = RowCursorUtils.FromColumnsToPredicate(dependingColumns, OutputSchema);
            GetActive(predicate, out var inputColumns);
            return inputColumns;
        }

        public DataViewSchema InputSchema => Source.Schema;

        DataViewRow IRowToRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
        {
            Host.CheckValue(input, nameof(input));
            Host.CheckValue(activeColumns, nameof(activeColumns));
            Host.Check(input.Schema == Source.Schema, "Schema of input row must be the same as the schema the mapper is bound to");

            using (var ch = Host.Start("GetEntireRow"))
            {
                var activeArr = new bool[OutputSchema.Count];
                foreach (var column in activeColumns)
                {
                    Host.Assert(column.Index < activeArr.Length, $"The columns {activeColumns.Select(c => c.Name)} are not suitable for the OutputSchema.");
                    activeArr[column.Index] = true;
                }
                var pred = GetActiveOutputColumns(activeArr);
                var getters = _mapper.CreateGetters(input, pred, out Action disp);
                return new RowImpl(input, this, OutputSchema, getters, disp);
            }
        }

        IDataTransform ITransformTemplate.ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));

            Contracts.CheckValue(newSource, nameof(newSource));
            if (_mapperFactory != null)
            {
                var newMapper = _mapperFactory(newSource.Schema);
                return new RowToRowMapperTransform(env.Register(nameof(RowToRowMapperTransform)), newSource, newMapper, _mapperFactory);
            }
            // Revert to serialization. This was how it worked in all the cases, now it's only when we can't re-create the mapper.
            using (var stream = new MemoryStream())
            {
                using (var rep = RepositoryWriter.CreateNew(stream, env))
                {
                    ModelSaveContext.SaveModel(rep, this, "model");
                    rep.Commit();
                }

                stream.Position = 0;
                using (var rep = RepositoryReader.Open(stream, env))
                {
                    IDataTransform newData;
                    ModelLoadContext.LoadModel<IDataTransform, SignatureLoadDataTransform>(env,
                        out newData, rep, "model", newSource);
                    return newData;
                }
            }
        }

        private sealed class RowImpl : WrappingRow
        {
            private readonly Delegate[] _getters;
            private readonly RowToRowMapperTransform _parent;
            private readonly Action _disposer;

            public override DataViewSchema Schema { get; }

            public RowImpl(DataViewRow input, RowToRowMapperTransform parent, DataViewSchema schema, Delegate[] getters, Action disposer)
                : base(input)
            {
                _parent = parent;
                Schema = schema;
                _getters = getters;
                _disposer = disposer;
            }

            protected override void DisposeCore(bool disposing)
            {
                if (disposing)
                    _disposer?.Invoke();
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                bool isSrc;
                int index = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return Input.GetGetter<TValue>(Input.Schema[index]);

                Contracts.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Contracts.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                bool isSrc;
                int index = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return Input.IsColumnActive(Schema[index]);
                return _getters[index] != null;
            }
        }

        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly Delegate[] _getters;
            private readonly bool[] _active;
            private readonly ColumnBindings _bindings;
            private readonly Action _disposer;
            private bool _disposed;

            public override DataViewSchema Schema => _bindings.Schema;

            public Cursor(IChannelProvider provider, DataViewRowCursor input, RowToRowMapperTransform parent, bool[] active)
                : base(provider, input)
            {
                var pred = parent.GetActiveOutputColumns(active);
                _getters = parent._mapper.CreateGetters(input, pred, out _disposer);
                _active = active;
                _bindings = parent._bindings;
            }

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < _bindings.Schema.Count);
                return _active[column.Index];
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Ch.Check(IsColumnActive(column));

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return Input.GetGetter<TValue>(Input.Schema[index]);

                Ch.AssertValue(_getters);
                var getter = _getters[index];
                Ch.Assert(getter != null);
                var fn = getter as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                    _disposer?.Invoke();
                _disposed = true;
                base.Dispose(disposing);
            }
        }

        internal ITransformer GetTransformer()
        {
            return _mapper.GetTransformer();
        }
    }
}
