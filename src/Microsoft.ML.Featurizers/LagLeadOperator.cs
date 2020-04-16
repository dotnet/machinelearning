// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;
using static Microsoft.ML.SchemaShape.Column;

[assembly: LoadableClass(typeof(LagLeadOperatorTransformer), null, typeof(SignatureLoadModel),
    LagLeadOperatorTransformer.UserName, LagLeadOperatorTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(LagLeadOperatorTransformer), null, typeof(SignatureLoadDataTransform),
   LagLeadOperatorTransformer.UserName, LagLeadOperatorTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(LagLeadOperatorEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class LagLeadOperatorExtensionClass
    {
        /// <summary>
        /// Creates a <see cref="LagLeadOperatorEstimator"/>. This copies values from prior or future rows based on grain.
        /// The Horizon represents the maximum value in a range [1, N], where each element in that range is a delta applied to each offset.
        /// The resulting vector output dimensions are K rows x N cols, where K is the number of offsets and N is the horizon.
        /// </summary>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="grainColumns">List of columns to use as grains.</param>
        /// <param name="outputColumn">The output column.</param>
        /// <param name="horizon">Maximum horizon.</param>
        /// <param name="offsets">List of additional offsets. Negative values are prior rows, positive values are future rows.</param>
        /// <param name="inputColumn">If input column is different from output.</param>
        /// <returns></returns>
        public static LagLeadOperatorEstimator CreateLagsAndLeads(this TransformsCatalog catalog, string[] grainColumns, string outputColumn, UInt32 horizon, long[] offsets, string inputColumn = null)
        {
            var options = new LagLeadOperatorEstimator.Options
            {
                GrainColumns = grainColumns,
                Column = new[] { new LagLeadOperatorEstimator.Column() { Name = outputColumn, Source = inputColumn ?? outputColumn } },
                Horizon = horizon,
                Offsets = offsets
            };

            return new LagLeadOperatorEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }

        /// <summary>
        /// Creates a <see cref="LagLeadOperatorEstimator"/>. This copies values from prior or future rows based on grain.
        /// The Horizon represents the maximum value in a range [1, N], where each element in that range is a delta applied to each offset.
        /// The resulting vector output dimensions are K rows x N cols, where K is the number of offsets and N is the horizon.
        /// </summary>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="grainColumns">List of columns to use as grains.</param>
        /// <param name="columns">List of input/output column pairs.</param>
        /// <param name="horizon">Maximum horizon.</param>
        /// <param name="offsets">List of additional offsets. Negative values are prior rows, positive values are future rows.</param>
        /// <returns></returns>
        public static LagLeadOperatorEstimator CreateLagsAndLeads(this TransformsCatalog catalog, string[] grainColumns, InputOutputColumnPair[] columns, UInt32 horizon, long[] offsets)
        {
            var options = new LagLeadOperatorEstimator.Options
            {
                GrainColumns = grainColumns,
                Column = columns.Select(x => new LagLeadOperatorEstimator.Column { Name = x.OutputColumnName, Source = x.InputColumnName ?? x.OutputColumnName }).ToArray(),
                Horizon = horizon,
                Offsets = offsets
            };

            return new LagLeadOperatorEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    /// <summary>
    /// Creates a <see cref="LagLeadOperatorEstimator"/>. This copies values from prior or future rows based on grain.
    /// The Horizon represents the maximum value in a range [1, N], where each element in that range is a delta applied to each offset.
    /// The resulting vector output dimensions are K rows x N cols, where K is the number of offsets and N is the horizon.
    ///
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Double |
    /// | Output column data type | 2d Vector of double |
    /// | Exportable to ONNX | No |
    ///
    /// Creates a <see cref="LagLeadOperatorEstimator"/>. This copies values from prior or future rows based on grain.
    /// The Horizon represents the maximum value in a range [1, N], where each element in that range is a delta applied to each offset.
    /// The resulting vector output dimensions are K rows x N cols, where K is the number of offsets and N is the horizon.
    ///
    /// A simple example would be horizon = 1 and we have offsets as [-3, 1] (which means lag 3 and lead 1). If our input column is "target" and our output is "output":
    ///      +-------+-------+---------------------+
    ///      | grain | target| output              |
    ///      +=======+=======+=====================+
    ///      |Walmart| 8     | [[NAN], [  9]]      |
    ///      +-------+-------+---------------------+
    ///      |Walmart| 9     | [[NAN], [ 10]]      |
    ///      +-------+-------+---------------------+
    ///      |Walmart| 10    | [[NAN], [ 11]]      |
    ///      +-------+-------+---------------------+
    ///      |Walmart| 11    | [[  8], [NAN]]      |
    ///      +-------+-------+---------------------+
    ///
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="LagLeadOperatorExtensionClass.CreateLagsAndLeads(TransformsCatalog, string[], string, UInt32, long[], string)"/>
    /// <seealso cref="LagLeadOperatorExtensionClass.CreateLagsAndLeads(TransformsCatalog, string[], InputOutputColumnPair[], UInt32, long[])"/>
    public class LagLeadOperatorEstimator : IEstimator<LagLeadOperatorTransformer>
    {
        private Options _options;
        private readonly IHost _host;

        /* Codegen: Add additional needed class members here */

        #region Options

        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument((ArgumentType.MultipleUnique | ArgumentType.Required), HelpText = "List of grain columns",
                Name = "GrainColumns", ShortName = "grains", SortOrder = 0)]
            public string[] GrainColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Maximum horizon value",
                Name = "Horizon", ShortName = "hor", SortOrder = 2)]
            public UInt32 Horizon;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Lag and Lead offset to use. A negative number is a lag, positive is a lead",
                Name = "Offsets", ShortName = "off", SortOrder = 3)]
            public long[] Offsets;
        }

        #endregion

        internal LagLeadOperatorEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(LagLeadOperatorEstimator));
            _host.CheckValue(options.GrainColumns, nameof(options.GrainColumns), "Grain columns should not be null.");
            _host.CheckNonEmpty(options.GrainColumns, nameof(options.GrainColumns), "Need at least one grain column.");
            _host.CheckValue(options.Column, nameof(options.Column), "Columns should not be null.");
            _host.CheckNonEmpty(options.Column, nameof(options.Column), "Need at least one column pair.");
            _host.CheckValue(options.Offsets, nameof(options.Offsets), "Offsets should not be null.");
            _host.CheckNonEmpty(options.Offsets, nameof(options.Offsets), "Need at least one offset.");
            _host.Check(options.Horizon > 0, "Can't have a horizon of 0.");
            _host.Check(options.Horizon <= int.MaxValue, "Horizon must be less then or equal to int.max");

            _options = options;
        }

        public LagLeadOperatorTransformer Fit(IDataView input)
        {
            return new LagLeadOperatorTransformer(_host, input, _options);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            if (!AllGrainColumnsAreStrings(inputSchema, _options.GrainColumns))
                throw new InvalidOperationException("Grain columns can only be of type string");

            var columns = inputSchema.ToDictionary(x => x.Name);

            foreach (var column in _options.Column)
            {

                var inputColumn = columns[column.Source];

                if (!LagLeadOperatorTransformer.TypedColumn.IsColumnTypeSupported(inputColumn.ItemType.RawType))
                    throw new InvalidOperationException($"Type {inputColumn.ItemType.RawType} for column {column.Source} not a supported type.");

                // Create annotations
                var annotations = new DataViewSchema.Annotations.Builder();
                ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "LagLead".AsMemory();
                ValueGetter<VBuffer<long>> offsetsValueGetter = (ref VBuffer<long> dst) => dst = new VBuffer<long>(_options.Offsets.Length, _options.Offsets);

                annotations.Add<ReadOnlyMemory<char>>($"FeaturizerName=LagLead", TextDataViewType.Instance, nameValueGetter);
                annotations.Add<VBuffer<long>>($"Offsets={String.Join(",", _options.Offsets)}", new VectorDataViewType(NumberDataViewType.Int64, _options.Offsets.Length), offsetsValueGetter);

                columns[column.Name] = new SchemaShape.Column(column.Name, VectorKind.Vector,
                    NumberDataViewType.Double, false, SchemaShape.Create(annotations.ToAnnotations().Schema));
            }

            return new SchemaShape(columns.Values);
        }
    }

    public sealed class LagLeadOperatorTransformer : ITransformer, IDisposable
    {
        #region Class data members

        internal const string Summary = "Uses the offset list with the horizon to create lags and leads";
        internal const string UserName = "LagLeadOperator";
        internal const string ShortName = "LagLead";
        internal const string LoadName = "LagLeadTransformer";
        internal const string LoaderSignature = "LagLeadTransformer";

        private TypedColumn[] _columns;
        private readonly IHost _host;
        private LagLeadOperatorEstimator.Options _options;

        #endregion

        // Normal constructor.
        internal LagLeadOperatorTransformer(IHostEnvironment host, IDataView input, LagLeadOperatorEstimator.Options options)
        {
            _host = host.Register(nameof(LagLeadOperatorTransformer));
            var schema = input.Schema;
            _options = options;

            _columns = options.Column.Select(x => TypedColumn.CreateTypedColumn(x.Name, x.Source, schema[x.Source].Type.RawType.ToString(), this)).ToArray();
            foreach (var column in _columns)
            {
                column.CreateTransformerFromEstimator(input);
            }
        }

        // Factory method for SignatureLoadModel.
        internal LagLeadOperatorTransformer(IHostEnvironment host, ModelLoadContext ctx)
        {
            _host = host.Register(nameof(LagLeadOperatorTransformer));
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // length of grain column array
            // all column names in grain column array
            // length of offset array
            // all values in array
            // horizon
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            var grainColumns = new string[ctx.Reader.ReadInt32()];
            for (int i = 0; i < grainColumns.Length; i++)
            {
                grainColumns[i] = ctx.Reader.ReadString();
            }

            var offsets = new long[ctx.Reader.ReadInt32()];
            for (int i = 0; i < offsets.Length; i++)
            {
                offsets[i] = ctx.Reader.ReadInt64();
            }

            var horizon = ctx.Reader.ReadUInt32();

            var columnCount = ctx.Reader.ReadInt32();
            _columns = new TypedColumn[columnCount];

            _options = new LagLeadOperatorEstimator.Options()
            {
                GrainColumns = grainColumns,
                Column = new LagLeadOperatorEstimator.Column[columnCount],
                Horizon = horizon,
                Offsets = offsets
            };

            for (int i = 0; i < columnCount; i++)
            {
                var colName = ctx.Reader.ReadString();
                var sourceName = ctx.Reader.ReadString();
                _options.Column[i] = new LagLeadOperatorEstimator.Column()
                {
                    Name = colName,
                    Source = sourceName
                };

                _columns[i] = TypedColumn.CreateTypedColumn(colName, sourceName, ctx.Reader.ReadString(), this);

                // Load the C++ state and create the C++ transformer.
                var dataLength = ctx.Reader.ReadInt32();
                var data = ctx.Reader.ReadByteArray(dataLength);
                _columns[i].CreateTransformerFromSavedData(data);
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            return (IDataTransform)(new LagLeadOperatorTransformer(env, ctx).Transform(input));
        }

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            if (!AllGrainColumnsAreStrings(inputSchema, _options.GrainColumns))
                throw new InvalidOperationException("Grain columns can only be of type string");

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumns(inputSchema.AsEnumerable());

            foreach (var column in _options.Column)
            {
                var inputColumn = inputSchema.GetColumnOrNull(column.Source).Value;

                if (!TypedColumn.IsColumnTypeSupported(inputColumn.Type.RawType))
                    throw new InvalidOperationException($"Type {inputColumn.Type.RawType} for column {column.Source} not a supported type.");

                // Create annotations
                var annotations = new DataViewSchema.Annotations.Builder();
                ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "LagLead".AsMemory();
                ValueGetter<VBuffer<long>> offsetsValueGetter = (ref VBuffer<long> dst) => dst = new VBuffer<long>(_options.Offsets.Length, _options.Offsets);

                annotations.Add<ReadOnlyMemory<char>>($"FeaturizerName=LagLead", TextDataViewType.Instance, nameValueGetter);
                annotations.Add<VBuffer<long>>($"Offsets={String.Join(",", _options.Offsets)}", new VectorDataViewType(NumberDataViewType.Int64, _options.Offsets.Length), offsetsValueGetter);

                schemaBuilder.AddColumn(column.Name, new VectorDataViewType(NumberDataViewType.Double, _options.Offsets.Length, (int)_options.Horizon), annotations.ToAnnotations());
            }

            return schemaBuilder.ToSchema();
        }

        public bool IsRowToRowMapper => false;

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => throw new InvalidOperationException("Not a RowToRowMapper.");

        private static VersionInfo GetVersionInfo()
        {
            /* Codegen: Change these as needed */
            return new VersionInfo(
                modelSignature: "LAGLED T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LagLeadOperatorTransformer).Assembly.FullName);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // length of grain column array
            // all column names in grain column array
            // length of offset array
            // all values in array
            // horizon
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            ctx.Writer.Write(_options.GrainColumns.Length);
            foreach (var column in _options.GrainColumns)
                ctx.Writer.Write(column);

            ctx.Writer.Write(_options.Offsets.Length);
            foreach (var offset in _options.Offsets)
                ctx.Writer.Write(offset);

            ctx.Writer.Write(_options.Horizon);

            // Save interop data.
            ctx.Writer.Write(_columns.Count());
            foreach (var column in _columns)
            {
                ctx.Writer.Write(column.Name);
                ctx.Writer.Write(column.Source);
                ctx.Writer.Write(column.Type);

                // Save C++ state
                var data = column.CreateTransformerSaveData();
                ctx.Writer.Write(data.Length);
                ctx.Writer.Write(data);
            }
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        internal LagLeadOperatorDataView MakeDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            return new LagLeadOperatorDataView(_host, input, _options, this);
        }

        internal TransformerEstimatorSafeHandle[] CloneTransformers()
        {
            var transformers = new TransformerEstimatorSafeHandle[_columns.Length];
            for (int i = 0; i < _columns.Length; i++)
            {
                transformers[i] = _columns[i].CloneTransformer();
            }
            return transformers;
        }

        public void Dispose()
        {
            foreach (var column in _columns)
            {
                column.Dispose();
            }
        }

        #region IDataView

        internal sealed class LagLeadOperatorDataView : IDataTransform
        {
            private LagLeadOperatorTransformer _parent;
            private readonly IDataView _source;
            private readonly IHostEnvironment _host;
            private readonly DataViewSchema _schema;
            private readonly LagLeadOperatorEstimator.Options _options;

            internal LagLeadOperatorDataView(IHostEnvironment env, IDataView input, LagLeadOperatorEstimator.Options options, LagLeadOperatorTransformer parent)
            {
                _host = env;
                _source = input;

                _options = options;
                _parent = parent;

                _schema = _parent.GetOutputSchema(input.Schema);
            }

            public bool CanShuffle => false;

            public DataViewSchema Schema => _schema;

            public IDataView Source => _source;

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                _host.AssertValueOrNull(rand);

                return new Cursor(_host, _source, _parent.CloneTransformers(), _options, _schema);
            }

            // Can't use parallel cursors so this defaults to calling non-parallel version
            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) =>
                 new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };

            // We aren't changing the row count, so just get the _source row count
            public long? GetRowCount() => _source.GetRowCount();

            public void Save(ModelSaveContext ctx)
            {
                _parent.Save(ctx);
            }

            #region Cursor
            private sealed class Cursor : DataViewRowCursor
            {
                private readonly IChannelProvider _ch;
                private IDataView _dataView;

                // Have 2 row cursors since one will be offset.
                private DataViewRowCursor _sourceCursor;
                private DataViewRowCursor _offsetCursor;
                private long _position;
                private bool _sourceIsGood;
                private bool _offsetIsGood;
                private readonly DataViewSchema _schema;
                private TypedColumn[] _columns;
                private readonly LagLeadOperatorEstimator.Options _options;
                private ValueGetter<ReadOnlyMemory<char>>[] _grainGetters;
                private List<string> _grainOrder;
                private bool _hasFlushed;

                // These are class variables so they are only allocated once.
                private GCHandle[] _grainHandles;
                private IntPtr[] _grainArray;
                private GCHandle _grainArrayHandle;

                public Cursor(IChannelProvider provider, IDataView input, TransformerEstimatorSafeHandle[] transformers, LagLeadOperatorEstimator.Options options, DataViewSchema schema)
                {
                    _ch = provider;
                    _ch.CheckValue(input, nameof(input));

                    _dataView = input;
                    _position = -1;
                    _schema = schema;
                    _options = options;
                    _grainGetters = new ValueGetter<ReadOnlyMemory<char>>[_options.GrainColumns.Length];

                    _grainHandles = new GCHandle[_options.GrainColumns.Length];
                    _grainArray = new IntPtr[_options.GrainColumns.Length];
                    _grainOrder = new List<string>();

                    _sourceIsGood = true;
                    _offsetIsGood = true;
                    _hasFlushed = false;

                    _sourceCursor = _dataView.GetRowCursorForAllColumns();
                    _offsetCursor = _dataView.GetRowCursorForAllColumns();

                    InitializeGrainGetters(_options.GrainColumns, ref _grainGetters);

                    _columns = new TypedColumn[transformers.Length];
                    for (int i = 0; i < transformers.Length; i++)
                    {
                        _columns[i] = TypedColumn.CreateTypedColumn(options.Column[i].Name, options.Column[i].Source, input.Schema[options.Column[i].Source].Type.RawType.ToString(), transformers[i], this);
                    }
                }

                public sealed override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    return
                           (ref DataViewRowId val) =>
                           {
                               _ch.Check(_sourceIsGood, RowCursorUtils.FetchValueStateError);
                               val = new DataViewRowId((ulong)Position, 0);
                           };
                }

                public sealed override DataViewSchema Schema => _schema;

                /// <summary>
                /// Since rows will be dropped
                /// </summary>
                public override bool IsColumnActive(DataViewSchema.Column column) => true;

                protected override void Dispose(bool disposing)
                {
                    foreach (var column in _columns)
                    {
                        column.Dispose();
                    }
                    _sourceCursor.Dispose();
                    _offsetCursor.Dispose();
                }

                /// <summary>
                /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
                /// This throws if the column is not active in this row, or if the type.
                /// Since all we are doing is dropping rows, we can just use the source getter.
                /// <typeparamref name="TValue"/> differs from this column's type.
                /// </summary>
                /// <typeparam name="TValue"> is the column's content type.</typeparam>
                /// <param name="column"> is the output column whose getter should be returned.</param>
                public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
                {
                    _ch.Check(IsColumnActive(column));

                    if (_columns.Any(x => x.Name == column.Name && x.Type == column.Type.RawType.ToString()))
                        return (ValueGetter<TValue>)_columns.Where(x => x.Name == column.Name).First().GetGetter();

                    return _sourceCursor.GetGetter<TValue>(column);
                }

                public override bool MoveNext()
                {
                    bool colMoveNext;

                    // Move the non-offset cursor forward by one.
                    _sourceIsGood = _sourceCursor.MoveNext();

                    bool exitLoop = false;
                    while (!exitLoop && _offsetIsGood)
                    {
                        // Loop using the offset cursor until the native featurizer returns 1 result
                        _offsetIsGood = _offsetCursor.MoveNext();
                        if (!_offsetIsGood)
                            break;
                        try
                        {
                            // When the native featurizer returns the data from flush its not in the correct order.
                            // This is needed to put it back in the correct order.
                            _grainOrder.Add(GrainsToString(_grainGetters));
                            CreateGrainStringArrays(_grainGetters, ref _grainHandles, ref _grainArrayHandle, ref _grainArray);

                            // All columns have the same parameters, so if one is good then all are good and vice versa.
                            colMoveNext = _columns[0].MoveNext();

                            if (colMoveNext)
                                exitLoop = true;

                        }
                        finally
                        {
                            FreeGrainStringArrays(ref _grainHandles, ref _grainArrayHandle);
                        }
                    }

                    if (!_offsetIsGood)
                    {
                        // If we haven't flushed the data from the native featurizer yet, do it now.
                        // Only want to do this once.
                        if (!_hasFlushed)
                        {
                            foreach (var column in _columns)
                            {
                                column.Flush();
                            }
                            _hasFlushed = true;
                        }

                        if (_sourceIsGood)
                        {
                            foreach (var column in _columns)
                            {
                                column.MoveNext(_grainOrder[0]);
                            }
                        }
                    }

                    // Remove the first item every time we return from the move next call if the source is still good.
                    if (_sourceIsGood)
                        _grainOrder.RemoveAt(0);

                    _position++;
                    return _sourceIsGood;
                }

                public sealed override long Position => _position;

                public sealed override long Batch => _sourceCursor.Batch;

                private void InitializeGrainGetters(string[] grainColumns, ref ValueGetter<ReadOnlyMemory<char>>[] grainGetters)
                {
                    // Create getters for the source grain columns.

                    for (int i = 0; i < _grainGetters.Length; i++)
                    {
                        // Inititialize the getter and move it to a valid position.
                        grainGetters[i] = _offsetCursor.GetGetter<ReadOnlyMemory<char>>(_dataView.Schema[grainColumns[i]]);
                    }
                }

                #region Typed Columns

                // Safe handle that frees the memory for the transformed data.
                // Is called automatically after each call to transform.
                internal unsafe delegate bool DestroyLagLeadTransformedDataNative(IntPtr grainsPointer, IntPtr grainsPointerSize, IntPtr outputCols, IntPtr outputRows, double** output, IntPtr outputItems, out IntPtr errorHandle);
                internal unsafe class LagLeadTransformedDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
                {
                    private DestroyLagLeadTransformedDataNative _destroySaveDataHandler;
                    private IntPtr _grainsPointer;
                    private IntPtr _grainsPointerSize;
                    private IntPtr _outputCols;
                    private IntPtr _outputRows;
                    private double** _output;
                    private IntPtr _outputItems;
                    public unsafe LagLeadTransformedDataSafeHandle(IntPtr grainsPointer, IntPtr grainsPointerSize, IntPtr outputCols, IntPtr outputRows, double** output, IntPtr outputItems, DestroyLagLeadTransformedDataNative destroyNativeData) : base(true)
                    {
                        SetHandle(grainsPointer);
                        _destroySaveDataHandler = destroyNativeData;
                        _grainsPointer = grainsPointer;
                        _grainsPointerSize = grainsPointerSize;
                        _outputCols = outputCols;
                        _outputRows = outputRows;
                        _output = output;
                        _outputItems = outputItems;
                    }

                    protected override bool ReleaseHandle()
                    {
                        // Not sure what to do with error stuff here.  There shoudln't ever be one though.
                        return _destroySaveDataHandler(_grainsPointer, _grainsPointerSize, _outputCols, _outputRows, _output, _outputItems, out IntPtr errorHandle);
                    }
                }

                private abstract class TypedColumn : IDisposable
                {
                    private protected readonly TransformerEstimatorSafeHandle TransformerHandle;
                    private protected readonly Cursor Parent;
                    private protected readonly string Source;
                    internal readonly string Type;
                    internal readonly string Name;

                    internal TypedColumn(string name, string source, TransformerEstimatorSafeHandle transformer, Cursor parent, string type)
                    {
                        Source = source;
                        Name = name;
                        TransformerHandle = transformer;
                        Parent = parent;
                        Type = type;
                    }

                    internal abstract Delegate GetGetter();
                    internal abstract bool MoveNext(string grainString = "");
                    public void Dispose()
                    {
                        if (!TransformerHandle.IsClosed)
                            TransformerHandle.Dispose();
                    }

                    internal static TypedColumn CreateTypedColumn(string name, string source, string type, TransformerEstimatorSafeHandle transformer, Cursor parent)
                    {
                        if (type == typeof(double).ToString())
                        {
                            return new DoubleTypedColumn(name, source, transformer, parent);
                        }

                        throw new InvalidOperationException($"Column {source} has an unsupported type {type}.");
                    }

                    internal abstract void Flush();
                }

                private abstract class TypedColumn<TInput, TOutput> : TypedColumn
                {
                    private protected ValueGetter<TOutput> Getter;
                    private protected ValueGetter<TInput> SourceGetter;
                    private protected TOutput Result;
                    private protected TInput InputValue;

                    // When we call flush we get all the values back sorted correctly by grain, but if the original
                    // wasn't sorted by grain this order is incorrect. We need to store them by grain so that we can
                    // return them in the right order.
                    private protected Dictionary<string, List<TOutput>> FlushedValuesCache;

                    internal TypedColumn(string name, string source, TransformerEstimatorSafeHandle transformer, Cursor parent, string type) :
                        base(name, source, transformer, parent, type)
                    {
                        SourceGetter = parent._offsetCursor.GetGetter<TInput>(parent._dataView.Schema[source]);
                        Result = default;
                        InputValue = default;
                        FlushedValuesCache = new Dictionary<string, List<TOutput>>();
                    }

                    internal override Delegate GetGetter()
                    {
                        return Getter;
                    }

                    internal override bool MoveNext(string grainString = "")
                    {
                        if (Parent._offsetIsGood)
                        {
                            SourceGetter(ref InputValue);
                            if (!Transform())
                                return false;
                            return true;
                        }
                        else
                        {
                            // In this case flush has already been called. We just need to match the correct
                            // grain from the internal FlushedValuesCache.
                            Debug.Assert(grainString != "", "Grain string should never be empty at this point.");
                            Result = FlushedValuesCache[grainString][0];

                            // Now that Result is set to the correct value, remove index 0
                            FlushedValuesCache[grainString].RemoveAt(0);
                        }

                        // Once Parent._offsetIsGood is false we no longer care about the return value from this.
                        // Because of that, defaulting to return true;

                        return true;
                    }

                    internal abstract bool Transform();
                }

                private class DoubleTypedColumn : TypedColumn<double, VBuffer<double>>
                {
                    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, IntPtr grainsArray, IntPtr grainsArraySize, double value, out IntPtr grainsPointer, out IntPtr grainsPointerSize, out IntPtr outputCols, out IntPtr outputRows, out double** output, out IntPtr outputItems, out IntPtr errorHandle);

                    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_Flush", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                    private static unsafe extern bool FlushDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr grainsPointer, out IntPtr grainsPointerSize, out IntPtr outputCols, out IntPtr outputRows, out double** output, out IntPtr outputItems, out IntPtr errorHandle);

                    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_DestroyTransformedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
                    private static unsafe extern bool DestroyTransformedDataNative(IntPtr grainsPointer, IntPtr grainsPointerSize, IntPtr outputCols, IntPtr outputRows, double** output, IntPtr outputItems, out IntPtr errorHandle);

                    internal DoubleTypedColumn(string name, string source, TransformerEstimatorSafeHandle transformer, Cursor parent) :
                        base(name, source, transformer, parent, typeof(VBuffer<double>).ToString())
                    {
                        InitializeGetter();
                    }

                    private void InitializeGetter()
                    {
                        double input = default;
                        Getter = (ref VBuffer<double> dst) =>
                        {
                            // If the offset is no longer good, then this is the last row.
                            if (Parent._offsetIsGood)
                                SourceGetter(ref input);
                            unsafe
                            {
                                dst = Result;
                            }
                        };
                    }

                    internal unsafe override void Flush()
                    {
                        var success = FlushDataNative(TransformerHandle, out IntPtr grainsPointerPointer, out IntPtr grainsPointerSize, out IntPtr outputCols, out IntPtr outputRows, out double** output, out IntPtr outputItems, out IntPtr errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        var grainsPointerSizesP = (long*)grainsPointerSize.ToPointer();
                        var allGrainsPointer = (byte***)grainsPointerPointer.ToPointer();

                        for (int items = 0; items < outputItems.ToInt32(); items++)
                        {
                            var grainsArraySize = *grainsPointerSizesP++;
                            var grainArray = new string[grainsArraySize];
                            for (int grainItems = 0; grainItems < grainsArraySize; grainItems++)
                            {
                                var grainArrayPointer = *allGrainsPointer;
                                grainArray[grainItems] = PointerToString(new IntPtr(*grainArrayPointer));
                            }

                            VBuffer<double> res = default;

                            if (IntPtr.Size == 4)
                            {
                                res = ParseTransformResultx32((int*)outputCols.ToPointer(), (int*)outputRows.ToPointer(), output++, outputItems);
                                outputCols += 4;
                                outputRows += 4;
                            }
                            else
                            {
                                res = ParseTransformResultx64((long*)outputCols.ToPointer(), (long*)outputRows.ToPointer(), output++, outputItems);
                                outputCols += 8;
                                outputRows += 8;
                            }

                            var grainString = string.Join("", grainArray);

                            if (!FlushedValuesCache.ContainsKey(grainString))
                                FlushedValuesCache[grainString] = new List<VBuffer<double>>();

                            FlushedValuesCache[grainString].Add(res);
                        }
                    }

                    internal override unsafe bool Transform()
                    {
                        var success = TransformDataNative(TransformerHandle, Parent._grainArrayHandle.AddrOfPinnedObject(), new IntPtr(Parent._grainArray.Length), InputValue, out IntPtr grainsPointerPointer, out IntPtr grainsPointerSize, out IntPtr outputCols, out IntPtr outputRows, out double** output, out IntPtr outputItems, out IntPtr errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        if (outputItems == IntPtr.Zero)
                            return false;

                        using var handler = new LagLeadTransformedDataSafeHandle(grainsPointerPointer, grainsPointerSize, outputCols, outputRows, output, outputItems, DestroyTransformedDataNative);

                        // For the call to transform, we dont need to use the grainsPointerPointer or grainsPointerSizes, and there is only ever 1 row.
                        // These will be used when we flush all the output data, but the signature is the same for transform.
                        // Since we know there is only 1 output row, we can short circuit some of the loops.

                        // x32
                        if (IntPtr.Size == 4)
                            Result = ParseTransformResultx32((int*)outputCols.ToPointer(), (int*)outputRows.ToPointer(), output, outputItems);
                        else
                            Result = ParseTransformResultx64((long*)outputCols.ToPointer(), (long*)outputRows.ToPointer(), output, outputItems);

                        return true;
                    }

                    // For the call to transform, we dont need to use the grainsPointerPointer or grainsPointerSizes, and there is only ever 1 row.
                    // These will be used when we flush all the output data, but the signature is the same for transform.
                    // Since we know there is only 1 output row, we can short circuit some of the loops.
                    private unsafe VBuffer<double> ParseTransformResultx64(long* outputCols, long* outputRows, double** output, IntPtr outputItems)
                    {
                        var outputArray = new double[(*outputRows) * (*outputCols)];
                        var drefOuput = *output;

                        for (int i = 0; i < outputArray.Length; i++)
                        {
                            outputArray[i] = *drefOuput++;
                        }

                        return new VBuffer<double>(outputArray.Length, outputArray);
                    }

                    // For the call to transform, we dont need to use the grainsPointerPointer or grainsPointerSizes, and there is only ever 1 row.
                    // These will be used when we flush all the output data, but the signature is the same for transform.
                    // Since we know there is only 1 output row, we can short circuit some of the loops.
                    private unsafe VBuffer<double> ParseTransformResultx32(int* outputCols, int* outputRows, double** output, IntPtr outputItems)
                    {
                        var outputArray = new double[(*outputRows) * (*outputCols)];
                        var drefOuput = *output;

                        for (int i = 0; i < outputArray.Length; i++)
                        {
                            outputArray[i] = *drefOuput++;
                        }

                        return new VBuffer<double>(outputArray.Length, outputArray);
                    }
                }

                #endregion Typed Columns
            }

            #endregion Cursor
        }

        #endregion IDataView

        #region ColumnInfo

        #region BaseClass

        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Name;
            internal readonly string Source;
            internal readonly string Type;

            private protected TransformerEstimatorSafeHandle TransformerHandler;
            private protected LagLeadOperatorTransformer Parent;
            private static readonly Type[] _supportedTypes = new Type[] { typeof(double) };

            internal TypedColumn(string name, string source, string type, LagLeadOperatorTransformer parent)
            {
                Name = name;
                Source = source;
                Type = type;
                Parent = parent;
            }

            internal abstract void CreateTransformerFromEstimator(IDataView input);
            private protected abstract unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            private protected abstract bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected abstract bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            public abstract void Dispose();

            public abstract Type ReturnType();

            internal byte[] CreateTransformerSaveData()
            {

                var success = CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var savedDataHandle = new SaveDataSafeHandle(buffer, bufferSize))
                {
                    byte[] savedData = new byte[bufferSize.ToInt32()];
                    Marshal.Copy(buffer, savedData, 0, savedData.Length);
                    return savedData;
                }
            }

            internal unsafe void CreateTransformerFromSavedData(byte[] data)
            {
                fixed (byte* rawData = data)
                {
                    IntPtr dataSize = new IntPtr(data.Count());
                    TransformerHandler = CreateTransformerFromSavedDataHelper(rawData, dataSize);
                }
            }

            internal unsafe TransformerEstimatorSafeHandle CloneTransformer()
            {
                byte[] data = CreateTransformerSaveData();
                fixed (byte* rawData = data)
                {
                    IntPtr dataSize = new IntPtr(data.Count());
                    return CreateTransformerFromSavedDataHelper(rawData, dataSize);
                }
            }

            internal static bool IsColumnTypeSupported(Type type)
            {
                return _supportedTypes.Contains(type);
            }

            internal static TypedColumn CreateTypedColumn(string name, string source, string type, LagLeadOperatorTransformer parent)
            {
                if (type == typeof(double).ToString())
                {
                    return new DoubleTypedColumn(name, source, parent);
                }

                throw new InvalidOperationException($"Column {name} has an unsupported type {type}.");
            }
        }

        internal abstract class TypedColumn<TSourceType, TOutputType> : TypedColumn
        {
            private protected DataViewRowCursor Cursor;
            private protected ValueGetter<ReadOnlyMemory<char>>[] GrainGetters;

            internal TypedColumn(string name, string source, string type, LagLeadOperatorTransformer parent) :
                base(name, source, type, parent)
            {
                // Initialize to the correct length
                GrainGetters = new ValueGetter<ReadOnlyMemory<char>>[parent._options.GrainColumns.Length];
                Cursor = null;
            }

            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected unsafe abstract bool FitHelper(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, TSourceType value, out FitResult fitResult, out IntPtr errorHandle);
            private protected abstract bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

            private protected TransformerEstimatorSafeHandle CreateTransformerFromEstimatorBase(IDataView input)
            {
                var success = CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var estimatorHandle = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorHelper))
                {
                    TrainingState trainingState;
                    FitResult fitResult;

                    // Declare these outside the loop so the size is only set once;
                    GCHandle[] grainHandles = new GCHandle[Parent._options.GrainColumns.Length];
                    IntPtr[] grainArray = new IntPtr[Parent._options.GrainColumns.Length];
                    GCHandle arrayHandle = default;

                    InitializeGrainGetters(input);

                    // Can't use a using with this because it potentially needs to be reset. Manually disposing as needed.
                    var data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                    data.MoveNext();
                    while (true)
                    {
                        // Get the state of the native estimator.
                        success = GetStateHelper(estimatorHandle, out trainingState, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        // If we are no longer training then exit loop.
                        if (trainingState != TrainingState.Training)
                            break;

                        // Build the grain string array
                        try
                        {
                            CreateGrainStringArrays(GrainGetters, ref grainHandles, ref arrayHandle, ref grainArray);

                            // Train the estimator
                            success = FitHelper(estimatorHandle, arrayHandle.AddrOfPinnedObject(), new IntPtr(grainArray.Length), data.Current, out fitResult, out errorHandle);
                            if (!success)
                                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        }
                        finally
                        {
                            FreeGrainStringArrays(ref grainHandles, ref arrayHandle);
                        }

                        // If we need to reset the data to the beginning.
                        if (fitResult == FitResult.ResetAndContinue)
                        {
                            data.Dispose();
                            data = input.GetColumn<TSourceType>(Source).GetEnumerator();

                            InitializeGrainGetters(input);
                        }

                        // If we are at the end of the data.
                        if (!data.MoveNext() && !Cursor.MoveNext())
                        {
                            OnDataCompletedHelper(estimatorHandle, out errorHandle);
                            if (!success)
                                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                            // Re-initialize the data
                            data.Dispose();
                            data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                            data.MoveNext();

                            InitializeGrainGetters(input);
                        }
                    }

                    // When done training complete the estimator.
                    success = CompleteTrainingHelper(estimatorHandle, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // Create the native transformer from the estimator;
                    success = CreateTransformerFromEstimatorHelper(estimatorHandle, out IntPtr transformer, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // Manually dispose of the IEnumerator and Cursor since we dont have a using statement;
                    data.Dispose();
                    Cursor.Dispose();

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }

            private void InitializeGrainGetters(IDataView input)
            {
                // Create getters for the grain columns. Cant use using for the cursor because it may need to be reset.
                // Manually dispose of the cursor if its not null
                if (Cursor != null)
                    Cursor.Dispose();

                Cursor = input.GetRowCursor(input.Schema.Where(x => Parent._options.GrainColumns.Contains(x.Name)));

                for (int i = 0; i < Parent._options.GrainColumns.Length; i++)
                {
                    // Inititialize the enumerator and move it to a valid position.
                    GrainGetters[i] = Cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent._options.GrainColumns[i]]);
                }

                // Move cursor to valid spot.
                Cursor.MoveNext();
            }

            public override Type ReturnType()
            {
                return typeof(TOutputType);
            }
        }

        #endregion BaseClass

        #region DoubleTypedColumn

        internal sealed class DoubleTypedColumn : TypedColumn<double, VBuffer<double>>
        {
            internal DoubleTypedColumn(string name, string source, LagLeadOperatorTransformer parent) :
                base(name, source, typeof(double).ToString(), parent)
            {
            }

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateEstimatorNative(UInt32 horizon, long* offsets, IntPtr offsetsSize, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                /* Codegen: do any extra checks/paramters here */
                unsafe
                {
                    fixed (long* offsetPointer = Parent._options.Offsets)
                    {
                        return CreateEstimatorNative(Parent._options.Horizon, offsetPointer, new IntPtr(Parent._options.Offsets.Length), out estimator, out errorHandle);
                    }
                }
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, double value, out FitResult fitResult, out IntPtr errorHandle);
            private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, double value, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, grainsArray, grainsArraySize, value, out fitResult, out errorHandle);
            }

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }
        }

        #endregion DoubleTypedColumn

        #endregion ColumnInfo
    }

    internal static class LagLeadOperatorEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.LagLeadOperator",
            Desc = LagLeadOperatorTransformer.Summary,
            UserName = LagLeadOperatorTransformer.UserName,
            ShortName = LagLeadOperatorTransformer.ShortName)]
        public static CommonOutputs.TransformOutput LagLeadOperator(IHostEnvironment env, LagLeadOperatorEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, LagLeadOperatorTransformer.ShortName, input);
            var xf = new LagLeadOperatorEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}