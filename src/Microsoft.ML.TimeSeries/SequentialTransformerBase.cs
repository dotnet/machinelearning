// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{

    /// <summary>
    /// The base class for sequential processing transforms. This class implements the basic sliding window buffering. The derived classes need to specify the transform logic,
    /// the initialization logic and the learning logic via implementing the abstract methods TransformCore(), InitializeStateCore() and LearnStateFromDataCore(), respectively
    /// </summary>
    /// <typeparam name="TInput">The input type of the sequential processing.</typeparam>
    /// <typeparam name="TOutput">The dst type of the sequential processing.</typeparam>
    /// <typeparam name="TState">The dst type of the sequential processing.</typeparam>
    internal abstract class SequentialTransformerBase<TInput, TOutput, TState> : IStatefulTransformer
        where TState : SequentialTransformerBase<TInput, TOutput, TState>.StateBase, new()
    {

        /// <summary>
        /// The base class for encapsulating the State object for sequential processing. This class implements a windowed buffer.
        /// </summary>
        internal class StateBase
        {
            // Ideally this class should be private. However, due to the current constraints with the LambdaTransform, we need to have
            // access to the state class when inheriting from SequentialTransformerBase.
            private protected IHost Host;

            /// <summary>
            /// A reference to the parent transform that operates on the state object.
            /// </summary>
            protected SequentialTransformerBase<TInput, TOutput, TState> ParentTransform;

            /// <summary>
            /// The internal windowed buffer for buffering the values in the input sequence.
            /// </summary>
            private protected FixedSizeQueue<TInput> WindowedBuffer { get; set; }

            /// <summary>
            /// The buffer used to buffer the training data points.
            /// </summary>
            private protected FixedSizeQueue<TInput> InitialWindowedBuffer { get; set; }

            private protected int WindowSize { get; private set; }

            private protected int InitialWindowSize { get; private set; }

            /// <summary>
            /// Counts the number of rows observed by the transform so far.
            /// </summary>
            protected long RowCounter { get; private set; }

            public StateBase()
            {
            }

            protected long IncrementRowCounter()
            {
                RowCounter++;
                return RowCounter;
            }

            protected long PreviousPosition;

            private protected StateBase(BinaryReader reader)
            {
                WindowSize = reader.ReadInt32();
                InitialWindowSize = reader.ReadInt32();
            }

            internal virtual void Save(BinaryWriter writer)
            {
                writer.Write(WindowSize);
                writer.Write(InitialWindowSize);
            }

            /// <summary>
            /// This method sets the window size and initializes the buffer only once.
            /// Since the class needs to implement a default constructor, this methods provides a mechanism to initialize the window size and buffer.
            /// </summary>
            /// <param name="windowSize">The size of the windowed buffer</param>
            /// <param name="initialWindowSize">The size of the windowed initial buffer used for training</param>
            /// <param name="parentTransform">The parent transform of this state object</param>
            /// <param name="host">The host</param>
            public void InitState(int windowSize, int initialWindowSize, SequentialTransformerBase<TInput, TOutput, TState> parentTransform,
                IHost host)
            {
                Contracts.CheckValue(host, nameof(host), "The host cannot be null.");
                host.CheckValue(parentTransform, nameof(parentTransform));
                host.CheckParam(windowSize >= 0, nameof(windowSize), "Must be non-negative.");
                host.CheckParam(initialWindowSize >= 0, nameof(initialWindowSize), "Must be non-negative.");

                Host = host;
                WindowSize = windowSize;
                InitialWindowSize = initialWindowSize;
                ParentTransform = parentTransform;
                WindowedBuffer = (WindowSize > 0) ? new FixedSizeQueue<TInput>(WindowSize) : new FixedSizeQueue<TInput>(1);
                InitialWindowedBuffer = (InitialWindowSize > 0) ? new FixedSizeQueue<TInput>(InitialWindowSize) : new FixedSizeQueue<TInput>(1);
                RowCounter = 0;

                InitializeStateCore();
                PreviousPosition = -1;
            }

            public void InitState(SequentialTransformerBase<TInput, TOutput, TState> parentTransform, IHost host)
            {
                Contracts.CheckValue(host, nameof(host), "The host cannot be null.");
                host.CheckValue(parentTransform, nameof(parentTransform));

                Host = host;
                ParentTransform = parentTransform;
                RowCounter = 0;
                InitializeStateCore(true);
                PreviousPosition = -1;
            }

            /// <summary>
            /// This method implements the basic resetting mechanism for a state object and clears the buffer.
            /// </summary>
            public virtual void Reset()
            {
                Host.Assert(WindowedBuffer != null);
                Host.Assert(InitialWindowedBuffer != null);

                RowCounter = 0;
                WindowedBuffer.Clear();
                InitialWindowedBuffer.Clear();
                PreviousPosition = -1;
            }

            public void UpdateState(ref TInput input, long rowPosition, bool buffer = true)
            {
                if (rowPosition > PreviousPosition)
                {
                    PreviousPosition = rowPosition;
                    UpdateStateCore(ref input, buffer);
                    Consume(input);
                }
            }

            public void UpdateStateCore(ref TInput input, bool buffer = true)
            {
                if (InitialWindowedBuffer.Count < InitialWindowSize)
                {
                    InitialWindowedBuffer.AddLast(input);
                    if (InitialWindowedBuffer.Count >= InitialWindowSize - WindowSize && buffer)
                        WindowedBuffer.AddLast(input);
                }
                else
                {
                    if (buffer)
                        WindowedBuffer.AddLast(input);

                    IncrementRowCounter();
                }
            }

            public void Process(ref TInput input, ref TOutput output)
            {
                if (PreviousPosition == -1)
                    UpdateStateCore(ref input);

                if (InitialWindowedBuffer.Count < InitialWindowSize)
                {
                    SetNaOutput(ref output);

                    if (InitialWindowedBuffer.Count == InitialWindowSize)
                        LearnStateFromDataCore(InitialWindowedBuffer);
                }
                else
                {
                    TransformCore(ref input, WindowedBuffer, RowCounter - InitialWindowSize, ref output);
                }
            }

            public void ProcessWithoutBuffer(ref TInput input, ref TOutput output)
            {
                if (PreviousPosition == -1)
                    UpdateStateCore(ref input, false);

                if (InitialWindowedBuffer.Count < InitialWindowSize)
                {
                    SetNaOutput(ref output);

                    if (InitialWindowedBuffer.Count == InitialWindowSize)
                        LearnStateFromDataCore(InitialWindowedBuffer);
                }
                else
                    TransformCore(ref input, WindowedBuffer, RowCounter - InitialWindowSize, ref output);
            }

            /// <summary>
            /// The abstract method that specifies the NA value for the dst type.
            /// </summary>
            /// <returns></returns>
            private protected virtual void SetNaOutput(ref TOutput dst) { }

            /// <summary>
            /// The abstract method that realizes the main logic for the transform.
            /// </summary>
            /// <param name="input">A reference to the input object.</param>
            /// <param name="dst">A reference to the dst object.</param>
            /// <param name="windowedBuffer">A reference to the windowed buffer.</param>
            /// <param name="iteration">A long number that indicates the number of times TransformCore has been called so far (starting value = 0).</param>
            private protected virtual void TransformCore(ref TInput input, FixedSizeQueue<TInput> windowedBuffer, long iteration, ref TOutput dst)
            {

            }

            /// <summary>
            /// The abstract method that realizes the logic for initializing the state object.
            /// </summary>
            private protected virtual void InitializeStateCore(bool disk = false)
            {

            }

            /// <summary>
            /// The abstract method that realizes the logic for learning the parameters and the initial state object from data.
            /// </summary>
            /// <param name="data">A queue of data points used for training</param>
            private protected virtual void LearnStateFromDataCore(FixedSizeQueue<TInput> data)
            {
            }

            public virtual void Consume(TInput value)
            {

            }

            public TState Clone()
            {
                var clone = (TState)MemberwiseClone();
                CloneCore(clone);
                return clone;
            }

            private protected virtual void CloneCore(TState state)
            {
                state.WindowedBuffer = WindowedBuffer.Clone();
                state.InitialWindowedBuffer = InitialWindowedBuffer.Clone();
            }
        }

        internal readonly IHost Host;

        /// <summary>
        /// The window size for buffering.
        /// </summary>
        internal readonly int WindowSize;

        /// <summary>
        /// The number of datapoints from the beginning of the sequence that are used for learning the initial state.
        /// </summary>
        private protected int InitialWindowSize;

        internal readonly string InputColumnName;
        internal readonly string OutputColumnName;
        private protected DataViewType OutputColumnType;

        bool ITransformer.IsRowToRowMapper => false;

        internal TState StateRef { get; set; }

        public int StateRefCount;

        /// <summary>
        /// The main constructor for the sequential transform
        /// </summary>
        /// <param name="host">The host.</param>
        /// <param name="windowSize">The size of buffer used for windowed buffering.</param>
        /// <param name="initialWindowSize">The number of datapoints picked from the beginning of the series for training the transform parameters if needed.</param>
        /// <param name="outputColumnName">The name of the dst column.</param>
        /// <param name="inputColumnName">The name of the input column.</param>
        /// <param name="outputColType"></param>
        private protected SequentialTransformerBase(IHost host, int windowSize, int initialWindowSize, string outputColumnName, string inputColumnName, DataViewType outputColType)
        {
            Host = host;
            Host.CheckParam(initialWindowSize >= 0, nameof(initialWindowSize), "Must be non-negative.");
            Host.CheckParam(windowSize >= 0, nameof(windowSize), "Must be non-negative.");
            // REVIEW: Very bad design. This base class is responsible for reporting errors on
            // the arguments, but the arguments themselves are not derived form any base class.
            Host.CheckNonEmpty(inputColumnName, nameof(PercentileThresholdTransform.Arguments.Source));
            Host.CheckNonEmpty(outputColumnName, nameof(PercentileThresholdTransform.Arguments.Source));

            InputColumnName = inputColumnName;
            OutputColumnName = outputColumnName;
            OutputColumnType = outputColType;
            InitialWindowSize = initialWindowSize;
            WindowSize = windowSize;
        }

        private protected SequentialTransformerBase(IHost host, ModelLoadContext ctx)
        {
            Host = host;
            Host.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // int: _windowSize
            // int: _initialWindowSize
            // int (string ID): _sourceColumnName
            // int (string ID): _outputColumnName
            // ColumnType: _transform.Schema.GetColumnType(0)

            var windowSize = ctx.Reader.ReadInt32();
            Host.CheckDecode(windowSize >= 0);

            var initialWindowSize = ctx.Reader.ReadInt32();
            Host.CheckDecode(initialWindowSize >= 0);

            var inputColumnName = ctx.LoadNonEmptyString();
            var outputColumnName = ctx.LoadNonEmptyString();

            InputColumnName = inputColumnName;
            OutputColumnName = outputColumnName;
            InitialWindowSize = initialWindowSize;
            WindowSize = windowSize;

            BinarySaver bs = new BinarySaver(Host, new BinarySaver.Arguments());
            OutputColumnType = bs.LoadTypeDescriptionOrNull(ctx.Reader.BaseStream);
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        private protected virtual void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(InitialWindowSize >= 0);
            Host.Assert(WindowSize >= 0);

            // *** Binary format ***
            // int: _windowSize
            // int: _initialWindowSize
            // int (string ID): _sourceColumnName
            // int (string ID): _outputColumnName
            // ColumnType: _transform.Schema.GetColumnType(0)

            ctx.Writer.Write(WindowSize);
            ctx.Writer.Write(InitialWindowSize);
            ctx.SaveNonEmptyString(InputColumnName);
            ctx.SaveNonEmptyString(OutputColumnName);

            var bs = new BinarySaver(Host, new BinarySaver.Arguments());
            bs.TryWriteTypeDescription(ctx.Writer.BaseStream, OutputColumnType, out int byteWritten);
        }

        public abstract DataViewSchema GetOutputSchema(DataViewSchema inputSchema);

        internal abstract IStatefulRowMapper MakeRowMapper(DataViewSchema schema);

        internal SequentialDataTransform MakeDataTransform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return new SequentialDataTransform(Host, this, input, MakeRowMapper(input.Schema));
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        {
            throw new InvalidOperationException("Not a RowToRowMapper.");
        }

        IRowToRowMapper IStatefulTransformer.GetStatefulRowToRowMapper(DataViewSchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            return new TimeSeriesRowToRowMapperTransform(Host, new EmptyDataView(Host, inputSchema), MakeRowMapper(inputSchema));
        }

        internal virtual IStatefulTransformer Clone() => (SequentialTransformerBase<TInput, TOutput, TState>)MemberwiseClone();

        IStatefulTransformer IStatefulTransformer.Clone() => Clone();

        internal sealed class SequentialDataTransform : TransformBase, ITransformTemplate, IRowToRowMapper
        {
            private readonly IStatefulRowMapper _mapper;
            private readonly SequentialTransformerBase<TInput, TOutput, TState> _parent;
            private readonly IDataTransform _transform;
            private readonly ColumnBindings _bindings;

            private MetadataDispatcher Metadata { get; }

            public SequentialDataTransform(IHost host, SequentialTransformerBase<TInput, TOutput, TState> parent,
                IDataView input, IStatefulRowMapper mapper)
                : base(parent.Host, input)
            {
                Metadata = new MetadataDispatcher(1);
                _parent = parent;
                _transform = CreateLambdaTransform(_parent.Host, input, _parent.InputColumnName,
                    _parent.OutputColumnName, InitFunction, _parent.WindowSize > 0, _parent.OutputColumnType);
                _mapper = mapper;
                _bindings = new ColumnBindings(input.Schema, _mapper.GetOutputColumns());
            }

            public void CloneStateInMapper() => _mapper.CloneState();

            private static IDataTransform CreateLambdaTransform(IHost host, IDataView input, string inputColumnName,
                string outputColumnName, Action<TState> initFunction, bool hasBuffer, DataViewType outputColTypeOverride)
            {
                var inputSchema = SchemaDefinition.Create(typeof(DataBox<TInput>));
                inputSchema[0].ColumnName = inputColumnName;

                var outputSchema = SchemaDefinition.Create(typeof(DataBox<TOutput>));
                outputSchema[0].ColumnName = outputColumnName;

                if (outputColTypeOverride != null)
                    outputSchema[0].ColumnType = outputColTypeOverride;

                Action<DataBox<TInput>, DataBox<TOutput>, TState> lambda;
                if (hasBuffer)
                    lambda = MapFunction;
                else
                    lambda = MapFunctionWithoutBuffer;

                return LambdaTransform.CreateMap(host, input, lambda, initFunction, inputSchema, outputSchema);
            }

            private static void MapFunction(DataBox<TInput> input, DataBox<TOutput> output, TState state)
            {
                state.Process(ref input.Value, ref output.Value);
            }

            private static void MapFunctionWithoutBuffer(DataBox<TInput> input, DataBox<TOutput> output, TState state)
            {
                state.ProcessWithoutBuffer(ref input.Value, ref output.Value);
            }

            private void InitFunction(TState state)
            {
                state.InitState(_parent.WindowSize, _parent.InitialWindowSize, _parent, _parent.Host);
            }

            public override bool CanShuffle { get { return false; } }

            protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                var srcCursor = _transform.GetRowCursor(columnsNeeded, rand);
                var clone = (SequentialDataTransform)MemberwiseClone();
                clone.CloneStateInMapper();
                return new Cursor(Host, clone, srcCursor);
            }

            protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
            {
                Host.AssertValue(predicate);
                return false;
            }

            public override long? GetRowCount()
                => _transform.GetRowCount();

            public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
                => new DataViewRowCursor[] { GetRowCursorCore(columnsNeeded, rand) };

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                (_parent as ICanSaveModel).Save(ctx);
            }

            IDataTransform ITransformTemplate.ApplyToData(IHostEnvironment env, IDataView newSource)
            {
                return new SequentialDataTransform(Contracts.CheckRef(env, nameof(env)).Register("SequentialDataTransform"), _parent, newSource, _mapper);
            }

            public DataViewSchema InputSchema => Source.Schema;

            public override DataViewSchema OutputSchema => _bindings.Schema;

            /// <summary>
            /// Given a set of columns, return the input columns that are needed to generate those output columns.
            /// </summary>
            public IEnumerable<DataViewSchema.Column> GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns)
            {
                if (dependingColumns.Count() == 0)
                    return Enumerable.Empty<DataViewSchema.Column>();

                return InputSchema;
            }

            DataViewRow IRowToRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                var active = RowCursorUtils.FromColumnsToPredicate(activeColumns, OutputSchema);
                var getters = _mapper.CreateGetters(input, active, out Action disposer);
                var pingers = _mapper.CreatePinger(input, active, out Action pingerDisposer);
                return new RowImpl(_bindings.Schema, input, getters, pingers, disposer + pingerDisposer);
            }
        }

        private sealed class RowImpl : StatefulRow
        {
            private readonly DataViewSchema _schema;
            private readonly DataViewRow _input;
            private readonly Delegate[] _getters;
            private readonly Action<long> _pinger;
            private readonly Action _disposer;
            private bool _disposed;

            public override DataViewSchema Schema => _schema;

            public override long Position => _input.Position;

            public override long Batch => _input.Batch;

            public RowImpl(DataViewSchema schema, DataViewRow input, Delegate[] getters, Action<long> pinger, Action disposer)
            {
                Contracts.CheckValue(schema, nameof(schema));
                Contracts.CheckValue(input, nameof(input));
                Contracts.Check(Utils.Size(getters) == schema.Count);
                _schema = schema;
                _input = input;
                _getters = getters ?? new Delegate[0];
                _pinger = pinger;
                _disposer = disposer;
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

            public override ValueGetter<DataViewRowId> GetIdGetter()
                => _input.GetIdGetter();

            public override ValueGetter<T> GetGetter<T>(DataViewSchema.Column column)
            {
                Contracts.CheckParam(column.Index < _getters.Length, nameof(column), "Invalid col value in GetGetter");
                Contracts.Check(IsColumnActive(column));
                var fn = _getters[column.Index] as ValueGetter<T>;
                if (fn == null)
                    throw Contracts.Except("Unexpected TValue in GetGetter");
                return fn;
            }

            public override Action<long> GetPinger() =>
                _pinger as Action<long> ?? throw Contracts.Except("Invalid TValue in GetPinger: '{0}'", typeof(long));

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Contracts.Check(column.Index < _getters.Length);
                return _getters[column.Index] != null;
            }
        }

        /// <summary>
        /// A wrapper around the cursor which replaces the schema.
        /// </summary>
        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly SequentialDataTransform _parent;

            public Cursor(IHost host, SequentialDataTransform parent, DataViewRowCursor input)
                : base(host, input)
            {
                Ch.Assert(input.Schema.Count == parent.OutputSchema.Count);
                _parent = parent;
            }

            public override DataViewSchema Schema => _parent.OutputSchema;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < Schema.Count, nameof(column));
                return Input.IsColumnActive(column);
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
                Ch.Check(IsColumnActive(column), nameof(column));
                return Input.GetGetter<TValue>(column);
            }
        }
    }

    /// <summary>
    /// This class is a transform that can add any number of output columns, that depend on any number of input columns.
    /// It does so with the help of an <see cref="IRowMapper"/>, that is given a schema in its constructor, and has methods
    /// to get the dependencies on input columns and the getters for the output columns, given an active set of output columns.
    /// </summary>

    internal sealed class TimeSeriesRowToRowMapperTransform : RowToRowTransformBase, IStatefulRowToRowMapper,
        ITransformCanSaveOnnx, ITransformCanSavePfa
    {
        private readonly IStatefulRowMapper _mapper;
        private readonly ColumnBindings _bindings;
        public const string RegistrationName = "TimeSeriesRowToRowMapperTransform";
        public const string LoaderSignature = "TimeSeriesRowToRowMapper";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TS ROW MPPR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TimeSeriesRowToRowMapperTransform).Assembly.FullName);
        }

        public override DataViewSchema OutputSchema => _bindings.Schema;

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => _mapper is ICanSaveOnnx onnxMapper ? onnxMapper.CanSaveOnnx(ctx) : false;

        bool ICanSavePfa.CanSavePfa => _mapper is ICanSavePfa pfaMapper ? pfaMapper.CanSavePfa : false;

        public TimeSeriesRowToRowMapperTransform(IHostEnvironment env, IDataView input, IStatefulRowMapper mapper)
            : base(env, RegistrationName, input)
        {
            Contracts.CheckValue(mapper, nameof(mapper));
            _mapper = mapper;
            _bindings = new ColumnBindings(input.Schema, mapper.GetOutputColumns());
        }

        public static DataViewSchema GetOutputSchema(DataViewSchema inputSchema, IRowMapper mapper)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.CheckValue(mapper, nameof(mapper));
            return new ColumnBindings(inputSchema, mapper.GetOutputColumns()).Schema;
        }

        private TimeSeriesRowToRowMapperTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            // *** Binary format ***
            // _mapper

            ctx.LoadModel<IStatefulRowMapper, SignatureLoadRowMapper>(host, out _mapper, "Mapper", input.Schema);
            _bindings = new ColumnBindings(input.Schema, _mapper.GetOutputColumns());
        }

        public static TimeSeriesRowToRowMapperTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model", ch => new TimeSeriesRowToRowMapperTransform(h, ctx, input));
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
        /// a predicate for the needed active input columns, and a predicate for the needed active
        /// output columns.
        /// </summary>
        private bool[] GetActive(Func<int, bool> predicate, out Func<int, bool> predicateInput)
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
            predicateInput =
                col => 0 <= col && col < activeInput.Length && (activeInput[col] || predicateIn(col));

            return active;
        }

        /// <summary>
        /// Produces the set of active columns for the data view (as a bool[] of length bindings.ColumnCount),
        /// a predicate for the needed active input columns, and a predicate for the needed active
        /// output columns.
        /// </summary>
        private IEnumerable<DataViewSchema.Column> GetActive(Func<int, bool> predicate)
        {
            Func<int, bool> predicateInput;

            var active = GetActive(predicate, out predicateInput);
            return _bindings.Schema.Where(col => predicateInput(col.Index));
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
            Func<int, bool> predicateInput;
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var active = GetActive(predicate, out predicateInput);
            var inputCols = Source.Schema.Where(x => predicateInput(x.Index));
            return new Cursor(Host, Source.GetRowCursor(inputCols, rand), this, active);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);

            Func<int, bool> predicateInput;
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var active = GetActive(predicate, out predicateInput);

            var inputCols = Source.Schema.Where(x => predicateInput(x.Index));
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
        /// Given a set of columns, return the input columns that are needed to generate those output columns.
        /// </summary>
        public IEnumerable<DataViewSchema.Column> GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns)
        {
            var predicate = RowCursorUtils.FromColumnsToPredicate(dependingColumns, OutputSchema);
            return GetActive(predicate);
        }

        DataViewSchema IRowToRowMapper.InputSchema => Source.Schema;

        DataViewRow IRowToRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
        {
            Host.CheckValue(input, nameof(input));
            Host.CheckValue(activeColumns, nameof(activeColumns));
            Host.Check(input.Schema == Source.Schema, "Schema of input row must be the same as the schema the mapper is bound to");

            using (var ch = Host.Start("GetEntireRow"))
            {
                var activeArr = Utils.BuildArray(OutputSchema.Count, activeColumns);
                var pred = GetActiveOutputColumns(activeArr);
                var getters = _mapper.CreateGetters(input, pred, out Action disp);
                var pingers = _mapper.CreatePinger(input, pred, out Action pingerDisp);
                return new StatefulRowImpl(input, this, OutputSchema, getters, pingers, disp + pingerDisp);
            }
        }

        private sealed class StatefulRowImpl : StatefulRow
        {
            private readonly DataViewRow _input;
            private readonly Delegate[] _getters;
            private readonly Action<long> _pinger;
            private readonly Action _disposer;

            private readonly TimeSeriesRowToRowMapperTransform _parent;

            public override long Batch => _input.Batch;

            public override long Position => _input.Position;

            public override DataViewSchema Schema { get; }

            public StatefulRowImpl(DataViewRow input, TimeSeriesRowToRowMapperTransform parent,
                DataViewSchema schema, Delegate[] getters, Action<long> pinger, Action disposer)
            {
                _input = input;
                _parent = parent;
                Schema = schema;
                _getters = getters;
                _pinger = pinger;
                _disposer = disposer;
            }

            protected override void Dispose(bool disposing)
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
                    return _input.GetGetter<TValue>(column);

                Contracts.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Contracts.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            public override Action<long> GetPinger() =>
                _pinger as Action<long> ?? throw Contracts.Except("Invalid TValue in GetPinger: '{0}'", typeof(long));

            public override ValueGetter<DataViewRowId> GetIdGetter() => _input.GetIdGetter();

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                bool isSrc;
                int index = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return _input.IsColumnActive(_input.Schema[index]);
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

            public Cursor(IChannelProvider provider, DataViewRowCursor input, TimeSeriesRowToRowMapperTransform parent, bool[] active)
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
                    return Input.GetGetter<TValue>(column);

                Ch.AssertValue(_getters);
                var getter = _getters[index];
                Ch.AssertValue(getter);
                if (getter is ValueGetter<TValue> fn)
                    return fn;
                throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
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
    }
}
