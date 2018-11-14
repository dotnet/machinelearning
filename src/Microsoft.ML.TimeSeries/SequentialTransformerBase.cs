// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Core.Data;
using Microsoft.ML.TimeSeries;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using System.Linq;
using Microsoft.ML;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// The base class for sequential processing transforms. This class implements the basic sliding window buffering. The derived classes need to specify the transform logic,
    /// the initialization logic and the learning logic via implementing the abstract methods TransformCore(), InitializeStateCore() and LearnStateFromDataCore(), respectively
    /// </summary>
    /// <typeparam name="TInput">The input type of the sequential processing.</typeparam>
    /// <typeparam name="TOutput">The dst type of the sequential processing.</typeparam>
    /// <typeparam name="TState">The state type of the sequential processing. Must be a class inherited from StateBase </typeparam>
    public abstract class SequentialTransformerBase<TInput, TOutput, TState> : ITransformer, ICanSaveModel
       where TState : SequentialTransformerBase<TInput, TOutput, TState>.StateBase, new()
    {
        /// <summary>
        /// The base class for encapsulating the State object for sequential processing. This class implements a windowed buffer.
        /// </summary>
        public abstract class StateBase
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
            private protected FixedSizeQueue<TInput> WindowedBuffer;

            /// <summary>
            /// The buffer used to buffer the training data points.
            /// </summary>
            private protected FixedSizeQueue<TInput> InitialWindowedBuffer;

            private protected int WindowSize { get; private set; }

            private protected int InitialWindowSize { get; private set; }

            /// <summary>
            /// Counts the number of rows observed by the transform so far.
            /// </summary>
            protected long RowCounter { get; private set; }

            private protected StateBase()
            {
            }

            protected long IncrementRowCounter()
            {
                RowCounter++;
                return RowCounter;
            }

            private bool _isIniatilized;

            private long _previousPosition;

            public IRow Row;

            /// <summary>
            /// This method sets the window size and initializes the buffer only once.
            /// Since the class needs to implement a default constructor, this methods provides a mechanism to initialize the window size and buffer.
            /// </summary>
            /// <param name="windowSize">The size of the windowed buffer</param>
            /// <param name="initialWindowSize">The size of the windowed initial buffer used for training</param>
            /// <param name="parentTransform">The parent transform of this state object</param>
            /// <param name="host">The host</param>
            public void InitState(int windowSize, int initialWindowSize, SequentialTransformerBase<TInput, TOutput, TState> parentTransform, IHost host)
            {
                Contracts.CheckValue(host, nameof(host), "The host cannot be null.");
                host.Check(!_isIniatilized, "The window size can be set only once.");
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
                _isIniatilized = true;
                _previousPosition = -1;
            }

            /// <summary>
            /// This method implements the basic resetting mechanism for a state object and clears the buffer.
            /// </summary>
            public virtual void Reset()
            {
                Host.Assert(_isIniatilized);
                Host.Assert(WindowedBuffer != null);
                Host.Assert(InitialWindowedBuffer != null);

                RowCounter = 0;
                WindowedBuffer.Clear();
                InitialWindowedBuffer.Clear();
                _previousPosition = -1;
            }

            public void Process(ref TInput input, ref TOutput output)
            {
                bool updateState = false;
                if (Row.Position > _previousPosition)
                {
                    _previousPosition = Row.Position;
                    updateState = true;
                }

                if (InitialWindowedBuffer.Count < InitialWindowSize)
                {
                    if (updateState)
                        InitialWindowedBuffer.AddLast(input);

                    SetNaOutput(ref output);

                    if (InitialWindowedBuffer.Count >= InitialWindowSize - WindowSize && updateState)
                        WindowedBuffer.AddLast(input);

                    if (InitialWindowedBuffer.Count == InitialWindowSize)
                        LearnStateFromDataCore(InitialWindowedBuffer);
                }
                else
                {
                    TransformCore(ref input, WindowedBuffer, RowCounter - InitialWindowSize, ref output);
                    if (updateState)
                    {
                        WindowedBuffer.AddLast(input);
                        IncrementRowCounter();
                    }
                }
            }

            public void ProcessWithoutBuffer(ref TInput input, ref TOutput output)
            {
                bool updateState = false;
                if (Row.Position > _previousPosition)
                {
                    _previousPosition = Row.Position;
                    updateState = true;
                }

                if (InitialWindowedBuffer.Count < InitialWindowSize)
                {
                    if (updateState)
                        InitialWindowedBuffer.AddLast(input);

                    SetNaOutput(ref output);

                    if (InitialWindowedBuffer.Count == InitialWindowSize)
                        LearnStateFromDataCore(InitialWindowedBuffer);
                }
                else
                {
                    TransformCore(ref input, WindowedBuffer, RowCounter - InitialWindowSize, ref output);
                    if (updateState)
                        IncrementRowCounter();
                }
            }

            /// <summary>
            /// The abstract method that specifies the NA value for the dst type.
            /// </summary>
            /// <returns></returns>
            private protected abstract void SetNaOutput(ref TOutput dst);

            /// <summary>
            /// The abstract method that realizes the main logic for the transform.
            /// </summary>
            /// <param name="input">A reference to the input object.</param>
            /// <param name="dst">A reference to the dst object.</param>
            /// <param name="windowedBuffer">A reference to the windowed buffer.</param>
            /// <param name="iteration">A long number that indicates the number of times TransformCore has been called so far (starting value = 0).</param>
            private protected abstract void TransformCore(ref TInput input, FixedSizeQueue<TInput> windowedBuffer, long iteration, ref TOutput dst);

            /// <summary>
            /// The abstract method that realizes the logic for initializing the state object.
            /// </summary>
            private protected abstract void InitializeStateCore();

            /// <summary>
            /// The abstract method that realizes the logic for learning the parameters and the initial state object from data.
            /// </summary>
            /// <param name="data">A queue of data points used for training</param>
            private protected abstract void LearnStateFromDataCore(FixedSizeQueue<TInput> data);
        }

        private protected readonly IHost Host;

        /// <summary>
        /// The window size for buffering.
        /// </summary>
        private protected readonly int WindowSize;

        /// <summary>
        /// The number of datapoints from the beginning of the sequence that are used for learning the initial state.
        /// </summary>
        private protected int InitialWindowSize;

        internal readonly string InputColumnName;
        internal readonly string OutputColumnName;
        private protected ColumnType OutputColumnType;

        public bool IsRowToRowMapper => true;

        /// <summary>
        /// The main constructor for the sequential transform
        /// </summary>
        /// <param name="host">The host.</param>
        /// <param name="windowSize">The size of buffer used for windowed buffering.</param>
        /// <param name="initialWindowSize">The number of datapoints picked from the beginning of the series for training the transform parameters if needed.</param>
        /// <param name="inputColumnName">The name of the input column.</param>
        /// <param name="outputColumnName">The name of the dst column.</param>
        /// <param name="outputColType"></param>
        private protected SequentialTransformerBase(IHost host, int windowSize, int initialWindowSize, string inputColumnName, string outputColumnName, ColumnType outputColType)
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
            // int (string ID): _inputColumnName
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

        public virtual void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(InitialWindowSize >= 0);
            Host.Assert(WindowSize >= 0);

            // *** Binary format ***
            // int: _windowSize
            // int: _initialWindowSize
            // int (string ID): _inputColumnName
            // int (string ID): _outputColumnName
            // ColumnType: _transform.Schema.GetColumnType(0)

            ctx.Writer.Write(WindowSize);
            ctx.Writer.Write(InitialWindowSize);
            ctx.SaveNonEmptyString(InputColumnName);
            ctx.SaveNonEmptyString(OutputColumnName);

            var bs = new BinarySaver(Host, new BinarySaver.Arguments());
            bs.TryWriteTypeDescription(ctx.Writer.BaseStream, OutputColumnType, out int byteWritten);
        }

        public abstract Schema GetOutputSchema(Schema inputSchema);

        protected abstract IRowMapper MakeRowMapper(ISchema schema);

        protected SequentialDataTransform MakeDataTransform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return new SequentialDataTransform(Host, this, input, MakeRowMapper(input.Schema));
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            return new TimeSeriesRowToRowMapperTransform(Host, new EmptyDataView(Host, inputSchema), MakeRowMapper(inputSchema));
        }

        public sealed class SequentialDataTransform : TransformBase, ITransformTemplate, IRowToRowMapper
        {
            private readonly IRowMapper _mapper;
            private readonly SequentialTransformerBase<TInput, TOutput, TState> _parent;
            private readonly IDataTransform _transform;
            private readonly ColumnBindings _bindings;

            private MetadataDispatcher Metadata { get; }

            public SequentialDataTransform(IHost host, SequentialTransformerBase<TInput, TOutput, TState> parent,
                IDataView input, IRowMapper mapper)
                : base(parent.Host, input)
            {
                Metadata = new MetadataDispatcher(1);
                _parent = parent;
                _transform = CreateLambdaTransform(_parent.Host, input, _parent.InputColumnName,
                    _parent.OutputColumnName, InitFunction, _parent.WindowSize > 0, _parent.OutputColumnType);
                _mapper = mapper;
                _bindings = new ColumnBindings(Schema.Create(input.Schema), _mapper.GetOutputColumns());
            }

            private static IDataTransform CreateLambdaTransform(IHost host, IDataView input, string inputColumnName, string outputColumnName,
    Action<TState> initFunction, bool hasBuffer, ColumnType outputColTypeOverride)
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

            protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
            {
                var srcCursor = _transform.GetRowCursor(predicate, rand);
                return new Cursor(Host, this, srcCursor);
            }

            protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
            {
                Host.AssertValue(predicate);
                return false;
            }

            public override Schema Schema => _bindings.Schema;

            public override long? GetRowCount(bool lazy = true)
            {
                return _transform.GetRowCount(lazy);
            }

            public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
            {
                consolidator = null;
                return new IRowCursor[] { GetRowCursorCore(predicate, rand) };
            }

            public override void Save(ModelSaveContext ctx)
            {
                _parent.Save(ctx);
            }

            public IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
            {
                return new SequentialDataTransform(Contracts.CheckRef(env, nameof(env)).Register("SequentialDataTransform"), _parent, newSource, _mapper);
            }

            public Schema InputSchema => Source.Schema;
            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < Schema.ColumnCount; i++)
                {
                    if (predicate(i))
                        return col => true;
                }
                return col => false;
            }

            public IRow GetRow(IRow input, Func<int, bool> active, out Action disposer) =>
                new Row(_bindings.Schema, input, _mapper.CreateGetters(input, active, out disposer));

        }

        public sealed class Row : IStatefulRow
        {
            private readonly Schema _schema;
            private readonly IRow _input;
            private readonly Delegate[] _getters;

            public Schema Schema { get { return _schema; } }

            public long Position { get { return _input.Position; } }

            public long Batch { get { return _input.Batch; } }

            public Row(Schema schema, IRow input, Delegate[] getters)
            {
                Contracts.CheckValue(schema, nameof(schema));
                Contracts.CheckValue(input, nameof(input));
                Contracts.Check(Utils.Size(getters) == schema.ColumnCount);
                _schema = schema;
                _input = input;
                _getters = getters ?? new Delegate[0];
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return _input.GetIdGetter();
            }

            public ValueGetter<T> GetGetter<T>(int col)
            {
                Contracts.CheckParam(0 <= col && col < _getters.Length, nameof(col), "Invalid col value in GetGetter");
                Contracts.Check(IsColumnActive(col));
                var fn = _getters[col] as ValueGetter<T>;
                if (fn == null)
                    throw Contracts.Except("Unexpected TValue in GetGetter");
                return fn;
            }

            public bool IsColumnActive(int col)
            {
                Contracts.Check(0 <= col && col < _getters.Length);
                return _getters[col] != null;
            }
        }

        /// <summary>
        /// A wrapper around the cursor which replaces the schema.
        /// </summary>
        private sealed class Cursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly SequentialDataTransform _parent;

            public Cursor(IHost host, SequentialDataTransform parent, IRowCursor input)
                : base(host, input)
            {
                Ch.Assert(input.Schema.ColumnCount == parent.Schema.ColumnCount);
                _parent = parent;
            }

            public Schema Schema { get { return _parent.Schema; } }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < Schema.ColumnCount, "col");
                return Input.IsColumnActive(col);
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col), "col");
                return Input.GetGetter<TValue>(col);
            }
        }
    }

    /// <summary>
    /// This class is a transform that can add any number of output columns, that depend on any number of input columns.
    /// It does so with the help of an <see cref="IRowMapper"/>, that is given a schema in its constructor, and has methods
    /// to get the dependencies on input columns and the getters for the output columns, given an active set of output columns.
    /// </summary>

    public class TimeSeriesRowToRowMapperTransform : RowToRowTransformBase, IRowToRowMapper,
        ITransformCanSaveOnnx, ITransformCanSavePfa
    {
        private readonly IRowMapper _mapper;
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

        public override Schema Schema => _bindings.Schema;
        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => _mapper is ICanSaveOnnx onnxMapper ? onnxMapper.CanSaveOnnx(ctx) : false;

        bool ICanSavePfa.CanSavePfa => _mapper is ICanSavePfa pfaMapper ? pfaMapper.CanSavePfa : false;

        public TimeSeriesRowToRowMapperTransform(IHostEnvironment env, IDataView input, IRowMapper mapper)
            : base(env, RegistrationName, input)
        {
            Contracts.CheckValue(mapper, nameof(mapper));
            _mapper = mapper;
            _bindings = new ColumnBindings(Schema.Create(input.Schema), mapper.GetOutputColumns());
        }

        public static Schema GetOutputSchema(ISchema inputSchema, IRowMapper mapper)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.CheckValue(mapper, nameof(mapper));
            return new ColumnBindings(Schema.Create(inputSchema), mapper.GetOutputColumns()).Schema;
        }

        private TimeSeriesRowToRowMapperTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            // *** Binary format ***
            // _mapper

            ctx.LoadModel<IRowMapper, SignatureLoadRowMapper>(host, out _mapper, "Mapper", input.Schema);
            _bindings = new ColumnBindings(Schema.Create(input.Schema), _mapper.GetOutputColumns());
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

        public override void Save(ModelSaveContext ctx)
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
            int n = _bindings.Schema.ColumnCount;
            var active = Utils.BuildArray(n, predicate);
            Contracts.Assert(active.Length == n);

            var activeInput = _bindings.GetActiveInput(predicate);
            Contracts.Assert(activeInput.Length == _bindings.InputSchema.ColumnCount);

            // Get a predicate that determines which outputs are active.
            var predicateOut = GetActiveOutputColumns(active);

            // Now map those to active input columns.
            var predicateIn = _mapper.GetDependencies(predicateOut);

            // Combine the two sets of input columns.
            predicateInput =
                col => 0 <= col && col < activeInput.Length && (activeInput[col] || predicateIn(col));

            return active;
        }

        private Func<int, bool> GetActiveOutputColumns(bool[] active)
        {
            Contracts.AssertValue(active);
            Contracts.Assert(active.Length == _bindings.Schema.ColumnCount);

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

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Func<int, bool> predicateInput;
            var active = GetActive(predicate, out predicateInput);
            return new RowCursor(Host, Source.GetRowCursor(predicateInput, rand), this, active);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            Func<int, bool> predicateInput;
            var active = GetActive(predicate, out predicateInput);

            var inputs = Source.GetRowCursorSet(out consolidator, predicateInput, n, rand);
            Host.AssertNonEmpty(inputs);

            if (inputs.Length == 1 && n > 1 && _bindings.AddedColumnIndices.Any(predicate))
                inputs = DataViewUtils.CreateSplitCursors(out consolidator, Host, inputs[0], n);
            Host.AssertNonEmpty(inputs);

            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new RowCursor(Host, inputs[i], this, active);
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

        public Func<int, bool> GetDependencies(Func<int, bool> predicate)
        {
            Func<int, bool> predicateInput;
            GetActive(predicate, out predicateInput);
            return predicateInput;
        }

        Schema IRowToRowMapper.InputSchema => Source.Schema;

        public virtual IRow GetRow(IRow input, Func<int, bool> active, out Action disposer)
        {
            Host.CheckValue(input, nameof(input));
            Host.CheckValue(active, nameof(active));
            Host.Check(input.Schema == Source.Schema, "Schema of input row must be the same as the schema the mapper is bound to");

            disposer = null;
            using (var ch = Host.Start("GetEntireRow"))
            {
                Action disp;
                var activeArr = new bool[Schema.ColumnCount];
                for (int i = 0; i < Schema.ColumnCount; i++)
                    activeArr[i] = active(i);
                var pred = GetActiveOutputColumns(activeArr);
                var getters = _mapper.CreateGetters(input, pred, out disp);
                disposer += disp;
                return new StatefulRow(input, this, Schema, getters);
            }
        }

        private sealed class StatefulRow : IStatefulRow
        {
            private readonly IRow _input;
            private readonly Delegate[] _getters;

            private readonly TimeSeriesRowToRowMapperTransform _parent;

            public long Batch { get { return _input.Batch; } }

            public long Position { get { return _input.Position; } }

            public Schema Schema { get; }

            public StatefulRow(IRow input, TimeSeriesRowToRowMapperTransform parent, Schema schema, Delegate[] getters)
            {
                _input = input;
                _parent = parent;
                Schema = schema;
                _getters = getters;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                bool isSrc;
                int index = _parent._bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return _input.GetGetter<TValue>(index);

                Contracts.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Contracts.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            public ValueGetter<UInt128> GetIdGetter() => _input.GetIdGetter();

            public bool IsColumnActive(int col)
            {
                bool isSrc;
                int index = _parent._bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return _input.IsColumnActive((index));
                return _getters[index] != null;
            }
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Delegate[] _getters;
            private readonly bool[] _active;
            private readonly ColumnBindings _bindings;
            private readonly Action _disposer;

            public Schema Schema => _bindings.Schema;

            public RowCursor(IChannelProvider provider, IRowCursor input, TimeSeriesRowToRowMapperTransform parent, bool[] active)
                : base(provider, input)
            {
                var pred = parent.GetActiveOutputColumns(active);
                _getters = parent._mapper.CreateGetters(input, pred, out _disposer);
                _active = active;
                _bindings = parent._bindings;
            }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.Schema.ColumnCount);
                return _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return Input.GetGetter<TValue>(index);

                Ch.AssertValue(_getters);
                var getter = _getters[index];
                Ch.Assert(getter != null);
                var fn = getter as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            public override void Dispose()
            {
                _disposer?.Invoke();
                base.Dispose();
            }
        }
    }

}
