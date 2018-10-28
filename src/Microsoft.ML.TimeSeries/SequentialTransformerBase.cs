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
            protected IHost Host;

            /// <summary>
            /// A reference to the parent transform that operates on the state object.
            /// </summary>
            protected SequentialTransformerBase<TInput, TOutput, TState> ParentTransform;

            /// <summary>
            /// The internal windowed buffer for buffering the values in the input sequence.
            /// </summary>
            protected FixedSizeQueue<TInput> WindowedBuffer;

            /// <summary>
            /// The buffer used to buffer the training data points.
            /// </summary>
            protected FixedSizeQueue<TInput> InitialWindowedBuffer;

            protected int WindowSize { get; private set; }

            protected int InitialWindowSize { get; private set; }

            /// <summary>
            /// Counts the number of rows observed by the transform so far.
            /// </summary>
            protected long RowCounter { get; private set; }

            protected long IncrementRowCounter()
            {
                RowCounter++;
                return RowCounter;
            }

            private bool _isIniatilized;

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
            }

            public void Process(ref TInput input, ref TOutput output)
            {
                if (InitialWindowedBuffer.Count < InitialWindowSize)
                {
                    InitialWindowedBuffer.AddLast(input);
                    SetNaOutput(ref output);

                    if (InitialWindowedBuffer.Count >= InitialWindowSize - WindowSize)
                        WindowedBuffer.AddLast(input);

                    if (InitialWindowedBuffer.Count == InitialWindowSize)
                        LearnStateFromDataCore(InitialWindowedBuffer);
                }
                else
                {
                    TransformCore(ref input, WindowedBuffer, RowCounter - InitialWindowSize, ref output);
                    WindowedBuffer.AddLast(input);
                    IncrementRowCounter();
                }
            }

            public void ProcessWithoutBuffer(ref TInput input, ref TOutput output)
            {
                if (InitialWindowedBuffer.Count < InitialWindowSize)
                {
                    InitialWindowedBuffer.AddLast(input);
                    SetNaOutput(ref output);

                    if (InitialWindowedBuffer.Count == InitialWindowSize)
                        LearnStateFromDataCore(InitialWindowedBuffer);
                }
                else
                {
                    TransformCore(ref input, WindowedBuffer, RowCounter - InitialWindowSize, ref output);
                    IncrementRowCounter();
                }
            }

            /// <summary>
            /// The abstract method that specifies the NA value for the dst type.
            /// </summary>
            /// <returns></returns>
            protected abstract void SetNaOutput(ref TOutput dst);

            /// <summary>
            /// The abstract method that realizes the main logic for the transform.
            /// </summary>
            /// <param name="input">A reference to the input object.</param>
            /// <param name="dst">A reference to the dst object.</param>
            /// <param name="windowedBuffer">A reference to the windowed buffer.</param>
            /// <param name="iteration">A long number that indicates the number of times TransformCore has been called so far (starting value = 0).</param>
            protected abstract void TransformCore(ref TInput input, FixedSizeQueue<TInput> windowedBuffer, long iteration, ref TOutput dst);

            /// <summary>
            /// The abstract method that realizes the logic for initializing the state object.
            /// </summary>
            protected abstract void InitializeStateCore();

            /// <summary>
            /// The abstract method that realizes the logic for learning the parameters and the initial state object from data.
            /// </summary>
            /// <param name="data">A queue of data points used for training</param>
            protected abstract void LearnStateFromDataCore(FixedSizeQueue<TInput> data);
        }

        protected readonly IHost Host;

        /// <summary>
        /// The window size for buffering.
        /// </summary>
        protected readonly int WindowSize;

        /// <summary>
        /// The number of datapoints from the beginning of the sequence that are used for learning the initial state.
        /// </summary>
        protected int InitialWindowSize;

        public string InputColumnName;
        public string OutputColumnName;
        protected ColumnType OutputColumnType;

        public bool IsRowToRowMapper => false;

        /// <summary>
        /// The main constructor for the sequential transform
        /// </summary>
        /// <param name="windowSize">The size of buffer used for windowed buffering.</param>
        /// <param name="initialWindowSize">The number of datapoints picked from the beginning of the series for training the transform parameters if needed.</param>
        /// <param name="inputColumnName">The name of the input column.</param>
        /// <param name="outputColumnName">The name of the dst column.</param>
        /// <param name="name"></param>
        /// <param name="env">A reference to the environment variable.</param>
        /// <param name="outputColType"></param>
        protected SequentialTransformerBase(int windowSize, int initialWindowSize, string inputColumnName, string outputColumnName,
            string name, IHostEnvironment env, ColumnType outputColType)
        {
            Contracts.CheckRef(env, nameof(env));
            Host = env.Register(name);
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

        protected SequentialTransformerBase(IHostEnvironment env, ModelLoadContext ctx, string name)
        {
            Contracts.CheckRef(env, nameof(env));
            Host = env.Register(name);
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

        protected RowToRowMapperTransform MakeDataTransform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return new RowToRowMapperTransform(Host, input, MakeRowMapper(input.Schema));
        }

        public IDataView Transform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return new DataView(this, input);
        }

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            throw new NotImplementedException("Not a RowToRowMapper.");
        }

        private sealed class DataView : IDataView
        {
            private readonly SequentialTransformerBase<TInput, TOutput, TState> _parent;
            private readonly IDataTransform _transform;

            public DataView(SequentialTransformerBase<TInput, TOutput, TState> parent, IDataView input)
            {
                _parent = parent;
                _transform = CreateLambdaTransform(_parent.Host, input, _parent.InputColumnName,
                    _parent.OutputColumnName, InitFunction, _parent.WindowSize > 0, _parent.OutputColumnType);
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

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null) => GetRowCursorCore(predicate, null);

            public bool CanShuffle { get { return false; } }

            public IHost Host { get { return _parent.Host; } }

            private IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
            {
                var srcCursor = _transform.GetRowCursor(predicate, rand);
                return new Cursor(this, srcCursor);
            }

            public Schema Schema
            {
                get { return _transform.Schema; }
            }

            public long? GetRowCount(bool lazy = true)
            {
                return _transform.GetRowCount(lazy);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
            {
                consolidator = null;
                return new IRowCursor[] { GetRowCursorCore(predicate, rand) };
            }
        }

        /// <summary>
        /// A wrapper around the cursor which replaces the schema.
        /// </summary>
        private sealed class Cursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly DataView _parent;

            public Cursor(DataView parent, IRowCursor input)
                : base(parent.Host, input)
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
}
