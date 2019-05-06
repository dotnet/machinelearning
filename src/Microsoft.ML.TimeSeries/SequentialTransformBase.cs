// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// The box class that is used to box the TInput and TOutput for the LambdaTransform.
    /// </summary>
    /// <typeparam name="T">The type to be boxed, e.g. TInput or TOutput</typeparam>
    internal sealed class DataBox<T>
    {
        public T Value;

        public DataBox()
        {
        }

        public DataBox(T value)
        {
            Value = value;
        }
    }

    /// <summary>
    /// The base class for sequential processing transforms. This class implements the basic sliding window buffering. The derived classes need to specify the transform logic,
    /// the initialization logic and the learning logic via implementing the abstract methods TransformCore(), InitializeStateCore() and LearnStateFromDataCore(), respectively
    /// </summary>
    /// <typeparam name="TInput">The input type of the sequential processing.</typeparam>
    /// <typeparam name="TOutput">The dst type of the sequential processing.</typeparam>
    /// <typeparam name="TState">The state type of the sequential processing. Must be a class inherited from StateBase </typeparam>
    internal abstract class SequentialTransformBase<TInput, TOutput, TState> : TransformBase
       where TState : SequentialTransformBase<TInput, TOutput, TState>.StateBase, new()
    {
        /// <summary>
        /// The base class for encapsulating the State object for sequential processing. This class implements a windowed buffer.
        /// </summary>
        public abstract class StateBase
        {
            // Ideally this class should be private. However, due to the current constraints with the LambdaTransform, we need to have
            // access to the state class when inheriting from SequentialTransformBase.
            protected IHost Host;

            /// <summary>
            /// A reference to the parent transform that operates on the state object.
            /// </summary>
            private protected SequentialTransformBase<TInput, TOutput, TState> ParentTransform;

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
            private protected int RowCounter { get; private set; }

            private protected int IncrementRowCounter()
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
            public void InitState(int windowSize, int initialWindowSize, SequentialTransformBase<TInput, TOutput, TState> parentTransform, IHost host)
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

            protected StateBase()
            {
                // Default constructor is required by the LambdaTransform.
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
            /// The abstract method that specifies the NA value for <paramref name="dst"/>'s type.
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

        /// <summary>
        /// The inner stateful Lambda Transform object.
        /// </summary>
        private readonly IDataTransform _transform;

        /// <summary>
        /// The window size for buffering.
        /// </summary>
        protected readonly int WindowSize;

        /// <summary>
        /// The number of datapoints from the beginning of the sequence that are used for learning the initial state.
        /// </summary>
        protected int InitialWindowSize;

        protected string InputColumnName;
        protected string OutputColumnName;

        private static IDataTransform CreateLambdaTransform(IHost host, IDataView input, string outputColumnName, string inputColumnName,
            Action<TState> initFunction, bool hasBuffer, DataViewType outputColTypeOverride)
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

        /// <summary>
        /// The main constructor for the sequential transform
        /// </summary>
        /// <param name="windowSize">The size of buffer used for windowed buffering.</param>
        /// <param name="initialWindowSize">The number of datapoints picked from the beginning of the series for training the transform parameters if needed.</param>
        /// <param name="outputColumnName">The name of the dst column.</param>
        /// <param name="inputColumnName">The name of the input column.</param>
        /// <param name="name">Name of the extending type.</param>
        /// <param name="env">A reference to the environment variable.</param>
        /// <param name="input">A reference to the input data view.</param>
        /// <param name="outputColTypeOverride"></param>
        private protected SequentialTransformBase(int windowSize, int initialWindowSize, string outputColumnName, string inputColumnName,
            string name, IHostEnvironment env, IDataView input, DataViewType outputColTypeOverride = null)
            : this(windowSize, initialWindowSize, outputColumnName, inputColumnName, Contracts.CheckRef(env, nameof(env)).Register(name), input, outputColTypeOverride)
        {
        }

        private protected SequentialTransformBase(int windowSize, int initialWindowSize, string outputColumnName, string inputColumnName,
            IHost host, IDataView input, DataViewType outputColTypeOverride = null)
            : base(host, input)
        {
            Contracts.AssertValue(Host);
            Host.CheckParam(initialWindowSize >= 0, nameof(initialWindowSize), "Must be non-negative.");
            Host.CheckParam(windowSize >= 0, nameof(windowSize), "Must be non-negative.");
            // REVIEW: Very bad design. This base class is responsible for reporting errors on
            // the arguments, but the arguments themselves are not derived form any base class.
            Host.CheckNonEmpty(inputColumnName, nameof(PercentileThresholdTransform.Arguments.Source));
            Host.CheckNonEmpty(outputColumnName, nameof(PercentileThresholdTransform.Arguments.Source));

            InputColumnName = inputColumnName;
            OutputColumnName = outputColumnName;
            InitialWindowSize = initialWindowSize;
            WindowSize = windowSize;

            _transform = CreateLambdaTransform(Host, input, OutputColumnName, InputColumnName, InitFunction, WindowSize > 0, outputColTypeOverride);
        }

        private protected SequentialTransformBase(IHostEnvironment env, ModelLoadContext ctx, string name, IDataView input)
            : base(env, name, input)
        {
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
            DataViewType ct = bs.LoadTypeDescriptionOrNull(ctx.Reader.BaseStream);

            _transform = CreateLambdaTransform(Host, input, OutputColumnName, InputColumnName, InitFunction, WindowSize > 0, ct);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
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

            int byteWritten;
            BinarySaver bs = new BinarySaver(Host, new BinarySaver.Arguments());

            int colIndex;
            if (!_transform.Schema.TryGetColumnIndex(OutputColumnName, out colIndex))
                throw Host.ExceptSchemaMismatch(nameof(_transform.Schema), "output", OutputColumnName);

            bs.TryWriteTypeDescription(ctx.Writer.BaseStream, _transform.Schema[colIndex].Type, out byteWritten);
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
            state.InitState(WindowSize, InitialWindowSize, this, Host);
        }

        public override bool CanShuffle => false;

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);
            return false;
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            var srcCursor = _transform.GetRowCursor(columnsNeeded, rand);
            return new Cursor(this, srcCursor);
        }

        public override DataViewSchema OutputSchema => _transform.Schema;

        public override long? GetRowCount()
        {
            return _transform.GetRowCount();
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            => new DataViewRowCursor[] { GetRowCursorCore(columnsNeeded, rand) };

        /// <summary>
        /// A wrapper around the cursor which replaces the schema.
        /// </summary>
        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly SequentialTransformBase<TInput, TOutput, TState> _parent;

            public Cursor(SequentialTransformBase<TInput, TOutput, TState> parent, DataViewRowCursor input)
                : base(parent.Host, input)
            {
                Ch.Assert(input.Schema.Count == parent.OutputSchema.Count);
                _parent = parent;
            }

            public override DataViewSchema Schema { get { return _parent.OutputSchema; } }

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
}
