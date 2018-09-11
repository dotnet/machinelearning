// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TensorFlow;

[assembly: LoadableClass(TensorFlowTransform.Summary, typeof(IDataTransform), typeof(TensorFlowTransform),
    typeof(TensorFlowTransform.Arguments), typeof(SignatureDataTransform), TensorFlowTransform.UserName, TensorFlowTransform.ShortName)]

[assembly: LoadableClass(TensorFlowTransform.Summary, typeof(IDataTransform), typeof(TensorFlowTransform), null, typeof(SignatureLoadDataTransform),
    TensorFlowTransform.UserName, TensorFlowTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(TensorFlowTransform), null, typeof(SignatureLoadModel),
    TensorFlowTransform.UserName, TensorFlowTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TensorFlowTransform), null, typeof(SignatureLoadRowMapper),
    TensorFlowTransform.UserName, TensorFlowTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(TensorFlowTransform))]

namespace Microsoft.ML.Transforms
{
    public sealed class TensorFlowTransform : ITransformer, ICanSaveModel
    {
        public sealed class Arguments : TransformInputBase
        {

            [Argument(ArgumentType.Required, HelpText = "This is the frozen protobuf model file. Please see https://www.tensorflow.org/mobile/prepare_models for more details.", ShortName = "model", SortOrder = 0)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the outputs", ShortName = "outputs", SortOrder = 2)]
            public string[] OutputColumns;
        }

        private readonly IHost _host;
        private const string RegistrationName = "TensorFlowTransform";

        internal readonly TFSession Session;
        internal readonly ColumnType[] OutputTypes;
        internal readonly TFDataType[] TFOutputTypes;
        internal readonly TFDataType[] TFInputTypes;
        internal readonly TFShape[] TFInputShapes;
        internal TFGraph Graph => Session.Graph;

        public readonly string[] Inputs;
        public readonly string[] Outputs;

        public static int BatchSize = 1;
        public const string Summary = "Transforms the data using the TensorFlow model.";
        public const string UserName = "TensorFlowTransform";
        public const string ShortName = "TFTransform";
        public const string LoaderSignature = "TensorFlowTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TENSFLOW",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002,  // Upgraded when change for multiple outputs was implemented.
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="modelFile">This is the frozen TensorFlow model file. https://www.tensorflow.org/mobile/prepare_models </param>
        /// <param name="name">Name of the output column. Keep it same as in the TensorFlow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the TensorFlow model.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string modelFile, string name, params string[] source)
        {
            return new TensorFlowTransform(env, modelFile, source, new[] { name }).MakeDataTransform(input);
        }

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="modelFile">This is the frozen tensorflow model file. https://www.tensorflow.org/mobile/prepare_models </param>
        /// <param name="names">Name of the output column(s). Keep it same as in the Tensorflow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the Tensorflow model.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string modelFile, string[] names, string[] source)
        {
            return new TensorFlowTransform(env, modelFile, source, names).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        public static TensorFlowTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            // *** Binary format ***
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            byte[] modelBytes = null;
            if (!ctx.TryLoadBinaryStream("TFModel", r => modelBytes = r.ReadByteArray()))
                throw env.ExceptDecode();
            var session = TensorFlowUtils.LoadTFSession(env, modelBytes);
            var numInputs = ctx.Reader.ReadInt32();
            env.CheckDecode(numInputs > 0);
            string[] inputs = new string[numInputs];
            for (int j = 0; j < inputs.Length; j++)
                inputs[j] = ctx.LoadNonEmptyString();

            bool isMultiOutput = ctx.Header.ModelVerReadable >= 0x00010002;
            var numOutputs = 1;
            if (isMultiOutput)
                numOutputs = ctx.Reader.ReadInt32();

            env.CheckDecode(numOutputs > 0);
            var outputs = new string[numOutputs];
            for (int j = 0; j < outputs.Length; j++)
                outputs[j] = ctx.LoadNonEmptyString();

            return new TensorFlowTransform(env, session, inputs, outputs);
        }

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));
            env.CheckValue(args.InputColumns, nameof(args.InputColumns));
            env.CheckValue(args.OutputColumns, nameof(args.OutputColumns));
            return new TensorFlowTransform(env, args.ModelFile, args.InputColumns, args.OutputColumns).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        public static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        public static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static TFSession CheckFileAndRead(IHostEnvironment env, string modelFile)
        {
            env.CheckNonWhiteSpace(modelFile, nameof(modelFile));
            env.CheckUserArg(File.Exists(modelFile), nameof(modelFile));
            var bytes = File.ReadAllBytes(modelFile);
            return TensorFlowUtils.LoadTFSession(env, bytes, modelFile);
        }

        public TensorFlowTransform(IHostEnvironment env, string modelFile, string[] inputs, string[] outputs) :
            this(env, CheckFileAndRead(env, modelFile), inputs, outputs, modelFile)
        {
        }

        private TensorFlowTransform(IHostEnvironment env, TFSession session, string[] inputs, string[] outputs, string modelFile = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(RegistrationName));
            _host.CheckValue(session, nameof(session));
            Session = session;
            foreach (var input in inputs)
            {
                _host.CheckNonWhiteSpace(input, nameof(inputs));
                if (Session.Graph[input] == null)
                    throw _host.ExceptParam(nameof(inputs), $"Input column '{input}' does not exist in the model");
                var tfInput = new TFOutput(Session.Graph[input]);
                if (!TensorFlowUtils.IsTypeSupported(tfInput.OutputType))
                    throw _host.ExceptParam(nameof(session), $"Input type '{tfInput.OutputType}' of input column '{input}' is not supported in TensorFlow");
            }

            var newNames = new HashSet<string>();
            foreach (var output in outputs)
            {
                _host.CheckNonEmpty(output, nameof(outputs));
                if (!newNames.Add(output))
                    throw _host.ExceptParam(nameof(outputs), $"Output column '{output}' specified multiple times");
                if (Session.Graph[output] == null)
                    throw _host.ExceptParam(nameof(outputs), $"Output column '{output}' does not exist in the model");
            }

            Inputs = inputs;
            TFInputTypes = new TFDataType[Inputs.Length];
            TFInputShapes = new TFShape[Inputs.Length];
            for (int i = 0; i < Inputs.Length; i++)
            {
                var tfInput = new TFOutput(Graph[Inputs[i]]);
                TFInputTypes[i] = tfInput.OutputType;
                TFInputShapes[i] = Graph.GetTensorShape(tfInput);
                var newShape = new long[TFInputShapes[i].NumDimensions];
                for (int j = 0; j < TFInputShapes[i].NumDimensions; j++)
                    newShape[j] = TFInputShapes[i][j] == -1 ? BatchSize : TFInputShapes[i][j];
                TFInputShapes[i] = new TFShape(newShape);
            }

            Outputs = outputs;
            OutputTypes = new ColumnType[Outputs.Length];
            TFOutputTypes = new TFDataType[Outputs.Length];
            for (int i = 0; i < Outputs.Length; i++)
            {
                var tfOutput = new TFOutput(Graph[Outputs[i]]);
                var shape = Graph.GetTensorShape(tfOutput);
                int[] dims = shape.ToIntArray().Skip(shape[0] == -1 ? BatchSize : 0).ToArray();
                var type = TensorFlowUtils.Tf2MlNetType(tfOutput.OutputType);
                OutputTypes[i] = new VectorType(type, dims);
                TFOutputTypes[i] = tfOutput.OutputType;
            }
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            foreach (var input in Inputs)
            {
                if (!inputSchema.TryGetColumnIndex(input, out int srcCol))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
            }
            return Transform(new EmptyDataView(_host, inputSchema)).Schema;
        }

        private IRowMapper MakeRowMapper(ISchema schema) => new Mapper(_host, this, schema);

        private RowToRowMapperTransform MakeDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            return new RowToRowMapperTransform(_host, input, MakeRowMapper(input.Schema));
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        public void Save(ModelSaveContext ctx)
        {
            _host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            // *** Binary format ***
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name

            var buffer = new TFBuffer();
            Session.Graph.ToGraphDef(buffer);

            ctx.SaveBinaryStream("TFModel", w =>
            {
                w.WriteByteArray(buffer.ToArray());
            });
            _host.AssertNonEmpty(Inputs);
            ctx.Writer.Write(Inputs.Length);
            foreach (var colName in Inputs)
                ctx.SaveNonEmptyString(colName);

            _host.AssertNonEmpty(Outputs);
            ctx.Writer.Write(Outputs.Length);
            foreach (var colName in Outputs)
                ctx.SaveNonEmptyString(colName);
        }

        private sealed class Mapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly ISchema _schema;
            private readonly TensorFlowTransform _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;

            public Mapper(IHostEnvironment env, TensorFlowTransform parent, ISchema inputSchema)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(Mapper));
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _schema = inputSchema;
                _inputColIndices = new int[_parent.Inputs.Length];
                _isInputVector = new bool[_parent.Inputs.Length];
                for (int i = 0; i < _parent.Inputs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.Inputs[i], out _inputColIndices[i]))
                        throw _host.Except($"Column {_parent.Inputs[i]} doesn't exist");

                    var type = inputSchema.GetColumnType(_inputColIndices[i]);
                    var expectedType = TensorFlowUtils.Tf2MlNetType(_parent.TFInputTypes[i]);
                    if (type.ItemType != expectedType)
                        throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i], expectedType.ToString(), type.ToString());
                    var originalShape = _parent.Graph.GetTensorShape(new TFOutput(_parent.Graph[_parent.Inputs[i]]));
                    var shape = originalShape.ToIntArray().Skip(originalShape[0] == -1 ? BatchSize : 0);
                    _isInputVector[i] = type.IsVector;
                    if (type.AsVector.DimCount == 1)
                    {
                        int valCount = shape.Aggregate((x, y) => x * y);
                        if (type.ValueCount != valCount)
                            throw _host.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {shape.ToString()}, but input data is of length {valCount}.");
                    }
                    else if (shape.Select((dim, j) => dim != type.AsVector.GetDim(j)).Any(b => b))
                        throw _host.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {shape.ToString()}, but input data is {type.AsVector.ToString()}.");
                }
            }

            public void Save(ModelSaveContext ctx)
            {
                _parent.Save(ctx);
            }

            private ITensorValueGetter CreateTensorValueGetter<T>(IRow input, bool isVector, int colIndex, TFShape tfShape)
            {
                if (isVector)
                    return new TensorValueGetterVec<T>(input, colIndex, tfShape);
                else
                    return new TensorValueGetter<T>(input, colIndex);
            }

            private ITensorValueGetter CreateTensorValueGetter(IRow input, TFDataType tfType, bool isVector, int colIndex, TFShape tfShape)
            {
                var type = TFTensor.TypeFromTensorType(tfType);
                _host.AssertValue(type);
                return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type, input, isVector, colIndex, tfShape);
            }

            private ITensorValueGetter[] GetTensorValueGetters(IRow input)
            {
                _host.Assert(input.Schema == _schema);
                var srcTensorGetters = new ITensorValueGetter[_inputColIndices.Length];
                for (int i = 0; i < _inputColIndices.Length; i++)
                {
                    int colIndex = _inputColIndices[i];
                    srcTensorGetters[i] = CreateTensorValueGetter(input, _parent.TFInputTypes[i], _isInputVector[i], colIndex, _parent.TFInputShapes[i]);
                }
                return srcTensorGetters;
            }

            private class OutputCache
            {
                public long Position;
                public Dictionary<string, TFTensor> Outputs;
                public OutputCache()
                {
                    Position = -1;
                    Outputs = new Dictionary<string, TFTensor>();
                }
            }

            private Delegate[] MakeGetters(IRow input, Func<int, bool> activeOutput)
            {
                _host.AssertValue(input);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();

                var valueGetters = new List<Delegate>();
                for (int i = 0; i < _parent.Outputs.Length; i++)
                {
                    if (activeOutput(i))
                    {
                        var type = TFTensor.TypeFromTensorType(_parent.TFOutputTypes[i]);
                        _host.Assert(type == _parent.OutputTypes[i].ItemType.RawType);
                        var srcTensorGetters = GetTensorValueGetters(input);
                        valueGetters.Add(Utils.MarshalInvoke(MakeGetter<int>, type, input, i, srcTensorGetters, activeOutputColNames, outputCache));
                    }
                }
                return valueGetters.ToArray();
            }

            private Delegate MakeGetter<T>(IRow input, int iinfo, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                _host.AssertValue(input);
                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                    var values = dst.Values;
                    var indices = dst.Indices;
                    if (Utils.Size(values) < _parent.OutputTypes[iinfo].VectorSize)
                    {
                        values = new T[_parent.OutputTypes[iinfo].VectorSize];
                        indices = new int[_parent.OutputTypes[iinfo].VectorSize];
                    }

                    TensorFlowUtils.FetchData<T>(outputCache.Outputs[_parent.Outputs[iinfo]].Data, values);
                    dst = new VBuffer<T>(values.Length, values, indices);
                };
                return valuegetter;
            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    var runner = _parent.Session.GetRunner();
                    for (int i = 0; i < _inputColIndices.Length; i++)
                    {
                        var inputName = _parent.Inputs[i];
                        runner.AddInput(inputName, srcTensorGetters[i].GetTensor());
                    }

                    var tensors = runner.Fetch(activeOutputColNames).Run();
                    Contracts.Assert(tensors.Length > 0);

                    for (int j = 0; j < tensors.Length; j++)
                        outputCache.Outputs[activeOutputColNames[j]] = tensors[j];

                    outputCache.Position = position;
                }
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                using (var ch = _host.Start("CreateGetters"))
                {
                    var getters = MakeGetters(input, activeOutput);
                    ch.Done();
                    return getters;
                }
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                var info = new RowMapperColumnInfo[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                    info[i] = new RowMapperColumnInfo(_parent.Outputs[i], _parent.OutputTypes[i], null);
                return info;
            }

            private interface ITensorValueGetter
            {
                TFTensor GetTensor();
            }

            private class TensorValueGetter<T> : ITensorValueGetter
            {
                private readonly ValueGetter<T> _srcgetter;

                public TensorValueGetter(IRow input, int colIndex)
                {
                    _srcgetter = input.GetGetter<T>(colIndex);
                }
                public TFTensor GetTensor()
                {
                    var scalar = default(T);
                    _srcgetter(ref scalar);
                    return TFTensor.CreateScalar(scalar);
                }
            }

            private class TensorValueGetterVec<T> : ITensorValueGetter
            {
                private readonly ValueGetter<VBuffer<T>> _srcgetter;
                private readonly TFShape _tfShape;
                private VBuffer<T> _vBuffer;
                private VBuffer<T> _vBufferDense;

                public TensorValueGetterVec(IRow input, int colIndex, TFShape tfShape)
                {
                    _srcgetter = input.GetGetter<VBuffer<T>>(colIndex);
                    _tfShape = tfShape;
                    _vBuffer = default;
                    _vBufferDense = default;
                }

                public TFTensor GetTensor()
                {
                    _srcgetter(ref _vBuffer);
                    _vBuffer.CopyToDense(ref _vBufferDense);
                    return TFTensor.Create(_vBufferDense.Values, _tfShape);
                }
            }
        }

        [TlcModule.EntryPoint(Name = "Transforms.TensorFlowScorer", Desc = Summary, UserName = UserName, ShortName = ShortName)]
        public static CommonOutputs.TransformOutput TensorFlowScorer(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "TensorFlow", input);
            var view = Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }

    public sealed class TensorFlowEstimator : TrivialEstimator<TensorFlowTransform>
    {
        public TensorFlowEstimator(IHostEnvironment env, string modelFile, string[] inputs, string[] outputs)
           : this(env, new TensorFlowTransform(env, modelFile, inputs, outputs))
        {
        }

        public TensorFlowEstimator(IHostEnvironment env, TensorFlowTransform transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TensorFlowTransform)), transformer)
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);
            for (var i = 0; i < Transformer.Inputs.Length; i++)
            {
                var input = Transformer.Inputs[i];
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                if (!(col.Kind == SchemaShape.Column.VectorKind.VariableVector || col.Kind == SchemaShape.Column.VectorKind.Vector))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, nameof(VectorType), col.GetTypeString());
                var expectedType = TensorFlowUtils.Tf2MlNetType(Transformer.TFInputTypes[i]);
                if (col.ItemType != expectedType)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, expectedType.ToString(), col.ItemType.ToString());
            }
            for (var i = 0; i < Transformer.Outputs.Length; i++)
                resultDic[Transformer.Outputs[i]] = new SchemaShape.Column(Transformer.Outputs[i], SchemaShape.Column.VectorKind.Vector, Transformer.OutputTypes[i].ItemType, false);
            return new SchemaShape(resultDic.Values);
        }
    }
}
