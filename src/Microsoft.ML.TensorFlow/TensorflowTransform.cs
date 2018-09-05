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

[assembly: LoadableClass(typeof(IRowMapper), typeof(TensorFlowTransform), null, typeof(SignatureLoadRowMapper),
    TensorFlowTransform.UserName, TensorFlowTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(TensorFlowTransform))]

namespace Microsoft.ML.Transforms
{
    public sealed class TensorFlowTransform : ITransformer, ICanSaveModel
    {
        public sealed class Arguments : TransformInputBase
        {

            [Argument(ArgumentType.Required, HelpText = "This is the frozen protobuf model file. Please see https://www.tensorflow.org/mobile/prepare_models for more details.", ShortName = "ModelDir", SortOrder = 0)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the output", ShortName = "output", SortOrder = 2)]
            public string[] OutputColumns;
        }

        private readonly IHost _host;
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
        private const string RegistrationName = "TensorFlowTransform";

        public const string LoaderSignature = "TensorFlowTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TENSFLOW",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
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
            env.CheckNonWhiteSpace(modelFile, nameof(modelFile));
            env.CheckUserArg(File.Exists(modelFile), nameof(modelFile));
            return new TensorFlowTransform(env, File.ReadAllBytes(modelFile), source, new[] { name }).MakeDataTransform(input);
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
            env.CheckNonWhiteSpace(modelFile, nameof(modelFile));
            env.CheckUserArg(File.Exists(modelFile), nameof(modelFile));
            return new TensorFlowTransform(env, File.ReadAllBytes(modelFile), source, names).MakeDataTransform(input);
        }

        public static TensorFlowTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return Create(host, ctx);
        }

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));
            env.CheckValue(args.InputColumns, nameof(args.InputColumns));
            env.CheckValue(args.OutputColumns, nameof(args.OutputColumns));
            env.CheckNonWhiteSpace(args.ModelFile, nameof(args.ModelFile));
            env.CheckUserArg(File.Exists(args.ModelFile), nameof(args.ModelFile));
            return new TensorFlowTransform(env, File.ReadAllBytes(args.ModelFile), args.InputColumns, args.OutputColumns).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        public static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        public static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private TFSession LoadTFSession(byte[] modelBytes, string modelArg = null)
        {
            var graph = new TFGraph();
            try
            {
                graph.Import(modelBytes, "");
            }
            catch (Exception ex)
            {
                if (!string.IsNullOrEmpty(modelArg))
                    throw _host.Except($"TensorFlow exception triggered while loading model from '{modelArg}'");
#pragma warning disable MSML_NoMessagesForLoadContext
                throw _host.ExceptDecode(ex, "Tensorflow exception triggered while loading model.");
#pragma warning restore MSML_NoMessagesForLoadContext

            }
            return new TFSession(graph);
        }

        internal TensorFlowTransform(IHostEnvironment env, byte[] modelStream, string[] inputs, string[] outputs)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TensorFlowTransform));
            _host.CheckValue(modelStream, nameof(modelStream));
            Session = LoadTFSession(modelStream);
            foreach (var input in inputs)
                _host.CheckNonWhiteSpace(input, nameof(inputs));
            _host.Check(inputs.All(name => Session.Graph[name] != null), "One of the input does not exist in the model");
            var newNames = new HashSet<string>();
            foreach (var output in outputs)
            {
                _host.CheckNonEmpty(output, nameof(outputs));
                if (!newNames.Add(output))
                    throw Contracts.ExceptParam(nameof(outputs), $"Output column '{output}' specified multiple times");
            }

            _host.Check(outputs.All(name => Session.Graph[name] != null), "One of the output does not exist in the model");
            Inputs = inputs;
            TFInputTypes = new TFDataType[Inputs.Length];
            TFInputShapes = new TFShape[Inputs.Length];
            for (int i = 0; i < Inputs.Length; i++)
            {
                var tfInput = new TFOutput(Graph[Inputs[i]]);
                TFInputTypes[i] = tfInput.OutputType;
                TFInputShapes[i] = Graph.GetTensorShape(tfInput);
                var l = new long[TFInputShapes[i].NumDimensions];
                for (int ishape = 0; ishape < TFInputShapes[i].NumDimensions; ishape++)
                {
                    l[ishape] = TFInputShapes[i][ishape] == -1 ? BatchSize : TFInputShapes[i][ishape];
                }
                TFInputShapes[i] = new TFShape(l);
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

        private IRowMapper MakeRowMapper(ISchema schema)
            => new TensorFlowMapper(_host, this, schema);

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

        private static TensorFlowTransform Create(IHost host, ModelLoadContext ctx)
        {
            byte[] modelStream = null;
            if (!ctx.TryLoadBinaryStream("TFModel", r => modelStream = r.ReadByteArray()))
                throw host.ExceptDecode();
            var numInputs = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(numInputs > 0);
            string[] inputs = new string[numInputs];
            for (int j = 0; j < inputs.Length; j++)
                inputs[j] = ctx.LoadNonEmptyString();

            var numOutputs = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(numOutputs > 0);
            string[] outputs = new string[numOutputs];
            for (int j = 0; j < outputs.Length; j++)
                outputs[j] = ctx.LoadNonEmptyString();

            return new TensorFlowTransform(host, modelStream, inputs, outputs);
        }

        private sealed class TensorFlowMapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly ISchema _schema;
            private readonly TensorFlowTransform _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private IDictionary<string, TFTensor> _cachedOutputs;
            private long _cachedPosition;

            public TensorFlowMapper(IHostEnvironment env, TensorFlowTransform parent, ISchema inputSchema)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(TensorFlowMapper));
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

                    var originalShape = _parent.Graph.GetTensorShape(new TFOutput(_parent.Graph[_parent.Inputs[i]]));
                    var shape = originalShape.ToIntArray().Skip(originalShape[0] == -1 ? BatchSize : 0);
                    _isInputVector[i] = type.IsVector;
                    if (type.AsVector.DimCount == 1)
                    {
                        int valCount = shape.Aggregate((x, y) => x * y);
                        if (type.ValueCount != valCount)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {shape.ToString()}, but input data is of length {valCount}.");
                    }
                    else if (shape.Select((dim, j) => dim != type.AsVector.GetDim(j)).Any(b => b))
                        throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {shape.ToString()}, but input data is {type.AsVector.ToString()}.");
                }

                _cachedOutputs = new Dictionary<string, TFTensor>();
                _cachedPosition = -1;
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
                Contracts.Assert(input.Schema == _schema);
                var srcTensorGetters = new ITensorValueGetter[_inputColIndices.Length];
                for (int i = 0; i < _inputColIndices.Length; i++)
                {
                    int colIndex = _inputColIndices[i];
                    srcTensorGetters[i] = CreateTensorValueGetter(input, _parent.TFInputTypes[i], _isInputVector[i], colIndex, _parent.TFInputShapes[i]);
                }
                return srcTensorGetters;
            }

            private Delegate MakeGetter(IRow input, int iinfo)
            {
                var type = TFTensor.TypeFromTensorType(_parent.TFOutputTypes[iinfo]);
                _host.Assert(type == _parent.OutputTypes[iinfo].ItemType.RawType);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, _parent.OutputTypes[iinfo], iinfo);
            }

            private Delegate MakeGetter<T>(IRow input, ColumnType columnType, int iinfo)
            {
                _host.AssertValue(input);
                _host.Assert(typeof(T) == columnType.ItemType.RawType);

                var srcTensorGetters = GetTensorValueGetters(input);

                ValueGetter<VBuffer<T>> valueGetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcTensorGetters);

                    var values = dst.Values;
                    if (Utils.Size(values) < _parent.OutputTypes[iinfo].VectorSize)
                        values = new T[_parent.OutputTypes[iinfo].VectorSize];

                    TensorFlowUtils.FetchData(_cachedOutputs[_parent.Outputs[iinfo]].Data, values);
                    dst = new VBuffer<T>(values.Length, values);
                };
                return valueGetter;
            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] srcTensorGetters)
            {
                if (_cachedPosition != position)
                {
                    var runner = _parent.Session.GetRunner();
                    for (int i = 0; i < _inputColIndices.Length; i++)
                    {
                        var inputName = _parent.Inputs[i];
                        runner.AddInput(inputName, srcTensorGetters[i].GetTensor());
                    }

                    var tensors = runner.Fetch(_parent.Outputs).Run();
                    Contracts.Assert(tensors.Length > 0);

                    for (int j = 0; j < tensors.Length; j++)
                    {
                        _cachedOutputs[_parent.Outputs[j]] = tensors[j];
                    }

                    _cachedPosition = position;
                }
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                _cachedPosition = -1;
                var getters = new Delegate[_parent.Outputs.Length];
                disposer = null;
                using (var ch = _host.Start("CreateGetters"))
                {
                    for (int i = 0; i < _parent.Outputs.Length; i++)
                    {
                        if (activeOutput(i))
                            getters[i] = MakeGetter(input, i);
                    }
                    ch.Done();
                    return getters;
                }
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => activeOutput(0) && _inputColIndices.Any(i => i == col);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                var info = new RowMapperColumnInfo[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                    info[i] = new RowMapperColumnInfo(_parent.Outputs[i], _parent.OutputTypes[i], null);
                return info;
            }

            private static (string[], int[], bool[], TFShape[], TFDataType[]) GetInputMetaData(TFGraph graph, string[] source, ISchema inputSchema)
            {
                var tfShapes = new TFShape[source.Length];
                var tfTypes = new TFDataType[source.Length];
                var colNames = new string[source.Length];
                var inputColIndices = new int[source.Length];
                var isInputVector = new bool[source.Length];
                for (int i = 0; i < source.Length; i++)
                {
                    colNames[i] = source[i];
                    if (!inputSchema.TryGetColumnIndex(colNames[i], out inputColIndices[i]))
                        throw Contracts.Except($"Column '{colNames[i]}' does not exist");

                    var tfoutput = new TFOutput(graph[colNames[i]]);
                    if (!TensorFlowUtils.IsTypeSupported(tfoutput.OutputType))
                        throw Contracts.Except($"Input type '{tfoutput.OutputType}' of input column '{colNames[i]}' is not supported in TensorFlow");

                    tfShapes[i] = graph.GetTensorShape(tfoutput);
                    var type = inputSchema.GetColumnType(inputColIndices[i]);
                    var shape = tfShapes[i].ToIntArray().Skip(tfShapes[i][0] == -1 ? BatchSize : 0);
                    if (type.AsVector.DimCount == 1)
                    {
                        int valCount = shape.Aggregate((x, y) => x * y);
                        if (type.ValueCount != valCount)
                            throw Contracts.Except($"Input shape mismatch: Input '{colNames[i]}' has shape {tfShapes[i].ToString()}, but input data is of length {valCount}.");
                    }
                    else if (shape.Select((dim, j) => dim != type.AsVector.GetDim(j)).Any(b => b))
                        throw Contracts.Except($"Input shape mismatch: Input '{colNames[i]}' has shape {tfShapes[i].ToString()}, but input data is {type.AsVector.ToString()}.");

                    isInputVector[i] = type.IsVector;

                    tfTypes[i] = tfoutput.OutputType;

                    var l = new long[tfShapes[i].NumDimensions];
                    for (int ishape = 0; ishape < tfShapes[i].NumDimensions; ishape++)
                    {
                        l[ishape] = tfShapes[i][ishape] == -1 ? BatchSize : tfShapes[i][ishape];
                    }
                    tfShapes[i] = new TFShape(l);
                }
                return (colNames, inputColIndices, isInputVector, tfShapes, tfTypes);
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

        public sealed class TensorFlowEstimator : IEstimator<TensorFlowTransform>
        {
            private readonly IHost _host;
            private readonly TensorFlowTransform _transformer;

            public TensorFlowTransform Fit(IDataView input) => _transformer;

            public TensorFlowEstimator(IHostEnvironment env, string modelFile, string[] inputs, string[] outputs)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(TensorFlowEstimator));
                _transformer = new TensorFlowTransform(env, File.ReadAllBytes(modelFile), inputs, outputs);
            }

            public SchemaShape GetOutputSchema(SchemaShape inputSchema)
            {
                _host.CheckValue(inputSchema, nameof(inputSchema));
                var result = inputSchema.Columns.ToDictionary(x => x.Name);
                foreach (var input in _transformer.Inputs)
                {
                    var col = inputSchema.FindColumn(input);
                    if (col == null)
                        throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                    var tfInput = new TFOutput(_transformer.Graph[input]);
                    if (!TensorFlowUtils.IsTypeSupported(tfInput.OutputType))
                        throw Contracts.Except($"Input type '{tfInput.OutputType}' of input column '{input}' is not supported in TensorFlow");
                    var tfShape = _transformer.Graph.GetTensorShape(tfInput);
                    var shape = tfShape.ToIntArray().Skip(tfShape[0] == -1 ? BatchSize : 0);

                    if (!(col.Kind == SchemaShape.Column.VectorKind.VariableVector || col.Kind == SchemaShape.Column.VectorKind.Vector))
                        throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, nameof(VectorType), col.GetTypeString());
                }

                var outColumns = new List<SchemaShape.Column>();
                for (var i = 0; i < _transformer.Outputs.Length; i++)
                {
                    //IVAN: not sure about VectorKind.
                    outColumns.Add(new SchemaShape.Column(_transformer.Outputs[i], SchemaShape.Column.VectorKind.Vector, _transformer.OutputTypes[i], false));
                }
                return new SchemaShape(outColumns);
            }

        }
    }
}
