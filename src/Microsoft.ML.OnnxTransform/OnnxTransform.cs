// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Scoring;
using Microsoft.ML.Transforms;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Core.Data;
using OnnxShape = System.Collections.Generic.List<long>;

[assembly: LoadableClass(OnnxTransform.Summary, typeof(IDataTransform), typeof(OnnxTransform),
    typeof(OnnxTransform.Arguments), typeof(SignatureDataTransform), OnnxTransform.UserName, OnnxTransform.ShortName, "OnnxTransform", "OnnxScorer")]

[assembly: LoadableClass(OnnxTransform.Summary, typeof(IDataTransform), typeof(OnnxTransform),
    null, typeof(SignatureLoadDataTransform), OnnxTransform.UserName, OnnxTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(OnnxTransform), null, typeof(SignatureLoadModel),
    OnnxTransform.UserName, OnnxTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(OnnxTransform), null, typeof(SignatureLoadRowMapper),
    OnnxTransform.UserName, OnnxTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(OnnxTransform))]

namespace Microsoft.ML.Transforms
{
    public sealed class OnnxTransform : ITransformer, ICanSaveModel
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Path to the onnx model file.", ShortName = "model", SortOrder = 0)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Name of the input column.", SortOrder = 1)]
            public string[] InputColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Name of the output column.", SortOrder = 2)]
            public string[] OutputColumns;
        }

        private readonly IHost _host;
        private readonly Arguments _args;
        internal readonly OnnxModel Model;
        private const string RegistrationName = "OnnxTransform";

        internal const string Summary = "Transforms the data using the Onnx model.";
        internal const string UserName = "ONNX Scoring Transform";
        internal const string ShortName = "Onnx";
        internal const string LoaderSignature = "OnnxTransform";

        public readonly string[] Inputs;
        public readonly string[] Outputs;
        public readonly ColumnType[] OutputTypes;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ONNXSCOR",
                verWrittenCur: 0x00010002, // Initial
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
	        loaderAssemblyName: typeof(OnnxTransform).Assembly.FullName);
        }

        public static IDataTransform Create(IHostEnvironment env, IDataView input, string modelFile, string[] inputColumns, string[] outputColumns)
        {
            var args = new Arguments { ModelFile = modelFile, InputColumns = inputColumns, OutputColumns = outputColumns };
            return Create(env, args, input);
        }

        // Factory method for SignatureDataTransform
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            return new OnnxTransform(env, args).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadModel.
        private static OnnxTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            byte[] modelBytes = null;
            if (!ctx.TryLoadBinaryStream("OnnxModel", r => modelBytes = r.ReadByteArray()))
                throw env.ExceptDecode();

            bool isMultiOutput = ctx.Header.ModelVerReadable > 0x00010001;

            //var inputColumn = ctx.LoadNonEmptyString();
            //var outputColumn = ctx.LoadNonEmptyString();

            var numInputs = 1;
            if (isMultiOutput)
                numInputs = ctx.Reader.ReadInt32();

            env.CheckDecode(numInputs > 0);
            var inputs = new string[numInputs];
            for (int j = 0; j < inputs.Length; j++)
                inputs[j] = ctx.LoadNonEmptyString();

            var numOutputs = 1;
            if (isMultiOutput)
                numOutputs = ctx.Reader.ReadInt32();

            env.CheckDecode(numOutputs > 0);
            var outputs = new string[numOutputs];
            for (int j = 0; j < outputs.Length; j++)
                outputs[j] = ctx.LoadNonEmptyString();

            var args = new Arguments() { InputColumns = inputs, OutputColumns = outputs };

            return new OnnxTransform(env, args, modelBytes);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private OnnxTransform(IHostEnvironment env, Arguments args, byte[] modelBytes = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(args, nameof(args));

            foreach (var col in args.InputColumns)
                _host.CheckNonWhiteSpace(col, nameof(args.InputColumns));
            foreach (var col in args.OutputColumns)
                _host.CheckNonWhiteSpace(col, nameof(args.OutputColumns));

            if (modelBytes == null)
            {
                _host.CheckNonWhiteSpace(args.ModelFile, nameof(args.ModelFile));
                _host.CheckUserArg(File.Exists(args.ModelFile), nameof(args.ModelFile));
                Model = new OnnxModel(args.ModelFile);
            }
            else
                Model = OnnxModel.CreateFromBytes(modelBytes);

            var modelInfo = Model.ModelInfo;
            //if (modelInfo.InputsInfo.Length != 1)
            //    throw env.Except($"OnnxTransform supports Onnx models with one input. The provided model has ${modelInfo.InputsInfo.Length} input(s).");
            //if (modelInfo.OutputsInfo.Length != 1)
            //    throw env.Except($"OnnxTransform supports Onnx models with one output. The provided model has ${modelInfo.OutputsInfo.Length} output(s).");

            Inputs = args.InputColumns;
            Outputs = args.OutputColumns;
            //var type = OnnxUtils.OnnxToMlNetType(outputNodeInfo.Type);
            //var shape = outputNodeInfo.Shape;
            //var dims = shape.Count > 0 ? shape.Skip(shape[0] < 0 ? 1 : 0).Select( x => (int) x ).ToArray() : new[] { 0 };

            OutputTypes = new ColumnType[args.OutputColumns.Length];

            var numModelOutputs = Model.ModelInfo.OutputsInfo.Length;
            for (int i=0; i < args.OutputColumns.Length; i++)
            {
                var idx = -1;
                for (var j = 0; j < Model.ModelInfo.OutputsInfo.Length; j++)
                    if (Model.ModelInfo.OutputsInfo[j].Name == args.OutputColumns[i])
                    {
                        idx = j;
                        break;
                    }
                if (idx < 0)
                    throw _host.Except($"Column {args.OutputColumns[i]} doesn't match output node names of model");
                var outputNodeInfo = Model.ModelInfo.OutputsInfo[idx];
                var shape = outputNodeInfo.Shape;
                var dims = shape.Count > 0 ? shape.Skip(shape[0] < 0 ? 1 : 0).Select(x => (int)x).ToArray() : new[] { 0 };
                OutputTypes[i] = new VectorType(OnnxUtils.OnnxToMlNetType(outputNodeInfo.Type), dims);
            }
            _args = args;
        }

        public OnnxTransform(IHostEnvironment env, string modelFile, string[] inputColumns, string[] outputColumns)
            : this(env, new Arguments() { ModelFile = modelFile, InputColumns = inputColumns, OutputColumns = outputColumns })
        {
        }

        public Schema GetOutputSchema(Schema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            foreach (var input in Inputs)
            {
                if (!inputSchema.TryGetColumnIndex(input, out int srcCol))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
            }
            var transform = Transform(new EmptyDataView(_host, inputSchema));
            return transform.Schema;
        }

        private IRowMapper MakeRowMapper(ISchema schema) => new Mapper(_host, this, Schema.Create(schema));

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

            ctx.SaveBinaryStream("OnnxModel", w => { w.WriteByteArray(Model.ToByteArray()); });
            //ctx.SaveNonEmptyString(_args.InputColumn);
            //ctx.SaveNonEmptyString(_args.OutputColumn);

            _host.AssertNonEmpty(Inputs);
            ctx.Writer.Write(Inputs.Length);
            foreach (var colName in Inputs)
                ctx.SaveNonEmptyString(colName);

            _host.AssertNonEmpty(Outputs);
            ctx.Writer.Write(Outputs.Length);
            foreach (var colName in Outputs)
                ctx.SaveNonEmptyString(colName);
        }

        public bool IsRowToRowMapper => true;

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            return MakeDataTransform(new EmptyDataView(_host, inputSchema));
        }

        private sealed class Mapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly OnnxTransform _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly OnnxShape[] _inputTensorShapes;
            private readonly DataType[] _inputOnnxTypes;

            public Mapper(IHostEnvironment env, OnnxTransform parent, Schema inputSchema)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(Mapper));
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckValue(parent, nameof(parent));

                _parent = parent;
                _inputColIndices = new int[_parent.Inputs.Length];
                _isInputVector = new bool[_parent.Inputs.Length];
                _inputTensorShapes = new OnnxShape[_parent.Inputs.Length];
                _inputOnnxTypes = new DataType[_parent.Inputs.Length];

                var model = _parent.Model;
                for (int i = 0; i <  _parent.Inputs.Length; i++)
                {
                    var idx = -1;
                    for (var j = 0; j < model.ModelInfo.InputsInfo.Length; j++)
                        if (model.ModelInfo.InputsInfo[j].Name == _parent.Inputs[i])
                        {
                            idx = j;
                            break;
                        }
                    if (idx < 0)
                        throw _host.Except($"Column {_parent.Inputs[i]} doesn't match input node names of model");

                    var inputNodeInfo = model.ModelInfo.InputsInfo[idx];

                    var shape = inputNodeInfo.Shape;
                    int[] inputdims = shape.Count > 0 ? shape.Skip(shape[0] < 0 ? 1 : 0).Select(x => (int)x).ToArray() : new[] { 0 };
                    var inputType = OnnxUtils.OnnxToMlNetType(inputNodeInfo.Type);

                    var inputShape = inputNodeInfo.Shape;
                    _inputTensorShapes[i] = inputShape;
                    _inputOnnxTypes[i] = inputNodeInfo.Type;

                    if (!inputSchema.TryGetColumnIndex(_parent.Inputs[i], out _inputColIndices[i]))
                        throw _host.Except($"Column {_parent.Inputs[i]} doesn't exist");

                    var type = inputSchema.GetColumnType(_inputColIndices[i]);
                    _isInputVector[i] = type.IsVector;

                    if (type.IsVector && type.VectorSize == 0)
                        throw _host.Except($"Variable length input columns not supported");

                    if (type.ItemType != inputType)
                        throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i], inputType.ToString(), type.ToString());

                    // If the column is one dimension we make sure that the total size of the Onnx shape matches.
                    // Compute the total size of the known dimensions of the shape.
                    int valCount = inputShape.Select(x => (int)x).Where(x => x > 0).Aggregate((x, y) => x * y);
                    // The column length should be divisible by this, so that the other dimensions can be integral.
                    if (type.ValueCount % valCount != 0)
                        throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {String.Join(",", inputShape)}, but input data is of length {type.ValueCount}.");

                    //_host.Assert(_outputItemRawType == _outputColType.ItemType.RawType);
                }
            }

            public Schema.Column[] GetOutputColumns()
            {
                var info = new Schema.Column[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                    info[i] = new Schema.Column(_parent.Outputs[i], _parent.OutputTypes[i], null);
                return info;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            public void Save(ModelSaveContext ctx)
            {
                _parent.Save(ctx);
            }

            private interface ITensorValueGetter
            {
                Tensor GetTensor();
            }
            private class OutputCache
            {
                public long Position;
                public Dictionary<string, Tensor> Outputs;
                public OutputCache()
                {
                    Position = -1;
                    Outputs = new Dictionary<string, Tensor>();
                }
            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    var inputTensors = new List<Tensor>();

                    for (int i = 0; i < _inputColIndices.Length; i++)
                        inputTensors.Add(srcTensorGetters[i].GetTensor());

                    var outputTensors = _parent.Model.Run(inputTensors);
                    Contracts.Assert(outputTensors.Count > 0);

                    for (int j = 0; j < outputTensors.Count; j++)
                        outputCache.Outputs[activeOutputColNames[j]] = outputTensors[j];

                    outputCache.Position = position;
                }
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                using (var ch = _host.Start("CreateGetters"))
                {
                    return MakeGetters(input, activeOutput);
                }
            }

            private Delegate[] MakeGetters(IRow input, Func<int, bool> activeOutput)
            {
                _host.AssertValue(input);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();

                var valueGetters = new Delegate[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                {
                    if (activeOutput(i))
                    {
                        var type = OnnxUtils.OnnxToMlNetType(_parent.Model.ModelInfo.OutputsInfo[i].Type).RawType;
                        _host.Assert(type == _parent.OutputTypes[i].ItemType.RawType);
                        var srcTensorGetters = GetTensorValueGetters(input, _inputColIndices, _isInputVector, _inputOnnxTypes, _inputTensorShapes);
                        valueGetters[i] = Utils.MarshalInvoke(MakeGetter<int>, type, input, i, srcTensorGetters, activeOutputColNames, outputCache);
                    }
                }
                return valueGetters;
            }

            private Delegate MakeGetter<T>(IRow input, int iinfo, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                _host.AssertValue(input);
                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                    var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                    var tensorSize = tensor.GetShape().Where(x => x > 0).Aggregate((x, y) => x * y);

                    var values = dst.Values;
                    if (Utils.Size(values) < tensorSize)
                        values = new T[tensorSize];

                    OnnxUtils.CopyTo(tensor, values);
                    dst = new VBuffer<T>(values.Length, values, dst.Indices);
                };
                return valuegetter;
            }

            private static ITensorValueGetter[] GetTensorValueGetters(IRow input,
                int[] inputColIndices,
                bool[] isInputVector,
                DataType[] onnxInputTypes,
                OnnxShape[] onnxInputShapes)
            {
                var srcTensorGetters = new ITensorValueGetter[inputColIndices.Length];
                for (int i = 0; i < inputColIndices.Length; i++)
                {
                    int colIndex = inputColIndices[i];
                    srcTensorGetters[i] = CreateTensorValueGetter(input, onnxInputTypes[i], isInputVector[i], colIndex, onnxInputShapes[i]);
                }
                return srcTensorGetters;
            }

            private static ITensorValueGetter CreateTensorValueGetter(IRow input, DataType onnxType, bool isVector, int colIndex, OnnxShape onnxShape)
            {
                var type = OnnxUtils.OnnxToMlNetType(onnxType).RawType;
                Contracts.AssertValue(type);
                return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type, input, isVector, colIndex, onnxShape);
            }

            private static ITensorValueGetter CreateTensorValueGetter<T>(IRow input, bool isVector, int colIndex, OnnxShape onnxShape)
            {
                if (isVector)
                    return new TensorValueGetterVec<T>(input, colIndex, onnxShape);
                return new TensorValueGetter<T>(input, colIndex);
            }

            private class TensorValueGetter<T> : ITensorValueGetter
            {
                private readonly ValueGetter<T> _srcgetter;

                public TensorValueGetter(IRow input, int colIndex)
                {
                    _srcgetter = input.GetGetter<T>(colIndex);
                }
                public Tensor GetTensor()
                {
                    var scalar = default(T);
                    _srcgetter(ref scalar);
                    return OnnxUtils.CreateScalarTensor(scalar);
                }
            }

            private class TensorValueGetterVec<T> : ITensorValueGetter
            {
                private readonly ValueGetter<VBuffer<T>> _srcgetter;
                private readonly OnnxShape _tensorShape;
                private VBuffer<T> _vBuffer;
                private VBuffer<T> _vBufferDense;
                public TensorValueGetterVec(IRow input, int colIndex, OnnxShape tensorShape)
                {
                    _srcgetter = input.GetGetter<VBuffer<T>>(colIndex);
                    _tensorShape = tensorShape;
                    _vBuffer = default;
                    _vBufferDense = default;
                }
                public Tensor GetTensor()
                {
                    _srcgetter(ref _vBuffer);
                    _vBuffer.CopyToDense(ref _vBufferDense);
                    return OnnxUtils.CreateTensor(_vBufferDense.Values, _tensorShape);
                }
            }
        }
    }
    public sealed class OnnxScoringEstimator : TrivialEstimator<OnnxTransform>
    {
        public OnnxScoringEstimator(IHostEnvironment env, string modelFile, string[] inputs, string[] outputs)
           : this(env, new OnnxTransform(env, modelFile, inputs, outputs))
        {
        }

        public OnnxScoringEstimator(IHostEnvironment env, OnnxTransform transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(OnnxTransform)), transformer)
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

                var idx = -1;
                var inputsInfo = Transformer.Model.ModelInfo.InputsInfo;
                for (var j = 0; j < inputsInfo.Length; j++)
                    if (inputsInfo[j].Name == input)
                    {
                        idx = j;
                        break;
                    }
                if (idx < 0)
                    throw Host.Except($"Column {input} doesn't match input node names of model.");

                var inputNodeInfo = inputsInfo[idx];
                var expectedType = OnnxUtils.OnnxToMlNetType(inputNodeInfo.Type);
                if (col.ItemType != expectedType)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, expectedType.ToString(), col.ItemType.ToString());
            }

            for (var i = 0; i < Transformer.Outputs.Length; i++)
            {
                resultDic[Transformer.Outputs[i]] = new SchemaShape.Column(Transformer.Outputs[i],
                    Transformer.OutputTypes[i].IsKnownSizeVector ? SchemaShape.Column.VectorKind.Vector
                    : SchemaShape.Column.VectorKind.VariableVector, NumberType.R4, false);
            }
            return new SchemaShape(resultDic.Values);
        }
    }

    public static class OnnxStaticExtensions
    {

        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(Vector<float> input, string modelFile)
                : base(new Reconciler(modelFile), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly string _modelFile;

            public Reconciler(string modelFile)
            {
                Contracts.AssertNonEmpty(modelFile);
                _modelFile = modelFile;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var outCol = (OutColumn)toOutput[0];
                return new OnnxScoringEstimator(env, _modelFile, new[] { inputNames[outCol.Input] }, new[] { outputNames[outCol] });
            }
        }

        /// <summary>
        /// Run a Onnx model on the input column and extract one output column.
        /// The inputs and outputs are matched to Onnx graph nodes by name.
        /// </summary>
        public static Vector<float> ApplyOnnxModel(this Vector<float> input, string modelFile)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckNonEmpty(modelFile, nameof(modelFile));
            return new OutColumn(input, modelFile);
        }
    }
}
