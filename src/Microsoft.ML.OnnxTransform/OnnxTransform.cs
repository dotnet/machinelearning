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
using Microsoft.ML.Data;
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
    public sealed class OnnxTransform : RowToRowTransformerBase
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

        private readonly Arguments _args;
        internal readonly OnnxModel Model;

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
                // version 10001 is single input & output.
                // version 10002 = multiple inputs & outputs
                verWrittenCur: 0x00010002,
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

            bool supportsMultiInputOutput = ctx.Header.ModelVerWritten > 0x00010001;

            var numInputs = (supportsMultiInputOutput) ? ctx.Reader.ReadInt32() : 1;

            env.CheckDecode(numInputs > 0);
            var inputs = new string[numInputs];
            for (int j = 0; j < inputs.Length; j++)
                inputs[j] = ctx.LoadNonEmptyString();

            var numOutputs = (supportsMultiInputOutput) ? ctx.Reader.ReadInt32() : 1;

            env.CheckDecode(numOutputs > 0);
            var outputs = new string[numOutputs];
            for (int j = 0; j < outputs.Length; j++)
                outputs[j] = ctx.LoadNonEmptyString();

            var args = new Arguments() { InputColumns = inputs, OutputColumns = outputs };

            return new OnnxTransform(env, args, modelBytes);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

        private OnnxTransform(IHostEnvironment env, Arguments args, byte[] modelBytes = null) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(OnnxTransform)))
        {
            Host.CheckValue(args, nameof(args));

            foreach (var col in args.InputColumns)
                Host.CheckNonWhiteSpace(col, nameof(args.InputColumns));
            foreach (var col in args.OutputColumns)
                Host.CheckNonWhiteSpace(col, nameof(args.OutputColumns));

            if (modelBytes == null)
            {
                Host.CheckNonWhiteSpace(args.ModelFile, nameof(args.ModelFile));
                Host.CheckUserArg(File.Exists(args.ModelFile), nameof(args.ModelFile));
                Model = new OnnxModel(args.ModelFile);
            }
            else
                Model = OnnxModel.CreateFromBytes(modelBytes);

            var modelInfo = Model.ModelInfo;
            Inputs = args.InputColumns;
            Outputs = args.OutputColumns;
            OutputTypes = new ColumnType[args.OutputColumns.Length];
            var numModelOutputs = Model.ModelInfo.OutputsInfo.Length;
            for (int i=0; i < args.OutputColumns.Length; i++)
            {
                var idx = Model.OutputNames.IndexOf(args.OutputColumns[i]);
                if (idx < 0)
                    throw Host.Except($"Column {args.OutputColumns[i]} doesn't match output node names of model");

                var outputNodeInfo = Model.ModelInfo.OutputsInfo[idx];
                var shape = outputNodeInfo.Shape;
                var dims = AdjustDimensions(shape);
                OutputTypes[i] = new VectorType(OnnxUtils.OnnxToMlNetType(outputNodeInfo.Type), dims);
            }
            _args = args;
        }

        public OnnxTransform(IHostEnvironment env, string modelFile, string inputColumn, string outputColumn)
            : this(env, new Arguments() { ModelFile = modelFile, InputColumns = new[] { inputColumn }, OutputColumns = new[] { outputColumn } })
        {
        }

        public OnnxTransform(IHostEnvironment env, string modelFile, string[] inputColumns, string[] outputColumns)
            : this(env, new Arguments() { ModelFile = modelFile, InputColumns = inputColumns, OutputColumns = outputColumns })
        {
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.SaveBinaryStream("OnnxModel", w => { w.WriteByteArray(Model.ToByteArray()); });

            Host.CheckNonEmpty(Inputs, nameof(Inputs));
            ctx.Writer.Write(Inputs.Length);
            foreach (var colName in Inputs)
                ctx.SaveNonEmptyString(colName);

            Host.CheckNonEmpty(Outputs, nameof(Outputs));
            ctx.Writer.Write(Outputs.Length);
            foreach (var colName in Outputs)
                ctx.SaveNonEmptyString(colName);
        }
        protected override IRowMapper MakeRowMapper(Schema inputSchema) => new Mapper(this, inputSchema);

        private static int[] AdjustDimensions(OnnxShape shape)
        {
            // if the model output is of type Map or Sequence, the shape property
            // will not be filled (so count=0). Don't throw an exception here
            // it will be runtime exception, util Maps and Sequences become supported.
            if (shape.Count > 0)
            {
                // some models may have -1 in first position.
                // skip this dimension when setting output column dimensions.
                if (shape[0] < 0)
                {
                    return shape.Skip(1).Select(x => (int)x).ToArray();
                }
                else
                {
                    return shape.Select(x => (int)x).ToArray();
                }
            }
            return new[] { 0 };
        }

        private sealed class Mapper : MapperBase
        {
            private readonly OnnxTransform _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly OnnxShape[] _inputTensorShapes;
            private readonly DataType[] _inputOnnxTypes;

            public Mapper(OnnxTransform parent, Schema inputSchema) :
                 base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema)
            {

                _parent = parent;
                _inputColIndices = new int[_parent.Inputs.Length];
                _isInputVector = new bool[_parent.Inputs.Length];
                _inputTensorShapes = new OnnxShape[_parent.Inputs.Length];
                _inputOnnxTypes = new DataType[_parent.Inputs.Length];

                var model = _parent.Model;
                for (int i = 0; i <  _parent.Inputs.Length; i++)
                {
                    var idx = model.InputNames.IndexOf(_parent.Inputs[i]);
                    if (idx < 0)
                        throw Host.Except($"Column {_parent.Inputs[i]} doesn't match input node names of model");

                    var inputNodeInfo = model.ModelInfo.InputsInfo[idx];

                    var shape = inputNodeInfo.Shape;
                    var inputType = OnnxUtils.OnnxToMlNetType(inputNodeInfo.Type);

                    var inputShape = inputNodeInfo.Shape;
                    _inputTensorShapes[i] = inputShape;
                    _inputOnnxTypes[i] = inputNodeInfo.Type;

                    if (!inputSchema.TryGetColumnIndex(_parent.Inputs[i], out _inputColIndices[i]))
                        throw Host.Except($"Column {_parent.Inputs[i]} doesn't exist");

                    var type = inputSchema.GetColumnType(_inputColIndices[i]);
                    _isInputVector[i] = type.IsVector;

                    if (type.IsVector && type.VectorSize == 0)
                        throw Host.Except($"Variable length input columns not supported");

                    if (type.ItemType != inputType)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i], inputType.ToString(), type.ToString());

                    // If the column is one dimension we make sure that the total size of the Onnx shape matches.
                    // Compute the total size of the known dimensions of the shape.
                    int valCount = inputShape.Select(x => (int)x).Where(x => x > 0).Aggregate((x, y) => x * y);
                    // The column length should be divisible by this, so that the other dimensions can be integral.
                    if (type.ValueCount % valCount != 0)
                        throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {String.Join(",", inputShape)}, but input data is of length {type.ValueCount}.");

                    //Host.Assert(_outputItemRawType == _outputColType.ItemType.RawType);
                }
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new Schema.DetachedColumn[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                    info[i] = new Schema.DetachedColumn(_parent.Outputs[i], _parent.OutputTypes[i], null);
                return info;
            }

            public override Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            public override void Save(ModelSaveContext ctx) => _parent.Save(ctx);

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

            protected override Delegate MakeGetter(IRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);
                //Host.Assert(typeof(T) == _outputItemRawType);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();
                var type = OnnxUtils.OnnxToMlNetType(_parent.Model.ModelInfo.OutputsInfo[iinfo].Type).RawType;
                Host.Assert(type == _parent.OutputTypes[iinfo].ItemType.RawType);
                var srcTensorGetters = GetTensorValueGetters(input, _inputColIndices, _isInputVector, _inputOnnxTypes, _inputTensorShapes);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, iinfo, srcTensorGetters, activeOutputColNames, outputCache);
            }

            private Delegate MakeGetter<T>(IRow input, int iinfo, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                Host.AssertValue(input);
                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);
                    var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                    var editor = VBufferEditor.Create(ref dst, tensor.GetSize());
                    OnnxUtils.CopyTo(tensor, editor.Values);
                    dst = editor.Commit();
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
                    return OnnxUtils.CreateTensor(_vBufferDense.GetValues(), _tensorShape);
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

                var inputsInfo = Transformer.Model.ModelInfo.InputsInfo;
                var idx = Transformer.Model.InputNames.IndexOf(input);
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

