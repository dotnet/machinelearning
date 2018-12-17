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
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Transforms;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using OnnxShape = System.Collections.Generic.List<int>;

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
    /// <summary>
    /// <p>A transform for scoring ONNX models in the ML.NET framework.</p>
    /// <format type="text/markdown">
    /// <![CDATA[
    /// [!code-csharp[MF](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/OnnxTransform.cs)]
    /// ]]>
    /// </format>
    /// </summary>
    /// <remarks>
    /// <p>Supports inferencing of models in 1.2 and 1.3 format, using the
    /// <a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/'>Microsoft.ML.OnnxRuntime</a> library
    /// </p>
    /// <p>The inputs and outputs of the onnx models must of of Tensors. Sequence and Maps are not yet supported.</p>
    /// <p>Visit https://github.com/onnx/models to see a list of readily available models to get started with.</p>
    /// <p>Refer to http://onnx.ai' for more information about ONNX.</p>
    /// </remarks>
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

        public static IDataTransform Create(IHostEnvironment env, IDataView input, string modelFile)
        {
            var args = new Arguments { ModelFile = modelFile, InputColumns = new string[] { }, OutputColumns = new string[] { } };
            return Create(env, args, input);
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

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
            Inputs = (args.InputColumns.Count() == 0 ) ? Model.InputNames.ToArray() : args.InputColumns;
            Outputs = (args.OutputColumns.Count() == 0 ) ? Model.OutputNames.ToArray() : args.OutputColumns;
            OutputTypes = new ColumnType[Outputs.Length];
            var numModelOutputs = Model.ModelInfo.OutputsInfo.Length;
            for (int i=0; i < Outputs.Length; i++)
            {
                var idx = Model.OutputNames.IndexOf(Outputs[i]);
                if (idx < 0)
                    throw Host.Except($"Column {Outputs[i]} doesn't match output node names of model");

                var outputNodeInfo = Model.ModelInfo.OutputsInfo[idx];
                var shape = outputNodeInfo.Shape;
                var dims = AdjustDimensions(shape);
                OutputTypes[i] = new VectorType(OnnxUtils.OnnxToMlNetType(outputNodeInfo.Type), dims.ToArray());
            }
            _args = args;
        }

        public OnnxTransform(IHostEnvironment env, string modelFile)
            : this(env, new Arguments() { ModelFile = modelFile, InputColumns = new string[] { }, OutputColumns = new string[] { } })
        {
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
        private protected override IRowMapper MakeRowMapper(Schema inputSchema) => new Mapper(this, inputSchema);

        private static IEnumerable<int> AdjustDimensions(OnnxShape shape)
        {
            // if the model output is of type Map or Sequence, the shape property
            // will not be filled (so count=0). Don't throw an exception here
            // it will be runtime exception, util Maps and Sequences become supported.
            if (shape.Count > 0)
            {
                return shape.Select(x => (x <= 0) ? 1 : x);
            }
            return new[] { 1 };
        }

        private sealed class Mapper : MapperBase
        {
            private readonly OnnxTransform _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly OnnxShape[] _inputTensorShapes;
            private readonly System.Type[] _inputOnnxTypes;

            public Mapper(OnnxTransform parent, Schema inputSchema) :
                 base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema)
            {

                _parent = parent;
                _inputColIndices = new int[_parent.Inputs.Length];
                _isInputVector = new bool[_parent.Inputs.Length];
                _inputTensorShapes = new OnnxShape[_parent.Inputs.Length];
                _inputOnnxTypes = new System.Type[_parent.Inputs.Length];

                var model = _parent.Model;
                for (int i = 0; i <  _parent.Inputs.Length; i++)
                {
                    var idx = model.InputNames.IndexOf(_parent.Inputs[i]);
                    if (idx < 0)
                        throw Host.Except($"Column {_parent.Inputs[i]} doesn't match input node names of model");

                    var inputNodeInfo = model.ModelInfo.InputsInfo[idx];

                    var shape = inputNodeInfo.Shape;
                    var inputType = OnnxUtils.OnnxToMlNetType(inputNodeInfo.Type);

                    var inputShape = AdjustDimensions(inputNodeInfo.Shape);
                    _inputTensorShapes[i] = inputShape.ToList();
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

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            public override void Save(ModelSaveContext ctx) => _parent.Save(ctx);

            private interface INamedOnnxValueGetter
            {
                NamedOnnxValue GetNamedOnnxValue();
            }
            private class OutputCache
            {
                public long Position;
                public Dictionary<string, NamedOnnxValue> Outputs;
                public OutputCache()
                {
                    Position = -1;
                    Outputs = new Dictionary<string, NamedOnnxValue>();
                }
            }

            private void UpdateCacheIfNeeded(long position, INamedOnnxValueGetter[] srcNamedOnnxValueGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    var inputNameOnnxValues = new List<NamedOnnxValue>();

                    for (int i = 0; i < _inputColIndices.Length; i++)
                    {
                        inputNameOnnxValues.Add(srcNamedOnnxValueGetters[i].GetNamedOnnxValue());
                    }

                    var outputNamedOnnxValues = _parent.Model.Run(inputNameOnnxValues);
                    Contracts.Assert(outputNamedOnnxValues.Count > 0);

                    foreach (var outputNameOnnxValue in outputNamedOnnxValues)
                    {
                        outputCache.Outputs[outputNameOnnxValue.Name] = outputNameOnnxValue;
                    }
                    outputCache.Position = position;
                }
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);
                //Host.Assert(typeof(T) == _outputItemRawType);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();
                var type = OnnxUtils.OnnxToMlNetType(_parent.Model.ModelInfo.OutputsInfo[iinfo].Type).RawType;
                Host.Assert(type == _parent.OutputTypes[iinfo].ItemType.RawType);
                var srcNamedValueGetters = GetNamedOnnxValueGetters(input, _parent.Inputs, _inputColIndices, _isInputVector, _inputOnnxTypes, _inputTensorShapes);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, iinfo, srcNamedValueGetters, activeOutputColNames, outputCache);
            }

            private Delegate MakeGetter<T>(Row input, int iinfo, INamedOnnxValueGetter[] srcNamedValueGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                Host.AssertValue(input);
                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcNamedValueGetters, activeOutputColNames, outputCache);
                    var namedOnnxValue = outputCache.Outputs[_parent.Outputs[iinfo]];
                    var denseTensor = namedOnnxValue.AsTensor<T>() as System.Numerics.Tensors.DenseTensor<T>;
                    if (denseTensor == null)
                        throw Host.Except($"Output column {namedOnnxValue.Name} doesn't contain a DenseTensor of expected type {typeof(T)}");
                    var editor = VBufferEditor.Create(ref dst, (int) denseTensor.Length);
                    denseTensor.Buffer.Span.CopyTo(editor.Values);
                    dst = editor.Commit();
                };
                return valuegetter;
            }

            private static INamedOnnxValueGetter[] GetNamedOnnxValueGetters(Row input,
                string[] inputColNames,
                int[] inputColIndices,
                bool[] isInputVector,
                System.Type[] onnxInputTypes,
                OnnxShape[] onnxInputShapes)
            {
                var srcNamedOnnxValueGetters = new INamedOnnxValueGetter[inputColIndices.Length];
                for (int i = 0; i < inputColIndices.Length; i++)
                {
                    int colIndex = inputColIndices[i];
                    srcNamedOnnxValueGetters[i] = CreateNamedOnnxValueGetter(input, onnxInputTypes[i], isInputVector[i], inputColNames[i], colIndex, onnxInputShapes[i]);
                }
                return srcNamedOnnxValueGetters;
            }

            private static INamedOnnxValueGetter CreateNamedOnnxValueGetter(Row input, System.Type onnxType, bool isVector, string colName, int colIndex, OnnxShape onnxShape)
            {
                var type = OnnxUtils.OnnxToMlNetType(onnxType).RawType;
                Contracts.AssertValue(type);
                return Utils.MarshalInvoke(CreateNameOnnxValueGetter<int>, type, input, isVector, colName, colIndex, onnxShape);
            }

            private static INamedOnnxValueGetter CreateNameOnnxValueGetter<T>(Row input, bool isVector, string colName, int colIndex, OnnxShape onnxShape)
            {
                if (isVector)
                    return new NamedOnnxValueGetterVec<T>(input, colName, colIndex, onnxShape);
                return new NameOnnxValueGetter<T>(input, colName, colIndex);
            }

            private class NameOnnxValueGetter<T> : INamedOnnxValueGetter
            {
                private readonly ValueGetter<T> _srcgetter;
                private readonly string _colName;

                public NameOnnxValueGetter(Row input, string colName, int colIndex)
                {
                    _colName = colName;
                    _srcgetter = input.GetGetter<T>(colIndex);
                }
                public NamedOnnxValue GetNamedOnnxValue()
                {
                    var scalar = default(T);
                    _srcgetter(ref scalar);
                    return OnnxUtils.CreateScalarNamedOnnxValue(_colName, scalar);
                }
            }

            private class NamedOnnxValueGetterVec<T> : INamedOnnxValueGetter
            {
                private readonly ValueGetter<VBuffer<T>> _srcgetter;
                private readonly OnnxShape _tensorShape;
                private readonly string _colName;
                private VBuffer<T> _vBuffer;
                private VBuffer<T> _vBufferDense;
                public NamedOnnxValueGetterVec(Row input, string colName, int colIndex, OnnxShape tensorShape)
                {
                    _srcgetter = input.GetGetter<VBuffer<T>>(colIndex);
                    _tensorShape = tensorShape;
                    _colName = colName;
                    _vBuffer = default;
                    _vBufferDense = default;
                }
                public NamedOnnxValue GetNamedOnnxValue()
                {
                    _srcgetter(ref _vBuffer);
                    _vBuffer.CopyToDense(ref _vBufferDense);
                    return OnnxUtils.CreateNamedOnnxValue(_colName, _vBufferDense.GetValues(), _tensorShape);
                }
            }
        }
    }

    /// <summary>
    /// A class implementing the estimator interface of the OnnxTransform.
    /// </summary>
    public sealed class OnnxScoringEstimator : TrivialEstimator<OnnxTransform>
    {
        public OnnxScoringEstimator(IHostEnvironment env, string modelFile)
            : this(env, new OnnxTransform(env, modelFile, new string[] { }, new string[] { }))
        {
        }

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
            var result = inputSchema.ToDictionary(x => x.Name);
            var resultDic = inputSchema.ToDictionary(x => x.Name);

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
                    : SchemaShape.Column.VectorKind.VariableVector, Transformer.OutputTypes[i].ItemType, false);
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

