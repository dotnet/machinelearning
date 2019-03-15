// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Onnx;
using OnnxShape = System.Collections.Generic.List<int>;

[assembly: LoadableClass(OnnxTransformer.Summary, typeof(IDataTransform), typeof(OnnxTransformer),
    typeof(OnnxTransformer.Options), typeof(SignatureDataTransform), OnnxTransformer.UserName, OnnxTransformer.ShortName, "OnnxTransform", "OnnxScorer")]

[assembly: LoadableClass(OnnxTransformer.Summary, typeof(IDataTransform), typeof(OnnxTransformer),
    null, typeof(SignatureLoadDataTransform), OnnxTransformer.UserName, OnnxTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(OnnxTransformer), null, typeof(SignatureLoadModel),
    OnnxTransformer.UserName, OnnxTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(OnnxTransformer), null, typeof(SignatureLoadRowMapper),
    OnnxTransformer.UserName, OnnxTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(OnnxTransformer))]

namespace Microsoft.ML.Transforms.Onnx
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
    /// <p>Supports inferencing of models in ONNX 1.2 and 1.3 format (opset 7, 8 and 9), using the
    /// <a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/'>Microsoft.ML.OnnxRuntime</a> library.
    /// </p>
    /// <p>Models are scored on CPU by default. If GPU execution is needed (optional), use the
    /// NuGet package available at
    /// <a href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/'>Microsoft.ML.OnnxRuntime.Gpu</a>
    /// and download
    /// <a href='https://developer.nvidia.com/cuda-downloads'>CUDA 9.1 Toolkit</a>
    /// and
    /// <a href='https://developer.nvidia.com/cudnn'>cuDNN</a>.
    ///  Set parameter 'gpuDeviceId' to a valid non-negative integer. Typical device ID values are 0 or 1.
    /// </p>
    /// <p>The inputs and outputs of the ONNX models must be Tensor type. Sequence and Maps are not yet supported.</p>
    /// <p>OnnxRuntime currently works on Windows and Ubuntu 16.04 Linux 64-bit platforms. Mac OS to be supported soon.</p>
    /// <p>Visit https://github.com/onnx/models to see a list of readily available models to get started with.</p>
    /// <p>Refer to http://onnx.ai' for more information about ONNX.</p>
    /// </remarks>
    public sealed class OnnxTransformer : RowToRowTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Path to the onnx model file.", ShortName = "model", SortOrder = 0)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Name of the input column.", SortOrder = 1)]
            public string[] InputColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Name of the output column.", SortOrder = 2)]
            public string[] OutputColumns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "GPU device id to run on (e.g. 0,1,..). Null for CPU. Requires CUDA 9.1.", SortOrder = 3)]
            public int? GpuDeviceId = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "If true, resumes execution on CPU upon GPU error. If false, will raise the GPU execption.", SortOrder = 4)]
            public bool FallbackToCpu = false;
        }

        private readonly Options _options;
        internal readonly OnnxModel Model;

        internal const string Summary = "Transforms the data using the Onnx model.";
        internal const string UserName = "ONNX Scoring Transform";
        internal const string ShortName = "Onnx";
        internal const string LoaderSignature = "OnnxTransform";

        internal readonly string[] Inputs;
        internal readonly string[] Outputs;
        internal readonly DataViewType[] OutputTypes;

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
            loaderAssemblyName: typeof(OnnxTransformer).Assembly.FullName);
        }

        // Factory method for SignatureDataTransform
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            return new OnnxTransformer(env, options).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadModel.
        private static OnnxTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
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

            var options = new Options() { InputColumns = inputs, OutputColumns = outputs };

            return new OnnxTransformer(env, options, modelBytes);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private OnnxTransformer(IHostEnvironment env, Options options, byte[] modelBytes = null) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(OnnxTransformer)))
        {
            Host.CheckValue(options, nameof(options));

            foreach (var col in options.InputColumns)
                Host.CheckNonWhiteSpace(col, nameof(options.InputColumns));
            foreach (var col in options.OutputColumns)
                Host.CheckNonWhiteSpace(col, nameof(options.OutputColumns));

            try
            {
                if (modelBytes == null)
                {
                    Host.CheckNonWhiteSpace(options.ModelFile, nameof(options.ModelFile));
                    Host.CheckIO(File.Exists(options.ModelFile), "Model file {0} does not exists.", options.ModelFile);
                    Model = new OnnxModel(options.ModelFile, options.GpuDeviceId, options.FallbackToCpu);
                }
                else
                    Model = OnnxModel.CreateFromBytes(modelBytes, options.GpuDeviceId, options.FallbackToCpu);
            }
            catch (OnnxRuntimeException e)
            {
                 throw Host.Except(e, $"Error initializing model :{e.ToString()}");
            }

            var modelInfo = Model.ModelInfo;
            Inputs = (options.InputColumns.Count() == 0) ? Model.InputNames.ToArray() : options.InputColumns;
            Outputs = (options.OutputColumns.Count() == 0) ? Model.OutputNames.ToArray() : options.OutputColumns;
            OutputTypes = new DataViewType[Outputs.Length];
            var numModelOutputs = Model.ModelInfo.OutputsInfo.Length;
            for (int i = 0; i < Outputs.Length; i++)
            {
                var idx = Model.OutputNames.IndexOf(Outputs[i]);
                if (idx < 0)
                    throw Host.Except($"Column {Outputs[i]} doesn't match output node names of model");

                var outputNodeInfo = Model.ModelInfo.OutputsInfo[idx];
                var shape = outputNodeInfo.Shape;
                var dims = AdjustDimensions(shape);
                OutputTypes[i] = new VectorType(OnnxUtils.OnnxToMlNetType(outputNodeInfo.Type), dims.ToArray());
            }
            _options = options;
        }

        /// <summary>
        /// Transform for scoring ONNX models. Input data column names/types must exactly match
        /// all model input names. All possible output columns are generated, with names/types
        /// specified by the model.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        internal OnnxTransformer(IHostEnvironment env, string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false)
            : this(env, new Options()
            {
                ModelFile = modelFile,
                InputColumns = new string[] {},
                OutputColumns = new string[] {},
                GpuDeviceId = gpuDeviceId,
                FallbackToCpu = fallbackToCpu
            })
        {
        }

        /// <summary>
        /// Transform for scoring ONNX models. Input data column name/type must exactly match
        /// the model specification. Only 1 output column is generated.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        internal OnnxTransformer(IHostEnvironment env, string outputColumnName, string modelFile, string inputColumnName = null, int? gpuDeviceId = null, bool fallbackToCpu = false)
            : this(env, new Options()
            {
                ModelFile = modelFile,
                InputColumns = new[] { inputColumnName ?? outputColumnName },
                OutputColumns = new[] { outputColumnName },
                GpuDeviceId = gpuDeviceId,
                FallbackToCpu = fallbackToCpu
            })
        {
        }

        /// <summary>
        /// Transform for scoring ONNX models. Input data column names/types must exactly match
        /// all model input names. Only the output columns specified will be generated.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="outputColumnNames">The output columns to generate. Names must match model specifications. Data types are inferred from model.</param>
        /// <param name="inputColumnNames">The name of the input data columns. Must match model's input names.</param>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        internal OnnxTransformer(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false)
            : this(env, new Options()
            {
                ModelFile = modelFile,
                InputColumns = inputColumnNames,
                OutputColumns = outputColumnNames,
                GpuDeviceId = gpuDeviceId,
                FallbackToCpu = fallbackToCpu
            })
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
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
        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema);

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
            private readonly OnnxTransformer _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly OnnxShape[] _inputTensorShapes;
            private readonly System.Type[] _inputOnnxTypes;

            public Mapper(OnnxTransformer parent, DataViewSchema inputSchema) :
                 base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {

                _parent = parent;
                _inputColIndices = new int[_parent.Inputs.Length];
                _isInputVector = new bool[_parent.Inputs.Length];
                _inputTensorShapes = new OnnxShape[_parent.Inputs.Length];
                _inputOnnxTypes = new System.Type[_parent.Inputs.Length];

                var model = _parent.Model;
                for (int i = 0; i < _parent.Inputs.Length; i++)
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

                    var col = inputSchema.GetColumnOrNull(_parent.Inputs[i]);
                    if (!col.HasValue)
                        throw Host.ExceptSchemaMismatch( nameof(inputSchema),"input", _parent.Inputs[i]);
                    _inputColIndices[i] = col.Value.Index;

                    var type = inputSchema[_inputColIndices[i]].Type;
                    var vectorType = type as VectorType;
                    _isInputVector[i] = vectorType != null;

                    if (vectorType != null && vectorType.Size == 0)
                        throw Host.Except($"Variable length input columns not supported");

                    if (type.GetItemType() != inputType)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i], inputType.ToString(), type.ToString());

                    // If the column is one dimension we make sure that the total size of the Onnx shape matches.
                    // Compute the total size of the known dimensions of the shape.
                    int valCount = inputShape.Where(x => x > 0).Aggregate((x, y) => x * y);
                    // The column length should be divisible by this, so that the other dimensions can be integral.
                    int typeValueCount = type.GetValueCount();
                    if (typeValueCount % valCount != 0)
                        throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {String.Join(",", inputShape)}, but input data is of length {typeValueCount}.");

                    //Host.Assert(_outputItemRawType == _outputColType.ItemType.RawType);
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                    info[i] = new DataViewSchema.DetachedColumn(_parent.Outputs[i], _parent.OutputTypes[i], null);
                return info;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

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

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);
                //Host.Assert(typeof(T) == _outputItemRawType);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();
                var type = OnnxUtils.OnnxToMlNetType(_parent.Model.ModelInfo.OutputsInfo[iinfo].Type).RawType;
                Host.Assert(type == _parent.OutputTypes[iinfo].GetItemType().RawType);
                var srcNamedValueGetters = GetNamedOnnxValueGetters(input, _parent.Inputs, _inputColIndices, _isInputVector, _inputOnnxTypes, _inputTensorShapes);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, iinfo, srcNamedValueGetters, activeOutputColNames, outputCache);
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo, INamedOnnxValueGetter[] srcNamedValueGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                Host.AssertValue(input);
                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcNamedValueGetters, activeOutputColNames, outputCache);
                    var namedOnnxValue = outputCache.Outputs[_parent.Outputs[iinfo]];
                    var denseTensor = namedOnnxValue.AsTensor<T>() as System.Numerics.Tensors.DenseTensor<T>;
                    if (denseTensor == null)
                        throw Host.Except($"Output column {namedOnnxValue.Name} doesn't contain a DenseTensor of expected type {typeof(T)}");
                    var editor = VBufferEditor.Create(ref dst, (int)denseTensor.Length);
                    denseTensor.Buffer.Span.CopyTo(editor.Values);
                    dst = editor.Commit();
                };
                return valuegetter;
            }

            private static INamedOnnxValueGetter[] GetNamedOnnxValueGetters(DataViewRow input,
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

            private static INamedOnnxValueGetter CreateNamedOnnxValueGetter(DataViewRow input, System.Type onnxType, bool isVector, string colName, int colIndex, OnnxShape onnxShape)
            {
                var type = OnnxUtils.OnnxToMlNetType(onnxType).RawType;
                Contracts.AssertValue(type);
                return Utils.MarshalInvoke(CreateNameOnnxValueGetter<int>, type, input, isVector, colName, colIndex, onnxShape);
            }

            private static INamedOnnxValueGetter CreateNameOnnxValueGetter<T>(DataViewRow input, bool isVector, string colName, int colIndex, OnnxShape onnxShape)
            {
                if (isVector)
                    return new NamedOnnxValueGetterVec<T>(input, colName, colIndex, onnxShape);
                return new NameOnnxValueGetter<T>(input, colName, colIndex);
            }

            private class NameOnnxValueGetter<T> : INamedOnnxValueGetter
            {
                private readonly ValueGetter<T> _srcgetter;
                private readonly string _colName;

                public NameOnnxValueGetter(DataViewRow input, string colName, int colIndex)
                {
                    _colName = colName;
                    _srcgetter = input.GetGetter<T>(input.Schema[colIndex]);
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
                public NamedOnnxValueGetterVec(DataViewRow input, string colName, int colIndex, OnnxShape tensorShape)
                {
                    _srcgetter = input.GetGetter<VBuffer<T>>(input.Schema[colIndex]);
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
    /// A class implementing the estimator interface of the <see cref="OnnxTransformer"/>.
    /// </summary>
    public sealed class OnnxScoringEstimator : TrivialEstimator<OnnxTransformer>
    {
        /// <summary>
        /// Transform for scoring ONNX models. Input data column names/types must exactly match
        /// all model input names. All possible output columns are generated, with names/types
        /// specified by model.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        [BestFriend]
        internal OnnxScoringEstimator(IHostEnvironment env, string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false)
            : this(env, new OnnxTransformer(env, new string[] { }, new string[] { }, modelFile, gpuDeviceId, fallbackToCpu))
        {
        }

        /// <summary>
        /// Transform for scoring ONNX models. Input data column names/types must exactly match
        /// all model input names. Only the output columns specified will be generated.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="outputColumnNames">The output columns to generate. Names must match model specifications. Data types are inferred from model.</param>
        /// <param name="inputColumnNames">The name of the input data columns. Must match model's input names.</param>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="gpuDeviceId">Optional GPU device ID to run execution on. Null for CPU.</param>
        /// <param name="fallbackToCpu">If GPU error, raise exception or fallback to CPU.</param>
        internal OnnxScoringEstimator(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, string modelFile,  int? gpuDeviceId = null, bool fallbackToCpu = false)
           : this(env, new OnnxTransformer(env, outputColumnNames, inputColumnNames, modelFile, gpuDeviceId, fallbackToCpu))
        {
        }

        internal OnnxScoringEstimator(IHostEnvironment env, OnnxTransformer transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(OnnxTransformer)), transformer)
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
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
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector", col.GetTypeString());

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
                    Transformer.OutputTypes[i].IsKnownSizeVector() ? SchemaShape.Column.VectorKind.Vector
                    : SchemaShape.Column.VectorKind.VariableVector, Transformer.OutputTypes[i].GetItemType(), false);
            }
            return new SchemaShape(resultDic.Values);
        }
    }
}
