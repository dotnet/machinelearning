// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Onnx;
using static Microsoft.ML.Model.OnnxConverter.OnnxCSharpToProtoWrapper;
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
    /// <see cref="ITransformer"/> resulting from fitting an <see cref="OnnxScoringEstimator"/>.
    /// Please refer to <see cref="OnnxScoringEstimator"/> to learn more about the necessary dependencies,
    /// and how to run it on a GPU.
    /// </summary>
    public sealed class OnnxTransformer : RowToRowTransformerBase, IDisposable
    {
        /// <summary>
        /// A class used for capturing shape information from command line.
        /// <see cref="Name"/> is a tensor name while <see cref="Shape"/> is that tenor's desired shape.
        /// <see cref="CustomShapeInfo"/> is useful because sometime we want to overwrite unknown
        /// shapes loaded from ONNX model.
        /// </summary>
        internal sealed class CustomShapeInfo
        {
            // Examples of how a column is defined in command line API:
            // 2-by-3 tensor:
            //      Name=tensorName shape=2 shape=3

            public CustomShapeInfo() { }

            public CustomShapeInfo(string name, int[] shape)
            {
                Name = name;
                Shape = shape;
            }

            [Argument(ArgumentType.Required, HelpText = "Name of the column")]
            public string Name;

            [Argument(ArgumentType.Multiple, HelpText = "Shape of the column")]
            public int[] Shape;
        }

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

            [Argument(ArgumentType.AtMostOnce, HelpText = "If true, resumes execution on CPU upon GPU error. If false, will raise the GPU exception.", SortOrder = 4)]
            public bool FallbackToCpu = false;

            [Argument(ArgumentType.Multiple, HelpText = "Shapes used to overwrite shapes loaded from ONNX file.", SortOrder = 5)]
            public CustomShapeInfo[] CustomShapeInfos;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Protobuf CodedInputStream recursion limit.", SortOrder = 6)]
            public int RecursionLimit = 100;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Controls the number of threads used to parallelize the execution of the graph (across nodes).", SortOrder = 7)]
            public int? InterOpNumThreads = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Controls the number of threads to use to run the model.", SortOrder = 8)]
            public int? IntraOpNumThreads = null;
        }

        /// <summary>
        /// Options used to construct this class.
        /// </summary>
        private readonly Options _options;
        /// <summary>
        /// This field is internal because the associated estimator may access it.
        /// </summary>
        internal readonly OnnxModel Model;

        internal const string Summary = "Transforms the data using the Onnx model.";
        internal const string UserName = "ONNX Scoring Transform";
        internal const string ShortName = "Onnx";
        internal const string LoaderSignature = "OnnxTransform";

        /// <summary>
        /// Input column names from ML.NET's perspective. It can be ordered differently than ONNX model's input list.
        /// It's also possible that the <see cref="Inputs"/> contains less variables than ONNX model's input list.
        /// For each name in <see cref="Inputs"/>, an input tensor with the same name can be found in the underlying ONNX model.
        /// </summary>
        internal string[] Inputs { get; }
        /// <summary>
        /// Output column names from ML.NET's perspective. It can be ordered differently than ONNX model's output list.
        /// It's also possible that the <see cref="Outputs"/> contains less variables than ONNX model's output list.
        /// For each name in <see cref="Outputs"/>, an output tensor with the same name can be found in the underlying ONNX model.
        /// </summary>
        internal string[] Outputs { get; }
        /// <summary>
        /// Types of <see cref="Outputs"/>. The i-th element is the type of the i-th output in <see cref="Outputs"/>.
        /// </summary>
        internal DataViewType[] OutputTypes { get; }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ONNXSCOR",
                // version 10001 is single input & output.
                // version 10002 = multiple inputs & outputs
                // version 10003 = custom protobuf recursion limit
                verWrittenCur: 0x00010003,
                verReadableCur: 0x00010003,
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

            // Save custom-provided shapes. Those shapes overwrite shapes loaded from the ONNX model file.
            int customShapeInfosLength = ctx.Reader.ReadInt32(); // 0 means no custom shape. Non-zero means count of custom shapes.
            CustomShapeInfo[] loadedCustomShapeInfos = null;
            if (customShapeInfosLength > 0)
            {
                loadedCustomShapeInfos = new CustomShapeInfo[customShapeInfosLength];
                for (int i = 0; i < customShapeInfosLength; ++i)
                {
                    var name = ctx.LoadNonEmptyString();
                    var shape = ctx.Reader.ReadIntArray();
                    loadedCustomShapeInfos[i] = new CustomShapeInfo() { Name = name, Shape = shape };
                }
            }

            int recursionLimit;

            // Recursion limit change
            if (ctx.Header.ModelVerWritten >= 0x00010003)
            {
                recursionLimit = ctx.Reader.ReadInt32();
            }
            else
            {
                // Default if not written inside ONNX model
                recursionLimit = 100;
            }

            var options = new Options()
            {
                InputColumns = inputs,
                OutputColumns = outputs,
                CustomShapeInfos = loadedCustomShapeInfos,
                RecursionLimit = recursionLimit
            };

            IHostEnvironmentInternal localEnvironment = env as IHostEnvironmentInternal;
            if (localEnvironment is not null)
            {
                options.GpuDeviceId = localEnvironment.GpuDeviceId;
                options.FallbackToCpu = localEnvironment.FallbackToCpu;
            }

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

            // Cast options.CustomShapeInfos so that the user-specified shapes can be consumed by other
            // internal functions. If nothing is provided, shapeDictionary is null.
            var shapeDictionary = new Dictionary<string, int[]>();
            if (options.CustomShapeInfos != null)
                foreach (var customShape in options.CustomShapeInfos)
                    shapeDictionary[customShape.Name] = customShape.Shape;

            // Use ONNXRuntime to figure out the right input and output configuration.
            // However, ONNXRuntime doesn't provide strongly-typed method to access the produced
            // variables, we will inspect the ONNX model file to get information regarding types.
            try
            {
                if (modelBytes == null)
                {
                    // Entering this region means that the model file is passed in by the user.
                    Host.CheckNonWhiteSpace(options.ModelFile, nameof(options.ModelFile));
                    Host.CheckIO(File.Exists(options.ModelFile), "Model file {0} does not exists.", options.ModelFile);
                    // Because we cannot delete the user file, ownModelFile should be false.
                    Model = new OnnxModel(options.ModelFile, options.GpuDeviceId, options.FallbackToCpu, ownModelFile: false, shapeDictionary: shapeDictionary, options.RecursionLimit,
                        options.InterOpNumThreads, options.IntraOpNumThreads);
                }
                else
                {
                    // Entering this region means that the byte[] is passed as the model. To feed that byte[] to ONNXRuntime, we need
                    // to create a temporal file to store it and then call ONNXRuntime's API to load that file.
                    Model = OnnxModel.CreateFromBytes(modelBytes, env, options.GpuDeviceId, options.FallbackToCpu, shapeDictionary: shapeDictionary, options.RecursionLimit);
                }
            }
            catch (OnnxRuntimeException e)
            {
                throw Host.Except(e, $"Error initializing model :{e.ToString()}");
            }

            var modelInfo = Model.ModelInfo;
            Inputs = (options.InputColumns.Count() == 0) ? Model.ModelInfo.InputNames.ToArray() : options.InputColumns;
            Outputs = (options.OutputColumns.Count() == 0) ? Model.ModelInfo.OutputNames.ToArray() : options.OutputColumns;
            OutputTypes = new DataViewType[Outputs.Length];
            var numModelOutputs = Model.ModelInfo.OutputsInfo.Length;
            for (int i = 0; i < Outputs.Length; i++)
            {
                var outputInfo = Model.ModelInfo.GetOutput(Outputs[i]);
                OutputTypes[i] = outputInfo.DataViewType;
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
        /// <param name="shapeDictionary"></param>
        /// <param name="recursionLimit">Optional, specifies the Protobuf CodedInputStream recursion limit. Default value is 100.</param>
        internal OnnxTransformer(IHostEnvironment env, string modelFile, int? gpuDeviceId = null,
            bool fallbackToCpu = false, IDictionary<string, int[]> shapeDictionary = null, int recursionLimit = 100)
            : this(env, new Options()
            {
                ModelFile = modelFile,
                InputColumns = new string[] { },
                OutputColumns = new string[] { },
                GpuDeviceId = gpuDeviceId,
                FallbackToCpu = fallbackToCpu,
                CustomShapeInfos = shapeDictionary?.Select(pair => new CustomShapeInfo(pair.Key, pair.Value)).ToArray(),
                RecursionLimit = recursionLimit
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
        /// <param name="shapeDictionary"></param>
        /// <param name="recursionLimit">Optional, specifies the Protobuf CodedInputStream recursion limit. Default value is 100.</param>
        /// <param name="interOpNumThreads">Controls the number of threads used to parallelize the execution of the graph (across nodes).</param>
        /// <param name="intraOpNumThreads">Controls the number of threads to use to run the model.</param>
        internal OnnxTransformer(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false,
            IDictionary<string, int[]> shapeDictionary = null, int recursionLimit = 100,
            int? interOpNumThreads = null, int? intraOpNumThreads = null)
            : this(env, new Options()
            {
                ModelFile = modelFile,
                InputColumns = inputColumnNames,
                OutputColumns = outputColumnNames,
                GpuDeviceId = gpuDeviceId,
                FallbackToCpu = fallbackToCpu,
                CustomShapeInfos = shapeDictionary?.Select(pair => new CustomShapeInfo(pair.Key, pair.Value)).ToArray(),
                RecursionLimit = recursionLimit,
                InterOpNumThreads = interOpNumThreads,
                IntraOpNumThreads = intraOpNumThreads
            })
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.SaveBinaryStream("OnnxModel", w => { w.WriteByteArray(File.ReadAllBytes(Model.ModelStream.Name)); });

            Host.CheckNonEmpty(Inputs, nameof(Inputs));
            ctx.Writer.Write(Inputs.Length);
            foreach (var colName in Inputs)
                ctx.SaveNonEmptyString(colName);

            Host.CheckNonEmpty(Outputs, nameof(Outputs));
            ctx.Writer.Write(Outputs.Length);
            foreach (var colName in Outputs)
                ctx.SaveNonEmptyString(colName);

            // Save custom-provided shapes. Those shapes overwrite shapes loaded from the ONNX model file.
            int customShapeInfosLength = _options.CustomShapeInfos != null ? _options.CustomShapeInfos.Length : 0;
            ctx.Writer.Write(customShapeInfosLength);
            for (int i = 0; i < customShapeInfosLength; ++i)
            {
                var info = _options.CustomShapeInfos[i];
                ctx.SaveNonEmptyString(info.Name);
                ctx.Writer.WriteIntArray(info.Shape);
            }

            ctx.Writer.Write(_options.RecursionLimit);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema);

        /// <summary>
        /// This design assumes that all unknown dimensions are 1s. It also convert scalar shape [] in ONNX to [1].
        /// [TODO] We should infer the unknown shape from input data instead of forcing them to be 1.
        /// </summary>
        private static IEnumerable<int> AdjustDimensions(OnnxShape shape)
        {
            if (shape.Count > 0)
            {
                return shape.Select(x => (x <= 0) ? 1 : x);
            }
            return new[] { 1 };
        }

        /// <summary>
        /// In the case that the ML.Net user wants a subset of columns or lists the columns in a different order then specified in the ONNX model,
        /// we need to map from the ML.Net dataview column index to the ONNX model output index. This method does that mapping.
        /// </summary>
        /// <param name="iinfo">The index of the ML.Net column requested.</param>
        /// <returns>The index of ONNX output.</returns>
        internal int MapDataViewColumnToOnnxOutputTensor(int iinfo)
        {
            return Model.ModelInfo.OutputNames.IndexOf(Outputs[iinfo]);
        }

        private bool _isDisposed;
        public void Dispose()
        {
            if (_isDisposed)
                return;
            Model?.Dispose();
            _isDisposed = true;
        }

        private sealed class Mapper : MapperBase
        {
            private readonly OnnxTransformer _parent;
            /// <summary>
            /// <see cref="_inputColIndices"/>'s i-th element value tells the <see cref="IDataView"/> column index to
            /// find the i-th ONNX input.
            /// </summary>
            private readonly int[] _inputColIndices;
            /// <summary>
            /// <see cref="_inputTensorShapes"/>'s i-th element value tells if the i-th ONNX input's shape if it's a tensor.
            /// </summary>
            private readonly OnnxShape[] _inputTensorShapes;
            /// <summary>
            /// <see cref="_inputOnnxTypes"/>'s i-th element value tells if the <see cref="Type"/> of the i-th ONNX input.
            /// </summary>
            private readonly Type[] _inputOnnxTypes;

            public Mapper(OnnxTransformer parent, DataViewSchema inputSchema) :
                 base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {

                _parent = parent;
                _inputColIndices = new int[_parent.Inputs.Length];
                _inputTensorShapes = new OnnxShape[_parent.Inputs.Length];
                _inputOnnxTypes = new Type[_parent.Inputs.Length];

                var model = _parent.Model;
                for (int i = 0; i < _parent.Inputs.Length; i++)
                {
                    var inputNodeInfo = model.ModelInfo.GetInput(_parent.Inputs[i]);

                    var shape = inputNodeInfo.Shape;

                    var inputShape = AdjustDimensions(inputNodeInfo.Shape);
                    _inputTensorShapes[i] = inputShape.ToList();
                    _inputOnnxTypes[i] = inputNodeInfo.TypeInOnnxRuntime;

                    var col = inputSchema.GetColumnOrNull(_parent.Inputs[i]);
                    if (!col.HasValue)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i]);

                    _inputColIndices[i] = col.Value.Index;

                    var type = inputSchema[_inputColIndices[i]].Type;
                    var vectorType = type as VectorDataViewType;

                    if (vectorType != null && vectorType.Size == 0)
                        throw Host.Except($"Variable length input columns not supported");

                    var itemType = type.GetItemType();
                    var nodeItemType = inputNodeInfo.DataViewType.GetItemType();
                    if (itemType != nodeItemType)
                    {
                        // If the ONNX model input node expects a type that mismatches with the type of the input IDataView column that is provided
                        // then throw an exception.
                        // This is done except in the case where the ONNX model input node expects a UInt32 but the input column is actually KeyDataViewType
                        // This is done to support a corner case originated in NimbusML. For more info, see: https://github.com/microsoft/NimbusML/issues/426
                        var isKeyType = itemType is KeyDataViewType;
                        if (!isKeyType || itemType.RawType != nodeItemType.RawType)
                            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i], inputNodeInfo.DataViewType.GetItemType().ToString(), type.ToString());
                    }

                    // If the column is one dimension we make sure that the total size of the Onnx shape matches.
                    // Compute the total size of the known dimensions of the shape.
                    int valCount = inputShape.Where(x => x > 0).Aggregate((x, y) => x * y);
                    // The column length should be divisible by this, so that the other dimensions can be integral.
                    int typeValueCount = type.GetValueCount();
                    if (typeValueCount % valCount != 0)
                        throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {String.Join(",", inputShape)}, but input data is of length {typeValueCount}.");
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var stdSuffix = ".output";
                var info = new DataViewSchema.DetachedColumn[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                {
                    var onnxOutputName = _parent.Outputs[i];
                    var columnName = onnxOutputName.EndsWith(stdSuffix) ? onnxOutputName.Replace(stdSuffix, "") : onnxOutputName;

                    var builder = new DataViewSchema.Annotations.Builder();
                    AddSlotNames(columnName, builder);

                    info[i] = new DataViewSchema.DetachedColumn(columnName, _parent.OutputTypes[i], builder.ToAnnotations());
                }
                return info;
            }

            private void AddSlotNames(string columnName, DataViewSchema.Annotations.Builder builder)
            {
                var graph = _parent.Model.Graph;
                var nodes = graph.Node;

                var slotNamesNodeName = $"mlnet.{columnName}.SlotNames";
                var slotsNode = nodes.FirstOrDefault(node => node.Name == slotNamesNodeName);
                var slotsAttr = slotsNode?.Attribute.FirstOrDefault(attr => attr.Name == "keys_strings");
                if (slotsAttr == null)
                    return;

                int count = slotsAttr.Strings.Count();
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    var dstEditor = VBufferEditor.Create(ref dst, count);
                    for (int i = 0; i < count; i++)
                    {
                        dstEditor.Values[i] = slotsAttr.Strings[i].ToString(Encoding.UTF8).AsMemory();
                    }
                    dst = dstEditor.Commit();
                };

                builder.AddSlotNames(count, getter);
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
                => throw new NotImplementedException("This should never be called!");

            private Delegate CreateGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, OnnxRuntimeOutputCacher outputCacher)
            {
                Host.AssertValue(input);

                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();

                if (_parent.Model.ModelInfo.OutputsInfo[_parent.MapDataViewColumnToOnnxOutputTensor(iinfo)].DataViewType is VectorDataViewType vectorType)
                {
                    var elemRawType = vectorType.ItemType.RawType;
                    var srcNamedValueGetters = GetNamedOnnxValueGetters(input, _inputColIndices, _inputOnnxTypes, _inputTensorShapes);
                    if (vectorType.ItemType is TextDataViewType)
                        return MakeStringTensorGetter(input, iinfo, srcNamedValueGetters, activeOutputColNames, outputCacher);
                    else
                        return Utils.MarshalInvoke(MakeTensorGetter<int>, elemRawType, input, iinfo, srcNamedValueGetters, activeOutputColNames, outputCacher);
                }
                else
                {
                    var type = _parent.Model.ModelInfo.OutputsInfo[_parent.MapDataViewColumnToOnnxOutputTensor(iinfo)].DataViewType.RawType;
                    var srcNamedValueGetters = GetNamedOnnxValueGetters(input, _inputColIndices, _inputOnnxTypes, _inputTensorShapes);
                    return Utils.MarshalInvoke(MakeObjectGetter<int>, type, input, iinfo, srcNamedValueGetters, activeOutputColNames, outputCacher);
                }
            }

            public override Delegate[] CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.Assert(input.Schema == InputSchema);

                OnnxRuntimeOutputCacher outputCacher = new OnnxRuntimeOutputCacher();

                int n = OutputColumns.Value.Length;
                var result = new Delegate[n];
                for (int i = 0; i < n; i++)
                {
                    if (!activeOutput(i))
                        continue;
                    result[i] = CreateGetter(input, i, activeOutput, outputCacher);
                }
                disposer = () =>
                {
                    outputCacher.Dispose();
                };
                return result;
            }

            private sealed class OnnxRuntimeOutputCacher : IDisposable
            {
                public long Position;
                public Dictionary<string, DisposableNamedOnnxValue> Outputs;
                public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> OutputOnnxValues;

                public OnnxRuntimeOutputCacher()
                {
                    Position = -1;
                    Outputs = new Dictionary<string, DisposableNamedOnnxValue>();
                }

                private bool _isDisposed;

                public void Dispose()
                {
                    if (_isDisposed)
                        return;
                    OutputOnnxValues?.Dispose();
                    _isDisposed = true;
                }
            }

            private void UpdateCacheIfNeeded(long position, INamedOnnxValueGetter[] srcNamedOnnxValueGetters, List<string> activeOutputColNames, OnnxRuntimeOutputCacher outputCache)
            {
                if (outputCache.Position != position)
                {
                    var inputNameOnnxValues = new List<NamedOnnxValue>();

                    for (int i = 0; i < _inputColIndices.Length; i++)
                    {
                        inputNameOnnxValues.Add(srcNamedOnnxValueGetters[i].GetNamedOnnxValue());
                    }

                    outputCache.OutputOnnxValues?.Dispose();
                    outputCache.OutputOnnxValues = _parent.Model.Run(inputNameOnnxValues, activeOutputColNames);
                    Contracts.Assert(outputCache.OutputOnnxValues.Count > 0);

                    foreach (var outputNameOnnxValue in outputCache.OutputOnnxValues)
                    {
                        outputCache.Outputs[outputNameOnnxValue.Name] = outputNameOnnxValue;
                    }
                    outputCache.Position = position;
                }
            }

            private Delegate MakeTensorGetter<T>(DataViewRow input, int iinfo, INamedOnnxValueGetter[] srcNamedValueGetters,
                string[] activeOutputColNames, OnnxRuntimeOutputCacher outputCacher)
            {
                Host.AssertValue(input);
                var listActiveOutputColumns = activeOutputColNames.ToList();
                ValueGetter<VBuffer<T>> valueGetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcNamedValueGetters, listActiveOutputColumns, outputCacher);
                    var namedOnnxValue = outputCacher.Outputs[_parent.Outputs[iinfo]];
                    var tensor = namedOnnxValue.AsTensor<T>() as Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<T>;
                    if (tensor == null)
                        throw Host.Except($"Output column {namedOnnxValue.Name} doesn't contain a DenseTensor of expected type {typeof(T)}");
                    var editor = VBufferEditor.Create(ref dst, (int)tensor.Length);
                    tensor.Buffer.Span.CopyTo(editor.Values);
                    dst = editor.Commit();
                };
                return valueGetter;
            }

            private Delegate MakeStringTensorGetter(DataViewRow input, int iinfo, INamedOnnxValueGetter[] srcNamedValueGetters,
                string[] activeOutputColNames, OnnxRuntimeOutputCacher outputCacher)
            {
                Host.AssertValue(input);
                var listActiveOutputColumns = activeOutputColNames.ToList();

                ValueGetter<VBuffer<ReadOnlyMemory<char>>> valueGetter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcNamedValueGetters, listActiveOutputColumns, outputCacher);
                    var namedOnnxValue = outputCacher.Outputs[_parent.Outputs[iinfo]];
                    var tensor = namedOnnxValue.AsTensor<string>() as Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<string>;
                    if (tensor == null)
                        throw Host.Except($"Output column {namedOnnxValue.Name} doesn't contain a DenseTensor of expected type {typeof(string)}");

                    // Create VBufferEditor to fill "dst" with the values in "denseTensor".
                    var editor = VBufferEditor.Create(ref dst, (int)tensor.Length);
                    for (int i = 0; i < tensor.Length; ++i)
                        // Cast because string in ML.NET is typed to ReadOnlyMemory<char>.
                        editor.Values[i] = tensor.GetValue(i).AsMemory();
                    dst = editor.Commit();
                };
                return valueGetter;
            }

            private Delegate MakeObjectGetter<T>(DataViewRow input, int iinfo, INamedOnnxValueGetter[] srcNamedValueGetters,
                string[] activeOutputColNames, OnnxRuntimeOutputCacher outputCacher)
            {
                Host.AssertValue(input);
                var listActiveOutputColumns = activeOutputColNames.ToList();

                ValueGetter<T> valueGetter = (ref T dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcNamedValueGetters, listActiveOutputColumns, outputCacher);
                    var namedOnnxValue = outputCacher.Outputs[_parent.Outputs[iinfo]];
                    var trueValue = namedOnnxValue.AsEnumerable<NamedOnnxValue>().Select(value => value.AsDictionary<string, float>());
                    var caster = _parent.Model.ModelInfo.OutputsInfo[_parent.MapDataViewColumnToOnnxOutputTensor(iinfo)].Caster;
                    dst = (T)caster(namedOnnxValue);
                };
                return valueGetter;
            }

            /// <summary>
            /// Helper function to wrap ML.NET getters to produce ONNXRuntime variables.
            /// For each required input of the ONNX model, there will be a <see cref="INamedOnnxValueGetter"/>,
            /// which first invokes a ML.NET getter and casts the obtained value to <see cref="NamedOnnxValue"/>.
            /// </summary>
            private static INamedOnnxValueGetter[] GetNamedOnnxValueGetters(DataViewRow input,
                int[] inputColIndices,
                Type[] onnxInputTypes,
                OnnxShape[] onnxInputShapes)
            {
                var srcNamedOnnxValueGetters = new INamedOnnxValueGetter[inputColIndices.Length];
                for (int i = 0; i < inputColIndices.Length; i++)
                {
                    int colIndex = inputColIndices[i];
                    var isVector = input.Schema[colIndex].Type is VectorDataViewType;
                    if (!isVector)
                        srcNamedOnnxValueGetters[i] = CreateNamedOnnxValueGetter(input, onnxInputTypes[i], colIndex, onnxInputShapes[i]);
                    else
                        srcNamedOnnxValueGetters[i] = CreateNamedOnnxValueGetterVec(input, onnxInputTypes[i], colIndex, onnxInputShapes[i]);
                }
                return srcNamedOnnxValueGetters;
            }

            /// <summary>
            /// Wrap ML.NET getter to produce NamedOnnxValue. The wrapper is used to fetch non-vector ML.NET column and cast ML.NET column to
            /// NamedOnnxValue which is consumable by ONNXRuntime.
            /// </summary>
            private static INamedOnnxValueGetter CreateNamedOnnxValueGetter(DataViewRow input, Type onnxType, int colIndex, OnnxShape onnxShape)
            {
                // This type is column type in ML.NET used to invoke ML.NET
                // getter, so we use just use the type provided by the input's Schema.
                // This function handles non-tensor types, so we directly access RawType.
                // For tensor types, we need to do GetItemType().RawType.
                var type = input.Schema[colIndex].Type.RawType;
                Contracts.AssertValue(type);
                return Utils.MarshalInvoke(CreateNamedOnnxValueGetterCore<int>, type, input, colIndex, onnxShape);
            }

            /// <summary>
            /// Function needed by reflection in <see cref="CreateNamedOnnxValueGetter(DataViewRow, Type, int, OnnxShape)"/>.
            /// </summary>
            private static INamedOnnxValueGetter CreateNamedOnnxValueGetterCore<T>(DataViewRow input, int colIndex, OnnxShape onnxShape)
            {
                return new NameOnnxValueGetter<T>(input, colIndex);
            }

            /// <summary>
            /// Wrap ML.NET getter to produce NamedOnnxValue. The wrapper is used to fetch vector-typed ML.NET column and cast ML.NET column to
            /// NamedOnnxValue which is consumable by ONNXRuntime.
            /// </summary>
            private static INamedOnnxValueGetter CreateNamedOnnxValueGetterVec(DataViewRow input, Type onnxType, int colIndex, OnnxShape onnxShape)
            {
                // This type is column type in ML.NET used to invoke ML.NET
                // getter, so we use just use the type provided by the input's Schema.
                // This function handles tensor types, so we need to call GetItemType()
                // to get the element type in VBuffer.
                var type = input.Schema[colIndex].Type.GetItemType().RawType;
                Contracts.AssertValue(type);
                return Utils.MarshalInvoke(CreateNamedOnnxValueGetterVecCore<int>, type, input, colIndex, onnxShape);
            }

            /// <summary>
            /// Function needed by reflection in <see cref="CreateNamedOnnxValueGetterVec(DataViewRow, Type, int, OnnxShape)"/>.
            /// </summary>
            private static INamedOnnxValueGetter CreateNamedOnnxValueGetterVecCore<T>(DataViewRow input, int colIndex, OnnxShape onnxShape)
            {
                return new NamedOnnxValueGetterVec<T>(input, colIndex, onnxShape);
            }

            /// <summary>
            /// Common function for wrapping ML.NET getter as a NamedOnnxValue getter.
            /// </summary>
            private interface INamedOnnxValueGetter
            {
                NamedOnnxValue GetNamedOnnxValue();
            }

            private class NameOnnxValueGetter<T> : INamedOnnxValueGetter
            {
                private readonly ValueGetter<T> _srcGetter;
                private readonly string _colName;

                public NameOnnxValueGetter(DataViewRow input, int colIndex)
                {
                    _colName = input.Schema[colIndex].Name;
                    _srcGetter = input.GetGetter<T>(input.Schema[colIndex]);
                }
                public NamedOnnxValue GetNamedOnnxValue()
                {
                    var scalar = default(T);
                    _srcGetter(ref scalar);
                    return OnnxUtils.CreateScalarNamedOnnxValue(_colName, scalar);
                }
            }

            private class NamedOnnxValueGetterVec<T> : INamedOnnxValueGetter
            {
                private readonly ValueGetter<VBuffer<T>> _srcGetter;
                private readonly OnnxShape _tensorShape;
                private readonly string _colName;
                private VBuffer<T> _vBuffer;
                private VBuffer<T> _vBufferDense;
                public NamedOnnxValueGetterVec(DataViewRow input, int colIndex, OnnxShape tensorShape)
                {
                    _srcGetter = input.GetGetter<VBuffer<T>>(input.Schema[colIndex]);
                    _tensorShape = tensorShape;
                    _colName = input.Schema[colIndex].Name;
                    _vBuffer = default;
                    _vBufferDense = default;
                }
                public NamedOnnxValue GetNamedOnnxValue()
                {
                    _srcGetter(ref _vBuffer);
                    _vBuffer.CopyToDense(ref _vBufferDense);
                    return OnnxUtils.CreateNamedOnnxValue(_colName, _vBufferDense.GetValues(), _tensorShape);
                }
            }
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> for scoring ONNX models in the ML.NET framework.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | Known-sized vector of <xref:System.Single> or <xref:System.Double> types |
    /// | Output column data type | As specified by the ONNX model |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.OnnxTransformer (always),  either Microsoft.ML.OnnxRuntime 1.6.0 (for CPU processing) or Microsoft.ML.OnnxRuntime.Gpu 1.6.0 (for GPU processing if GPU is available) |
    /// | Exportable to ONNX | No |
    ///
    /// To create this estimator use the following APIs:
    /// [ApplyOnnxModel](xref:Microsoft.ML.OnnxCatalog.ApplyOnnxModel*)
    ///
    /// Supports inferencing of models in ONNX 1.6 format (opset 11), using the [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/) library.
    /// Models are scored on CPU if the project references Microsoft.ML.OnnxRuntime and on the GPU if the project references Microsoft.ML.OnnxRuntime.Gpu.
    /// Every project using the OnnxScoringEstimator must reference one of the above two packages.
    ///
    /// To run on a GPU, use the
    /// NuGet package [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/) instead of the Microsoft.ML.OnnxRuntime nuget (which is for CPU processing). Microsoft.ML.OnnxRuntime.Gpu
    /// requires a [CUDA supported GPU](https://developer.nvidia.com/cuda-gpus#compute), the [CUDA 10.2 Toolkit](https://developer.nvidia.com/cuda-downloads), and [cuDNN 8.0.3](https://developer.nvidia.com/cudnn) (as indicated on [Onnxruntime's documentation](https://github.com/Microsoft/onnxruntime#system-requirements)).
    /// When creating the estimator through [ApplyOnnxModel](xref:Microsoft.ML.OnnxCatalog.ApplyOnnxModel*), set the parameter 'gpuDeviceId' to a valid non-negative integer. Typical device ID values are 0 or 1. If the GPU device isn't found but 'fallbackToCpu = true' then the estimator will run on the CPU. If the GPU device isn't found but 'fallbackToCpu = false' then the estimator will throw an exception
    ///
    /// The inputs and outputs of the ONNX models must be Tensor type. Sequence and Maps are not yet supported.
    ///
    /// Internally, OnnxTransformer (the return value of OnnxScoringEstimator.Fit()) holds a reference to an inference session which points to unmanaged memory owned by OnnxRuntime.dll.
    /// Whenever there is a call to [ApplyOnnxModel](xref:Microsoft.ML.OnnxCatalog.ApplyOnnxModel*) in a pipeline, it is advised to cast the return value of the Fit() call to IDisposable and call Dispose() to ensure that there are no memory leaks.
    ///
    /// OnnxRuntime works on Windows, MacOS and Ubuntu 16.04 Linux 64-bit platforms.
    /// Visit [ONNX Models](https://github.com/onnx/models) to see a list of readily available models to get started with.
    /// Refer to [ONNX](http://onnx.ai) for more information.
    ///
    /// ]]>
    /// </format>
    /// </remarks>
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
        /// <param name="shapeDictionary"></param>
        /// <param name="recursionLimit">Optional, specifies the Protobuf CodedInputStream recursion limit. Default value is 100.</param>
        [BestFriend]
        internal OnnxScoringEstimator(IHostEnvironment env, string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false,
            IDictionary<string, int[]> shapeDictionary = null, int recursionLimit = 100)
            : this(env, new OnnxTransformer(env, new string[] { }, new string[] { }, modelFile, gpuDeviceId, fallbackToCpu, shapeDictionary, recursionLimit))
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
        /// <param name="shapeDictionary"></param>
        /// <param name="recursionLimit">Optional, specifies the Protobuf CodedInputStream recursion limit. Default value is 100.</param>
        /// <param name="interOpNumThreads">Controls the number of threads used to parallelize the execution of the graph (across nodes).</param>
        /// <param name="intraOpNumThreads">Controls the number of threads to use to run the model.</param>
        internal OnnxScoringEstimator(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, string modelFile,
            int? gpuDeviceId = null, bool fallbackToCpu = false, IDictionary<string, int[]> shapeDictionary = null, int recursionLimit = 100,
            int? interOpNumThreads = null, int? intraOpNumThreads = null)
           : this(env, new OnnxTransformer(env, outputColumnNames, inputColumnNames, modelFile, gpuDeviceId, fallbackToCpu, shapeDictionary, recursionLimit, interOpNumThreads, intraOpNumThreads))
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

            // This loop checks if all input columns needed in the underlying transformer can be found
            // in inputSchema.
            // Since ML.NET can only produces tensors (scalars are converted to tensor with shape [1] before feeding
            // ML.NET them into ONNXRuntime), the bridge code in ONNX Transformer assumes that all inputs are tensors.
            for (var i = 0; i < Transformer.Inputs.Length; i++)
            {
                // Get the i-th IDataView input column's name in the underlying ONNX transformer.
                var input = Transformer.Inputs[i];

                // Make sure inputSchema contains the i-th input column.
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);

                // Make sure that the input columns in inputSchema are fixed shape tensors.
                if (col.Kind == SchemaShape.Column.VectorKind.VariableVector)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector", col.GetTypeString());

                var inputsInfo = Transformer.Model.ModelInfo.InputsInfo;
                var idx = Transformer.Model.ModelInfo.InputNames.IndexOf(input);
                if (idx < 0)
                    throw Host.Except($"Column {input} doesn't match input node names of model.");

                var inputNodeInfo = inputsInfo[idx];
                var expectedType = ((VectorDataViewType)inputNodeInfo.DataViewType).ItemType;
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
