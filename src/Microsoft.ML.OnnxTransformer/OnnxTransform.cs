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
    /// </summary>
    public sealed class OnnxTransformer : RowToRowTransformerBase // MYTODO: Should I consider not to inherit from this, since now OnnxTransformer would be able to drop columns and not use the RowToRowMapperTransform?
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
                verWrittenCur: 0x00010002,
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
            loaderAssemblyName: typeof(OnnxTransformer).Assembly.FullName);
        }

        // Factory method for SignatureDataTransform
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            var transformer = new OnnxTransformer(env, options);
            var mapper = new Mapper(transformer, input.Schema);
            return new OnnxDataTransform(env, input, mapper);
        }

        // Factory method for SignatureLoadDataTransform
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            var transformer = OnnxTransformer.Create(env, ctx);
            var mapper = new Mapper(transformer, input.Schema);
            return new OnnxDataTransform(env, input, mapper);
        }

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

            var options = new Options() { InputColumns = inputs, OutputColumns = outputs, CustomShapeInfos = loadedCustomShapeInfos };

            return new OnnxTransformer(env, options, modelBytes);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema) => Create(env, ctx).MakeRowMapper(inputSchema); // MYTODO: In what scenario is this called? Should I worry that the mapper, only by itself, isn't capable of dropping columns?

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
                foreach(var customShape in options.CustomShapeInfos)
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
                    Model = new OnnxModel(options.ModelFile, options.GpuDeviceId, options.FallbackToCpu, ownModelFile: false, shapeDictionary: shapeDictionary);
                }
                else
                {
                    // Entering this region means that the byte[] is passed as the model. To feed that byte[] to ONNXRuntime, we need
                    // to create a temporal file to store it and then call ONNXRuntime's API to load that file.
                    Model = OnnxModel.CreateFromBytes(modelBytes, options.GpuDeviceId, options.FallbackToCpu, shapeDictionary: shapeDictionary);
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
        internal OnnxTransformer(IHostEnvironment env, string modelFile, int? gpuDeviceId = null,
            bool fallbackToCpu = false, IDictionary<string, int[]> shapeDictionary = null)
            : this(env, new Options()
            {
                ModelFile = modelFile,
                InputColumns = new string[] { },
                OutputColumns = new string[] { },
                GpuDeviceId = gpuDeviceId,
                FallbackToCpu = fallbackToCpu,
                CustomShapeInfos = shapeDictionary?.Select(pair => new CustomShapeInfo(pair.Key, pair.Value)).ToArray()
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
        internal OnnxTransformer(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false,
             IDictionary<string, int[]> shapeDictionary = null)
            : this(env, new Options()
            {
                ModelFile = modelFile,
                InputColumns = inputColumnNames,
                OutputColumns = outputColumnNames,
                GpuDeviceId = gpuDeviceId,
                FallbackToCpu = fallbackToCpu,
                CustomShapeInfos = shapeDictionary?.Select(pair => new CustomShapeInfo(pair.Key, pair.Value)).ToArray()
            })
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.SaveBinaryStream("OnnxModel", w => { w.WriteByteArray(File.ReadAllBytes(Model.ModelFile)); });

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
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema); // MYTODO: Could I erase this? If I stop inheriting from RTRTB?

        protected override IRowToRowMapper GetRowToRowMapperCore(DataViewSchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            return new OnnxDataTransform(Host, new EmptyDataView(Host, inputSchema), new Mapper(this, inputSchema));
        }

        protected override DataViewSchema GetOutputSchemaCore(DataViewSchema inputSchema)
        {
            return OnnxDataTransform.GetOutputSchema(inputSchema, new Mapper(this, inputSchema));
        }

        private protected override IDataView MakeDataTransformCore(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return new OnnxDataTransform(Host, input, new Mapper(this, input.Schema));
        }

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
        /// In order to fully support onnx exportability from <see cref="ColumnSelectingTransformer"/>, it was decided
        /// that the <see cref="OnnxTransformer"/> should drop all columns that are used as input of the Onnx model,
        /// from the input schema.
        ///
        /// Any column that was already inside the input schema, but which isn't used by the onnx model itself,
        /// should simply propagate to the output.
        /// </summary>
        internal string[] GetDropColumnsNames()
        {
            return Model.ModelInfo.InputNames.ToArray();
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
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema),"input", _parent.Inputs[i]);

                    _inputColIndices[i] = col.Value.Index;

                    var type = inputSchema[_inputColIndices[i]].Type;
                    var vectorType = type as VectorDataViewType;

                    if (vectorType != null && vectorType.Size == 0)
                        throw Host.Except($"Variable length input columns not supported");

                    if (type.GetItemType() != inputNodeInfo.DataViewType.GetItemType())
                    {
                        // If the ONNX model input node expects a type that mismatches with the type of the input IDataView column that is provided
                        // then throw an exception.
                        // This is done except in the case where the ONNX model input node expects a UInt32 but the input column is actually KeyDataViewType
                        // This is done to support a corner case originated in NimbusML. For more info, see: https://github.com/microsoft/NimbusML/issues/426
                        if (!(type.GetItemType() is KeyDataViewType && inputNodeInfo.DataViewType.GetItemType().RawType == typeof(UInt32)))
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
            {
                disposer = null;
                Host.AssertValue(input);

                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();

                if (_parent.Model.ModelInfo.OutputsInfo[iinfo].DataViewType is VectorDataViewType vectorType)
                {
                    var elemRawType = vectorType.ItemType.RawType;
                    var srcNamedValueGetters = GetNamedOnnxValueGetters(input, _inputColIndices, _inputOnnxTypes, _inputTensorShapes);
                    if (vectorType.ItemType is TextDataViewType)
                        return MakeStringTensorGetter(input, iinfo, srcNamedValueGetters, activeOutputColNames);
                    else
                        return Utils.MarshalInvoke(MakeTensorGetter<int>, elemRawType, input, iinfo, srcNamedValueGetters, activeOutputColNames);
                }
                else
                {
                    var type = _parent.Model.ModelInfo.OutputsInfo[iinfo].DataViewType.RawType;
                    var srcNamedValueGetters = GetNamedOnnxValueGetters(input, _inputColIndices, _inputOnnxTypes, _inputTensorShapes);
                    return Utils.MarshalInvoke(MakeObjectGetter<int>, type, input, iinfo, srcNamedValueGetters, activeOutputColNames);
                }
            }

            private class OnnxRuntimeOutputCacher
            {
                public long Position;
                public Dictionary<string, NamedOnnxValue> Outputs;
                public OnnxRuntimeOutputCacher()
                {
                    Position = -1;
                    Outputs = new Dictionary<string, NamedOnnxValue>();
                }
            }

            private void UpdateCacheIfNeeded(long position, INamedOnnxValueGetter[] srcNamedOnnxValueGetters, string[] activeOutputColNames, OnnxRuntimeOutputCacher outputCache)
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

            private Delegate MakeTensorGetter<T>(DataViewRow input, int iinfo, INamedOnnxValueGetter[] srcNamedValueGetters, string[] activeOutputColNames)
            {
                Host.AssertValue(input);
                var outputCacher = new OnnxRuntimeOutputCacher();
                ValueGetter<VBuffer<T>> valueGetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcNamedValueGetters, activeOutputColNames, outputCacher);
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

            private Delegate MakeStringTensorGetter(DataViewRow input, int iinfo, INamedOnnxValueGetter[] srcNamedValueGetters, string[] activeOutputColNames)
            {
                Host.AssertValue(input);
                var outputCacher = new OnnxRuntimeOutputCacher();
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> valueGetter = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcNamedValueGetters, activeOutputColNames, outputCacher);
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

            private Delegate MakeObjectGetter<T>(DataViewRow input, int iinfo, INamedOnnxValueGetter[] srcNamedValueGetters, string[] activeOutputColNames)
            {
                Host.AssertValue(input);
                var outputCache = new OnnxRuntimeOutputCacher();
                ValueGetter<T> valueGetter = (ref T dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcNamedValueGetters, activeOutputColNames, outputCache);
                    var namedOnnxValue = outputCache.Outputs[_parent.Outputs[iinfo]];
                    var trueValue = namedOnnxValue.AsEnumerable<NamedOnnxValue>().Select(value => value.AsDictionary<string, float>());
                    var caster = _parent.Model.ModelInfo.OutputsInfo[iinfo].Caster;
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

            /// <summary>
            /// <see cref="OnnxTransformer.GetDropColumnsNames"/>
            /// </summary>
            public string[] GetDropColumnsNames()
            {
                return _parent.GetDropColumnsNames();
            }
        }

        /// <summary>
        /// Similar to <see cref="ColumnBindings"/>, but this class will enable dropping columns from the input
        /// schema, in order to let OnnxTransformer support the <see cref="ColumnSelectingTransformer"/> onnx export.
        /// </summary>
        [BestFriend] // MYTODO: Is this necessary?
        internal sealed class Bindings // MYTODO: Should I move this inside OnnxDataTransform?
        {
            // MYTODO: Should I simply inherit from ColumnBindings, since everything is the same except for the constructor (specifically, only, the way it created the _colMap)?

            // Indices of columns in the merged schema. Old indices are as is, new indices are stored as ~idx.
            private readonly int[] _colMap;

            /// <summary>
            /// The indices of added columns in the <see cref="Schema"/>.
            /// </summary>
            public IReadOnlyList<int> AddedColumnIndices { get; }

            /// <summary>
            /// The input schema.
            /// </summary>
            public DataViewSchema InputSchema { get; }

            /// <summary>
            /// The merged schema.
            /// </summary>
            public DataViewSchema Schema { get; }

            /// <summary>
            /// Create a new instance of <see cref="Bindings"/>.
            /// </summary>
            /// <param name="input">The input schema that we're adding columns to.</param>
            /// <param name="dropColumnsNames">Names of the columns to drop, so that they don't propagate from the input schema</param>
            /// <param name="addedColumns">The columns being added.</param>
            public Bindings(DataViewSchema input, List<string> dropColumnsNames, DataViewSchema.DetachedColumn[] addedColumns)
            {
                Contracts.CheckValue(input, nameof(input));
                Contracts.CheckValue(addedColumns, nameof(addedColumns));

                InputSchema = input;

                // Construct the indices.
                // Drop the indicated columns
                // And drop all hidden columns
                var indices = new List<int>();
                var namesUsed = new HashSet<string>();
                for (int i = 0; i < input.Count; i++)
                {
                    if (InputSchema[i].IsHidden || dropColumnsNames.Contains(InputSchema[i].Name)) // MYTODO: Should I drop all hidden columns? Only the ones that are inside the dropColumnsNames list?
                        continue;

                    namesUsed.Add(input[i].Name);
                    indices.Add(i);
                }

                for (int i = 0; i < addedColumns.Length; i++)
                {
                    string name = addedColumns[i].Name;
                    if (namesUsed.Add(name))
                    {
                        // New name. Append to the end.
                        indices.Add(~i);
                    }
                    else
                    {
                        // Old name. Find last instance and add after it.
                        for (int j = indices.Count - 1; j >= 0; j--)
                        {
                            var colName = indices[j] >= 0 ? input[indices[j]].Name : addedColumns[~indices[j]].Name;
                            if (colName == name)
                            {
                                indices.Insert(j + 1, ~i);
                                break;
                            }
                        }
                    }
                }

                // Contracts.Assert(indices.Count == addedColumns.Length + input.Count); // MYTODO: This assertion is no longer valid, and I can't think of a better one

                // Create the output schema.
                var schemaColumns = indices.Select(idx => idx >= 0 ? new DataViewSchema.DetachedColumn(input[idx]) : addedColumns[~idx]);
                Schema = SchemaExtensions.MakeSchema(schemaColumns);

                // Memorize column maps.
                _colMap = indices.ToArray();
                var addedIndices = new int[addedColumns.Length];
                for (int i = 0; i < _colMap.Length; i++)
                {
                    int colIndex = _colMap[i];
                    if (colIndex < 0)
                    {
                        Contracts.Assert(addedIndices[~colIndex] == 0);
                        addedIndices[~colIndex] = i;
                    }
                }

                AddedColumnIndices = addedIndices.AsReadOnly();
            }

            /// <summary>
            /// This maps a column index for this schema to either a source column index (when
            /// <paramref name="isSrcColumn"/> is true), or to an "iinfo" index of an added column
            /// (when <paramref name="isSrcColumn"/> is false).
            /// </summary>
            /// <param name="isSrcColumn">Whether the return index is for a source column</param>
            /// <param name="col">The column index for this schema</param>
            /// <returns>The index (either source index or iinfo index)</returns>
            public int MapColumnIndex(out bool isSrcColumn, int col)
            {
                Contracts.Assert(0 <= col && col < _colMap.Length);
                int index = _colMap[col];
                if (index < 0)
                {
                    index = ~index;
                    Contracts.Assert(index < AddedColumnIndices.Count);
                    isSrcColumn = false;
                }
                else
                {
                    Contracts.Assert(index < InputSchema.Count);
                    isSrcColumn = true;
                }
                return index;
            }

            /// <summary>
            /// The given predicate maps from output column index to whether the column is active.
            /// This builds an array of bools of length Input.ColumnCount containing the results of calling
            /// predicate on the output column index corresponding to each input column index.
            /// </summary>
            public bool[] GetActiveInput(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = new bool[InputSchema.Count];
                for (int dst = 0; dst < _colMap.Length; dst++)
                {
                    int src = _colMap[dst];
                    Contracts.Assert(-AddedColumnIndices.Count <= src && src < InputSchema.Count);
                    if (src >= 0 && predicate(dst))
                        active[src] = true;
                }
                return active;
            }
        }

        private class OnnxDataTransform : RowToRowTransformBase, IRowToRowMapper
        {
            // MYTODO: Is it even worth it to have this OnnxDataTransform class when it (including the RowImpl and Cursor)
            // are identical to the RowToRowMapperTransform? The differences are:
            // - This one expects specifically a OnnxTransformer.Mapper as _mapper from where to get the GetColumnsNames, whereas RTRMT expects a generic IRowMapper
            // - This one has a _bindings object which is off type OnnxTransformer.Bindings, whereas RTRMT expects a generic ColumnsBindings
            // - This one in here has a differend override for the Save method
            // - This one in here doesn't have (but I don't know if it could have) methods related to SaveOnnx, SavePfa, ApplyToData, and VersionInfo of RTRMT.
            // - RTRMT has an extra member called "_mapperFactory" that is used in ApplyToData

            private protected override void SaveModel(ModelSaveContext ctx) => (_mapper as IRowMapper).Save(ctx); // MYTODO: This is the only thing that differ between this and RTRMT. Wonder if it would work if I used theirs instead?

            private readonly Mapper _mapper;
            private readonly Bindings _bindings;

            public override DataViewSchema OutputSchema => _bindings.Schema;

            public OnnxDataTransform(IHostEnvironment env, IDataView input, Mapper mapper)
                : base(env.Register(nameof(OnnxDataTransform)), input)
            {
                _mapper = mapper;
                _bindings = new Bindings(input.Schema, mapper.GetDropColumnsNames().ToList(), (mapper as IRowMapper).GetOutputColumns());
            }

            public static DataViewSchema GetOutputSchema(DataViewSchema inputSchema, Mapper mapper)
            {
                Contracts.CheckValue(inputSchema, nameof(inputSchema));
                Contracts.CheckValue(mapper, nameof(mapper));
                return new Bindings(inputSchema, mapper.GetDropColumnsNames().ToList(), (mapper as IRowMapper).GetOutputColumns()).Schema;
            }

            /// <summary>
            /// Produces the set of active columns for the data view (as a bool[] of length bindings.ColumnCount),
            /// and the needed active input columns, given a predicate for the needed active output columns.
            /// </summary>
            private bool[] GetActive(Func<int, bool> predicate, out IEnumerable<DataViewSchema.Column> inputColumns)
            {
                int n = _bindings.Schema.Count;
                var active = Utils.BuildArray(n, predicate);
                Contracts.Assert(active.Length == n);

                var activeInput = _bindings.GetActiveInput(predicate);
                Contracts.Assert(activeInput.Length == _bindings.InputSchema.Count);

                // Get a predicate that determines which outputs are active.
                var predicateOut = GetActiveOutputColumns(active);

                // Now map those to active input columns.
                var predicateIn = (_mapper as IRowMapper).GetDependencies(predicateOut);

                // Combine the two sets of input columns.
                inputColumns = _bindings.InputSchema.Where(col => activeInput[col.Index] || predicateIn(col.Index));

                return active;
            }

            private Func<int, bool> GetActiveOutputColumns(bool[] active)
            {
                Contracts.AssertValue(active);
                Contracts.Assert(active.Length == _bindings.Schema.Count);

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
                if (_bindings.AddedColumnIndices.Any(predicate)) // MYTODO: This is copied from RowToRowMapperTransform. Why is this the case, and it ignores all the other columns that propagate from the input?
                    return true;
                return null;
            }

            protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
                var active = GetActive(predicate, out IEnumerable<DataViewSchema.Column> inputCols);

                return new Cursor(Host, Source.GetRowCursor(inputCols, rand), this, active);
            }

            public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                Host.CheckValueOrNull(rand);

                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
                var active = GetActive(predicate, out IEnumerable<DataViewSchema.Column> inputCols);

                var inputs = Source.GetRowCursorSet(inputCols, n, rand);
                Host.AssertNonEmpty(inputs);

                if (inputs.Length == 1 && n > 1 && _bindings.AddedColumnIndices.Any(predicate)) // MYTODO: This is copied from TowToRowMapperTransform. Shouldn't the last check actually call ShouldUseParallel?
                    inputs = DataViewUtils.CreateSplitCursors(Host, inputs[0], n);
                Host.AssertNonEmpty(inputs);

                var cursors = new DataViewRowCursor[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                    cursors[i] = new Cursor(Host, inputs[i], this, active);
                return cursors;
            }

            /// <summary>
            /// Given a set of output columns, return the input columns that are needed to generate those output columns.
            /// </summary>
            IEnumerable<DataViewSchema.Column> IRowToRowMapper.GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns)
            {
                var predicate = RowCursorUtils.FromColumnsToPredicate(dependingColumns, OutputSchema);
                GetActive(predicate, out var inputColumns);
                return inputColumns;
            }

            public DataViewSchema InputSchema => Source.Schema;

            DataViewRow IRowToRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                Host.CheckValue(input, nameof(input));
                Host.CheckValue(activeColumns, nameof(activeColumns));
                Host.Check(input.Schema == Source.Schema, "Schema of input row must be the same as the schema the mapper is bound to");

                using (var ch = Host.Start("GetEntireRow"))
                {
                    var activeArr = new bool[OutputSchema.Count];
                    foreach (var column in activeColumns)
                    {
                        Host.Assert(column.Index < activeArr.Length, $"The columns {activeColumns.Select(c => c.Name)} are not suitable for the OutputSchema.");
                        activeArr[column.Index] = true;
                    }
                    var pred = GetActiveOutputColumns(activeArr);
                    var getters = (_mapper as IRowMapper).CreateGetters(input, pred, out Action disp);

                    return new RowImpl(input, this, OutputSchema, getters, disp);
                }
            }

            // MYTODO: Should I also copy in here the ApplyToData method from RowToRowMapperTransform?

            private sealed class RowImpl : WrappingRow
            {
                private readonly Delegate[] _getters;
                private readonly OnnxDataTransform _parent;
                private readonly Action _disposer;

                public override DataViewSchema Schema { get; }

                public RowImpl(DataViewRow input, OnnxDataTransform parent, DataViewSchema schema, Delegate[] getters, Action disposer)
                    : base(input)
                {
                    _parent = parent;
                    Schema = schema;
                    _getters = getters;
                    _disposer = disposer;
                }

                protected override void DisposeCore(bool disposing)
                {
                    if (disposing)
                        _disposer?.Invoke();
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
                    bool isSrc;
                    int index = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                    if (isSrc)
                        return Input.GetGetter<TValue>(Input.Schema[index]);

                    Contracts.Assert(_getters[index] != null);
                    var fn = _getters[index] as ValueGetter<TValue>;
                    if (fn == null)
                        throw Contracts.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                    return fn;
                }

                /// <summary>
                /// Returns whether the given column is active in this row.
                /// </summary>
                public override bool IsColumnActive(DataViewSchema.Column column)
                {
                    bool isSrc;
                    int index = _parent._bindings.MapColumnIndex(out isSrc, column.Index);
                    if (isSrc)
                        return Input.IsColumnActive(Schema[index]);
                    return _getters[index] != null;
                }
            }

            private sealed class Cursor : SynchronizedCursorBase
            {
                private readonly Delegate[] _getters;
                private readonly bool[] _active;
                private readonly Bindings _bindings;
                private readonly Action _disposer;
                private bool _disposed;

                public override DataViewSchema Schema => _bindings.Schema;

                public Cursor(IChannelProvider provider, DataViewRowCursor realInput, OnnxDataTransform parent, bool[] active)
                    : base(provider, realInput)
                {
                    var pred = parent.GetActiveOutputColumns(active);
                    _getters = (parent._mapper as IRowMapper).CreateGetters(realInput, pred, out _disposer);
                    _active = active;
                    _bindings = parent._bindings;
                }

                /// <summary>
                /// Returns whether the given column is active in this row.
                /// </summary>
                public override bool IsColumnActive(DataViewSchema.Column column)
                {
                    Ch.Check(column.Index < _bindings.Schema.Count);
                    return _active[column.Index];
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
                    Ch.Check(IsColumnActive(column));

                    bool isSrc;
                    int index = _bindings.MapColumnIndex(out isSrc, column.Index);
                    if (isSrc)
                        return Input.GetGetter<TValue>(Input.Schema[index]);

                    Ch.AssertValue(_getters);
                    var getter = _getters[index];
                    Ch.Assert(getter != null);
                    var fn = getter as ValueGetter<TValue>;
                    if (fn == null)
                        throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                    return fn;
                }

                protected override void Dispose(bool disposing)
                {
                    if (_disposed)
                        return;
                    if (disposing)
                        _disposer?.Invoke();
                    _disposed = true;
                    base.Dispose(disposing);
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
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.OnnxTransformer (always),  either Microsoft.ML.OnnxRuntime 1.2.0 (for CPU processing) or Microsoft.ML.OnnxRuntime.Gpu 1.2.0 (for GPU processing if GPU is available) |
    /// | Exportable to ONNX | No |
    ///
    /// Supports inferencing of models in ONNX 1.6 format (opset 11), using the
    /// [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/) library (version 1.2.0).
    /// Models are scored on CPU by default.
    ///
    /// To run on a GPU, use the
    /// NuGet package [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/) (version 1.2.0) instead of the Microsoft.ML.OnnxRuntime nuget (which is for CPU processing). Microsoft.ML.OnnxRuntime.Gpu
    /// requires a [CUDA supported GPU](https://developer.nvidia.com/cuda-gpus#compute), the [CUDA 10.1 Toolkit](https://developer.nvidia.com/cuda-downloads), and [cuDNN 7.6.5](https://developer.nvidia.com/cudnn) (as indicated on [Onnxruntime's documentation](https://github.com/Microsoft/onnxruntime#default-gpu-cuda)).
    /// Set parameter 'gpuDeviceId' to a valid non-negative integer. Typical device ID values are 0 or 1.
    ///
    /// The inputs and outputs of the ONNX models must be Tensor type. Sequence and Maps are not yet supported.
    ///
    /// OnnxRuntime works on Windows, MacOS and Ubuntu 16.04 Linux 64-bit platforms.
    /// Visit [ONNX Models](https://github.com/onnx/models) to see a list of readily available models to get started with.
    /// Refer to [ONNX](http://onnx.ai) for more information.
    ///
    /// To create this estimator use the following:
    /// [ApplyOnnxModel](xref:Microsoft.ML.OnnxCatalog.ApplyOnnxModel*)
    ///
    /// Check the See Also section for links to usage examples.
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
        [BestFriend]
        internal OnnxScoringEstimator(IHostEnvironment env, string modelFile, int? gpuDeviceId = null, bool fallbackToCpu = false,
            IDictionary<string, int[]> shapeDictionary = null)
            : this(env, new OnnxTransformer(env, new string[] { }, new string[] { }, modelFile, gpuDeviceId, fallbackToCpu, shapeDictionary))
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
        internal OnnxScoringEstimator(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, string modelFile,
            int? gpuDeviceId = null, bool fallbackToCpu = false, IDictionary<string, int[]> shapeDictionary = null)
           : this(env, new OnnxTransformer(env, outputColumnNames, inputColumnNames, modelFile, gpuDeviceId, fallbackToCpu, shapeDictionary))
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

            var droppedInputs = new List<string>(Transformer.GetDropColumnsNames());
            var resultDic = inputSchema.Where(col => !droppedInputs.Contains(col.Name)).ToDictionary(x => x.Name); // MYTODO: Is this enough? Does SchemaShape should also worry about "hidden" columns?

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
