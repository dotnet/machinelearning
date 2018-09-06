// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
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

// This is for de-serialization from a binary model file.
[assembly: LoadableClass(typeof(TensorFlowTransform.TensorFlowMapper), null, typeof(SignatureLoadRowMapper),
    "", TensorFlowTransform.TensorFlowMapper.LoaderSignature)]

[assembly: EntryPointModule(typeof(TensorFlowTransform))]

namespace Microsoft.ML.Transforms
{
    public static class TensorFlowTransform
    {
        internal sealed class TensorFlowMapper : IRowMapper
        {
            private readonly IHost _host;

            /// <summary>
            /// TensorFlow session object
            /// </summary>
            private readonly TFSession _session;

            private readonly string[] _inputColNames;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isVectorInput;
            private readonly TFShape[] _tfInputShapes;
            private readonly TFDataType[] _tfInputTypes;
            private readonly bool _isFrozen;
            private readonly string _exportDir;

            private readonly string[] _outputColNames;
            private readonly ColumnType[] _outputColTypes;
            private readonly TFDataType[] _tfOutputTypes;
            private const int BatchSize = 1;
            public const string LoaderSignature = "TFMapper";
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

            public TensorFlowMapper(IHostEnvironment env, ISchema inputSchema, byte[] modelBytes, string[] inputColNames, string[] outputColNames)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("TensorFlowMapper");
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckNonEmpty(modelBytes, nameof(modelBytes));
                _host.CheckNonEmpty(inputColNames, nameof(inputColNames));
                _host.CheckNonEmpty(outputColNames, nameof(outputColNames));
                for (int i = 0; i < inputColNames.Length; i++)
                    _host.CheckNonWhiteSpace(inputColNames[i], nameof(inputColNames));
                for (int i = 0; i < outputColNames.Length; i++)
                    _host.CheckNonWhiteSpace(outputColNames[i], nameof(outputColNames));

                _isFrozen = true;
                _session = LoadTFSession(modelBytes, null);
                _host.Check(inputColNames.All(name => _session.Graph[name] != null), "One of the input does not exist in the model");
                _host.Check(outputColNames.All(name => _session.Graph[name] != null), "One of the output does not exist in the model");

                _outputColNames = outputColNames;
                (_outputColTypes, _tfOutputTypes) = GetOutputTypes(_session.Graph, _outputColNames);
                (_inputColNames, _inputColIndices, _isVectorInput, _tfInputShapes, _tfInputTypes) = GetInputMetaData(_session.Graph, inputColNames, inputSchema);
            }

            public TensorFlowMapper(IHostEnvironment env, ISchema inputSchema, string exportDir, string[] inputColNames, string[] outputColNames)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("TensorFlowMapper");
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckNonEmpty(inputColNames, nameof(inputColNames));
                _host.CheckNonEmpty(outputColNames, nameof(outputColNames));
                for (int i = 0; i < inputColNames.Length; i++)
                    _host.CheckNonWhiteSpace(inputColNames[i], nameof(inputColNames));
                for (int i = 0; i < outputColNames.Length; i++)
                    _host.CheckNonWhiteSpace(outputColNames[i], nameof(outputColNames));

                _isFrozen = false;
                _exportDir = exportDir;
                _session = LoadTFSession(exportDir);
                _host.Check(inputColNames.All(name => _session.Graph[name] != null), "One of the input does not exist in the model");
                _host.Check(outputColNames.All(name => _session.Graph[name] != null), "One of the output does not exist in the model");

                _outputColNames = outputColNames;
                (_outputColTypes, _tfOutputTypes) = GetOutputTypes(_session.Graph, _outputColNames);
                (_inputColNames, _inputColIndices, _isVectorInput, _tfInputShapes, _tfInputTypes) = GetInputMetaData(_session.Graph, inputColNames, inputSchema);
            }

            public static TensorFlowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel(GetVersionInfo());

                var isFrozen = ctx.Reader.ReadInt32();
                if (isFrozen == 1)
                {
                    var numInputs = ctx.Reader.ReadInt32();
                    Contracts.CheckDecode(numInputs > 0);

                    string[] source = new string[numInputs];
                    for (int j = 0; j < source.Length; j++)
                        source[j] = ctx.LoadNonEmptyString();

                    byte[] data = null;
                    if (!ctx.TryLoadBinaryStream("TFModel", r => data = r.ReadByteArray()))
                        throw env.ExceptDecode();

                    bool isMultiOutput = ctx.Header.ModelVerReadable >= 0x00010002;

                    var numOutputs = 1;
                    if (isMultiOutput)
                    {
                        numOutputs = ctx.Reader.ReadInt32();
                    }

                    Contracts.CheckDecode(numOutputs > 0);
                    var outputColNames = new string[numOutputs];
                    for (int j = 0; j < outputColNames.Length; j++)
                        outputColNames[j] = ctx.LoadNonEmptyString();

                    return new TensorFlowMapper(env, schema, data, source, outputColNames);
                }
                else
                {
                    var numInputs = ctx.Reader.ReadInt32();
                    Contracts.CheckDecode(numInputs > 0);

                    string[] source = new string[numInputs];
                    for (int j = 0; j < source.Length; j++)
                        source[j] = ctx.LoadNonEmptyString();

                    // Load model binary
                    byte[] tfFilesBin = null;
                    var load = ctx.TryLoadBinaryStream("TFSavedModel", br => tfFilesBin = br.ReadByteArray());
                    var tempDirName = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "_MLNET_TFTransform_" + Guid.NewGuid()));
                    var tempDir = Directory.CreateDirectory(tempDirName);
                    var tfZipFilePath = Path.Combine(tempDir.FullName, "tf_savedmodel.zip");
                    File.WriteAllBytes(tfZipFilePath, tfFilesBin);
                    ZipFile.ExtractToDirectory(tfZipFilePath, Path.Combine(tempDir.FullName, "tf_savedmodel"));

                    bool isMultiOutput = ctx.Header.ModelVerReadable >= 0x00010002;

                    var numOutputs = 1;
                    if (isMultiOutput)
                    {
                        numOutputs = ctx.Reader.ReadInt32();
                    }

                    Contracts.CheckDecode(numOutputs > 0);
                    var outputColNames = new string[numOutputs];
                    for (int j = 0; j < outputColNames.Length; j++)
                        outputColNames[j] = ctx.LoadNonEmptyString();

                    return new TensorFlowMapper(env, schema, Path.Combine(tempDir.FullName, "tf_savedmodel"), source, outputColNames);
                }
            }

            public void Save(ModelSaveContext ctx)
            {
                _host.AssertValue(ctx);
                ctx.CheckAtModel();
                ctx.SetVersionInfo(GetVersionInfo());

                ctx.Writer.Write(_isFrozen ? 1 : 0);
                if (_isFrozen)
                {
                    var buffer = new TFBuffer();
                    _session.Graph.ToGraphDef(buffer);

                    ctx.SaveBinaryStream("TFModel", w =>
                    {
                        w.WriteByteArray(buffer.ToArray());
                    });
                    _host.AssertNonEmpty(_inputColNames);
                    ctx.Writer.Write(_inputColNames.Length);
                    foreach (var colName in _inputColNames)
                        ctx.SaveNonEmptyString(colName);

                    _host.AssertNonEmpty(_outputColNames);
                    ctx.Writer.Write(_outputColNames.Length);
                    foreach (var colName in _outputColNames)
                        ctx.SaveNonEmptyString(colName);
                }
                else
                {
                    var tempDirName = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "_MLNET_TFTransform_" + Guid.NewGuid()));
                    var tempDir = Directory.CreateDirectory(tempDirName);
                    var tfZipFilePath = Path.Combine(tempDir.FullName, "tf_savedmodel.zip");

                    ZipFile.CreateFromDirectory(_exportDir, tfZipFilePath, CompressionLevel.Fastest, false);
                    byte[] byteArray = File.ReadAllBytes(tfZipFilePath);
                    ctx.SaveBinaryStream("TFSavedModel", w =>
                    {
                        w.WriteByteArray(byteArray);
                    });

                    _host.AssertNonEmpty(_inputColNames);
                    ctx.Writer.Write(_inputColNames.Length);
                    foreach (var colName in _inputColNames)
                        ctx.SaveNonEmptyString(colName);

                    _host.AssertNonEmpty(_outputColNames);
                    ctx.Writer.Write(_outputColNames.Length);
                    foreach (var colName in _outputColNames)
                        ctx.SaveNonEmptyString(colName);
                }
            }

            private TFSession LoadTFSession(byte[] modelBytes, string modelArg)
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
            private TFSession LoadTFSession(string exportDirSavedModel)
            {
                var sessionOptions = new TFSessionOptions();
                var exportDir = exportDirSavedModel;
                var tags = new string[] { "serve" };
                var graph = new TFGraph();
                var metaGraphDef = new TFBuffer();

                var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, graph, metaGraphDef);
                return session;
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
                var srcTensorGetters = new ITensorValueGetter[_inputColIndices.Length];
                for (int j = 0; j < _inputColIndices.Length; j++)
                {
                    int colIndex = _inputColIndices[j];
                    srcTensorGetters[j] = CreateTensorValueGetter(input, _tfInputTypes[j], _isVectorInput[j], colIndex, _tfInputShapes[j]);
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

            private Delegate[] MakeGetter(IRow input, Func<int, bool> activeOutput)
            {
                _host.AssertValue(input);

                var outputCache = new OutputCache();
                var activeOutputColNames = _outputColNames.Where((x, i) => activeOutput(i)).ToArray();

                var valueGetters = new List<Delegate>();
                for (int i = 0; i < _outputColNames.Length; i++)
                {
                    if (activeOutput(i))
                    {
                        var type = TFTensor.TypeFromTensorType(_tfOutputTypes[i]);
                        _host.Assert(type == _outputColTypes[i].ItemType.RawType);
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
                    if (Utils.Size(values) < _outputColTypes[iinfo].VectorSize)
                    {
                        values = new T[_outputColTypes[iinfo].VectorSize];
                        indices = new int[_outputColTypes[iinfo].VectorSize];
                    }

                    TensorFlowUtils.FetchData<T>(outputCache.Outputs[_outputColNames[iinfo]].Data, values);
                    dst = new VBuffer<T>(values.Length, values, indices);
                };
                return valuegetter;

            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    var runner = _session.GetRunner();
                    for (int i = 0; i < _inputColIndices.Length; i++)
                    {
                        var inputName = _inputColNames[i];
                        runner.AddInput(inputName, srcTensorGetters[i].GetTensor());
                    }

                    var tensors = runner.Fetch(activeOutputColNames).Run();
                    Contracts.Assert(tensors.Length > 0);

                    for (int j = 0; j < tensors.Length; j++)
                    {
                        outputCache.Outputs[activeOutputColNames[j]] = tensors[j];
                    }

                    outputCache.Position = position;
                }
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                using (var ch = _host.Start("CreateGetters"))
                {
                    var getters = MakeGetter(input, activeOutput);
                    ch.Done();
                    return getters;
                }
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _outputColNames.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                var info = new RowMapperColumnInfo[_outputColNames.Length];
                for (int i = 0; i < _outputColNames.Length; i++)
                    info[i] = new RowMapperColumnInfo(_outputColNames[i], _outputColTypes[i], null);
                return info;
            }

            private static (ColumnType[], TFDataType[]) GetOutputTypes(TFGraph graph, string[] columnNames)
            {
                Contracts.AssertValue(graph);
                Contracts.AssertNonEmpty(columnNames);
                Contracts.Assert(columnNames.All(name => graph[name] != null), "One of the output does not exist in the model");

                var columnTypes = new ColumnType[columnNames.Length];
                var tfTypes = new TFDataType[columnNames.Length];
                for (int i = 0; i < columnNames.Length; i++)
                {
                    var tfoutput = new TFOutput(graph[columnNames[i]]);
                    var shape = graph.GetTensorShape(tfoutput);

                    int[] dims = shape.ToIntArray().Skip(shape[0] == -1 ? BatchSize : 0).ToArray();

                    var type = TensorFlowUtils.Tf2MlNetType(tfoutput.OutputType);
                    columnTypes[i] = new VectorType(type, dims);
                    tfTypes[i] = tfoutput.OutputType;
                }
                return (columnTypes, tfTypes);
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
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "This is the frozen protobuf model file. Please see https://www.tensorflow.org/mobile/prepare_models for more details.", ShortName = "ModelDir", SortOrder = 0)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the output", ShortName = "output", SortOrder = 2)]
            public string[] OutputColumns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicator for frozen models", ShortName = "Frozen", SortOrder = 3)]
            public bool IsFrozen = true;
        }

        public const string Summary = "Transforms the data using the TensorFlow model.";
        public const string UserName = "TensorFlowTransform";
        public const string ShortName = "TFTransform";
        private const string RegistrationName = "TensorFlowTransform";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="modelFile">This is the frozen TensorFlow model file. https://www.tensorflow.org/mobile/prepare_models </param>
        /// <param name="isFrozen"></param>
        /// <param name="name">Name of the output column. Keep it same as in the TensorFlow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the TensorFlow model.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string modelFile, bool isFrozen, string name, params string[] source)
        {
            return Create(env, new Arguments() { InputColumns = source, OutputColumns = new[] { name }, ModelFile = modelFile, IsFrozen = isFrozen }, input);
        }

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="modelFile">This is the frozen tensorflow model file. https://www.tensorflow.org/mobile/prepare_models </param>
        /// <param name="isFrozen"></param>
        /// <param name="names">Name of the output column(s). Keep it same as in the Tensorflow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the Tensorflow model.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string modelFile, bool isFrozen, string[] names, string[] source)
        {
            return Create(env, new Arguments() { InputColumns = source, OutputColumns =  names, ModelFile = modelFile, IsFrozen = isFrozen }, input);
        }

        /// <include file='doc.xml' path='doc/members/member[@name="TensorflowTransform"]/*' />
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(args, nameof(args));
            host.CheckUserArg(Utils.Size(args.InputColumns) > 0, nameof(args.InputColumns));
            for (int i = 0; i < args.InputColumns.Length; i++)
                host.CheckNonWhiteSpace(args.InputColumns[i], nameof(args.InputColumns));
            for (int i = 0; i < args.OutputColumns.Length; i++)
                host.CheckNonWhiteSpace(args.OutputColumns[i], nameof(args.OutputColumns));
            host.CheckUserArg(args.OutputColumns.Distinct().Count() == args.OutputColumns.Length,
                nameof(args.OutputColumns), "Some of the output columns specified multiple times.");
            host.CheckNonWhiteSpace(args.ModelFile, nameof(args.ModelFile));

            TensorFlowMapper mapper = null;
            if (args.IsFrozen)
            {
                host.CheckUserArg(File.Exists(args.ModelFile), nameof(args.ModelFile));
                var modelBytes = File.ReadAllBytes(args.ModelFile);
                mapper = new TensorFlowMapper(host, input.Schema, modelBytes, args.InputColumns, args.OutputColumns);
            }
            else
            {
                var exportDir = args.ModelFile;
                mapper = new TensorFlowMapper(host, input.Schema, exportDir, args.InputColumns, args.OutputColumns);
            }
            return new RowToRowMapperTransform(host, input, mapper);
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
}
