// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TensorFlow;

[assembly: LoadableClass(TensorFlowTransform.Summary, typeof(IDataTransform), typeof(TensorFlowTransform.Arguments), typeof(SignatureDataTransform),
    TensorFlowTransform.UserName, TensorFlowTransform.ShortName)]

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

            private readonly string _outputColName;
            private readonly ColumnType _outputColType;
            private readonly TFDataType _tfOutputType;

            public const string LoaderSignature = "TFMapper";
            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "TENSFLOW",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature);
            }

            public TensorFlowMapper(IHostEnvironment env, ISchema inputSchema, byte[] modelBytes, string[] inputColNames, string outputCols)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("TensorFlowMapper");
                _host.CheckValue(inputSchema, nameof(inputSchema));

                _session = LoadTFSession(modelBytes, null);

                _outputColName = outputCols;
                (_outputColType, _tfOutputType) = GetOutputTypes(_session.Graph, _outputColName);
                (_inputColNames, _inputColIndices, _isVectorInput, _tfInputShapes, _tfInputTypes) = GetInputMetaData(_session.Graph, inputColNames, inputSchema);
            }

            public static TensorFlowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel(GetVersionInfo());

                var numInputs = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(numInputs > 0);

                string[] source = new string[numInputs];
                for (int j = 0; j < source.Length; j++)
                    source[j] = ctx.LoadNonEmptyString();

                byte[] data = null;
                if (!ctx.TryLoadBinaryStream("TFModel", r => data = r.ReadByteArray()))
                    throw env.ExceptDecode();

                var outputColName = ctx.LoadNonEmptyString();

                return new TensorFlowMapper(env, schema, data, source, outputColName);
            }

            public void Save(ModelSaveContext ctx)
            {
                _host.AssertValue(ctx);
                ctx.CheckAtModel();
                ctx.SetVersionInfo(GetVersionInfo());

                var buffer = new TFBuffer();
                _session.Graph.ToGraphDef(buffer);

                ctx.SaveBinaryStream("TFModel", w =>
                {
                    w.WriteByteArray(buffer.ToArray());
                });
                Contracts.AssertNonEmpty(_inputColNames);
                ctx.Writer.Write(_inputColNames.Length);
                foreach (var colName in _inputColNames)
                    ctx.SaveNonEmptyString(colName);

                ctx.SaveNonEmptyString(_outputColName);
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

            private Delegate MakeGetter(IRow input)
            {
                var type = TFTensor.TypeFromTensorType(_tfOutputType);
                _host.Assert(type == _outputColType.ItemType.RawType);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, _outputColType);
            }

            private Delegate MakeGetter<T>(IRow input, ColumnType columnType)
            {
                _host.AssertValue(input);
                _host.Assert(typeof(T) == columnType.ItemType.RawType);

                var srcTensorGetters = GetTensorValueGetters(input);

                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    var runner = _session.GetRunner();
                    for (int i = 0; i < _inputColIndices.Length; i++)
                    {
                        var inputName = _inputColNames[i];
                        runner.AddInput(inputName, srcTensorGetters[i].GetTensor());
                    }

                    var tensors = runner.Fetch(_outputColName).Run();

                    Contracts.Assert(tensors.Length > 0);

                    var values = dst.Values;
                    if (Utils.Size(values) < _outputColType.VectorSize)
                        values = new T[_outputColType.VectorSize];

                    TensorFlowUtils.FetchData<T>(tensors[0].Data, values);
                    dst = new VBuffer<T>(values.Length, values, dst.Indices);
                };
                return valuegetter;
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                var getters = new Delegate[1];
                disposer = null;
                using (var ch = _host.Start("CreateGetters"))
                {
                    if (activeOutput(0))
                        getters[0] = MakeGetter(input);
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
                var info = new RowMapperColumnInfo[1];
                info[0] = new RowMapperColumnInfo(_outputColName, _outputColType, null);
                return info;
            }

            private static (ColumnType, TFDataType) GetOutputTypes(TFGraph graph, string columnName)
            {
                var tfoutput = new TFOutput(graph[columnName]);
                var shape = graph.GetTensorShape(tfoutput);

                int[] dims = new int[shape.NumDimensions - 1];
                for (int k = 1; k < shape.NumDimensions; k++)
                    dims[k - 1] = (int)shape[k];

                var type = TensorFlowUtils.Tf2MlNetType(tfoutput.OutputType);
                return (new VectorType(type, dims), tfoutput.OutputType);
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
                    var shape = tfShapes[i].ToIntArray();
                    int valCount = 1;
                    for (int j = 1; j < shape.Length; j++)
                        valCount *= shape[j];
                    if (type.ValueCount != valCount)
                        throw Contracts.Except($"The size of model input '{colNames[i]}' does not match its size in the input data.");
                    isInputVector[i] = type.IsVector;

                    tfTypes[i] = tfoutput.OutputType;

                    var l = new long[tfShapes[i].NumDimensions];
                    for (int ishape = 0; ishape < tfShapes[i].NumDimensions; ishape++)
                    {
                        l[ishape] = tfShapes[i][ishape] == -1 ? 1 : tfShapes[i][ishape];
                    }
                    tfShapes[i] = new TFShape(l);
                }
                return (colNames, inputColIndices, isInputVector, tfShapes, tfTypes);
            }
        }

        public sealed class Arguments : TransformInputBase
        {

            [Argument(ArgumentType.Required, HelpText = "This is the frozen protobuf model file. Please see https://www.tensorflow.org/mobile/prepare_models for more detail(s).", ShortName = "ModelDir", SortOrder = 0)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the output", ShortName = "output", SortOrder = 2)]
            public string OutputColumn;
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
        /// <param name="name">Name of the output column. Keep it same as in the TensorFlow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the TensorFlow model.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string modelFile, string name, params string[] source)
        {
            return Create(env, new Arguments() { InputColumns = source, OutputColumn = name, ModelFile = modelFile }, input);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(args, nameof(args));
            host.CheckUserArg(Utils.Size(args.InputColumns) > 0, nameof(args.InputColumns));
            for (int i = 0; i < args.InputColumns.Length; i++)
                host.CheckNonWhiteSpace(args.InputColumns[i], nameof(args.InputColumns));
            host.CheckNonWhiteSpace(args.ModelFile, nameof(args.ModelFile));
            host.CheckUserArg(File.Exists(args.ModelFile), nameof(args.ModelFile));

            var modelBytes = File.ReadAllBytes(args.ModelFile);
            var mapper = new TensorFlowMapper(host, input.Schema, modelBytes, args.InputColumns, args.OutputColumn);
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
        public static CommonOutputs.TransformOutput Convert(IHostEnvironment env, Arguments input)
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
