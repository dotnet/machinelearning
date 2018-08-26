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

[assembly: LoadableClass(TensorflowTransform.Summary, typeof(IDataTransform), typeof(TensorflowTransform.Arguments), typeof(SignatureDataTransform),
    TensorflowTransform.UserName, TensorflowTransform.ShortName)]

// This is for de-serialization from a binary model file.
[assembly: LoadableClass(typeof(TensorflowTransform.TensorFlowMapper), null, typeof(SignatureLoadRowMapper),
    "", TensorflowTransform.TensorFlowMapper.LoaderSignature)]

[assembly: EntryPointModule(typeof(TensorflowTransform))]

namespace Microsoft.ML.Transforms
{
    public static class TensorflowTransform
    {
        internal sealed class TensorFlowMapper : IRowMapper
        {
            private readonly IHost _host;

            /// <summary>
            /// Tensorflow session object
            /// </summary>
            private readonly TFSession _session;

            public readonly string[] InputColNames;
            public readonly int[] InputColIndices;
            public readonly bool[] IsVectorInput;
            private readonly TFShape[] _tfInputShapes;
            private readonly TFDataType[] _tfInputTypes;

            public readonly string OutputColName;
            public readonly ColumnType OutputColType;
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

                OutputColName = outputCols;
                (OutputColType, _tfOutputType) = GetOutputTypes(_session.Graph, OutputColName);
                (InputColNames, InputColIndices, IsVectorInput, _tfInputShapes, _tfInputTypes) = GetInputMetaData(_session.Graph, inputColNames, inputSchema);
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
                Contracts.AssertNonEmpty(InputColNames);
                ctx.Writer.Write(InputColNames.Length);
                foreach (var colName in InputColNames)
                    ctx.SaveNonEmptyString(colName);

                ctx.SaveNonEmptyString(OutputColName);
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
                var srcTensorGetters = new ITensorValueGetter[InputColIndices.Length];
                for (int j = 0; j < InputColIndices.Length; j++)
                {
                    int colIndex = InputColIndices[j];
                    srcTensorGetters[j] = CreateTensorValueGetter(input, _tfInputTypes[j], IsVectorInput[j], colIndex, _tfInputShapes[j]);
                }
                return srcTensorGetters;
            }

            private Delegate MakeGetter(IRow input)
            {
                var type = TFTensor.TypeFromTensorType(_tfOutputType);
                _host.Assert(type == OutputColType.ItemType.RawType);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, OutputColType);
            }

            private Delegate MakeGetter<T>(IRow input, ColumnType columnType)
            {
                _host.AssertValue(input);
                _host.Assert(typeof(T) == columnType.ItemType.RawType);

                var srcTensorGetters = GetTensorValueGetters(input);

                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    var runner = _session.GetRunner();
                    for (int i = 0; i < InputColIndices.Length; i++)
                    {
                        var inputName = InputColNames[i];
                        runner.AddInput(inputName, srcTensorGetters[i].GetTensor());
                    }

                    var tensors = runner.Fetch(OutputColName).Run();

                    Contracts.Assert(tensors.Length > 0);

                    var values = dst.Values;
                    if (Utils.Size(values) < OutputColType.VectorSize)
                        values = new T[OutputColType.VectorSize];

                    TensorflowUtils.FetchData<T>(tensors[0].Data, values);
                    dst = new VBuffer<T>(values.Length, values);
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
                return col => activeOutput(0) && InputColIndices.Any(i => i == col);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                var info = new RowMapperColumnInfo[1];
                info[0] = new RowMapperColumnInfo(OutputColName, OutputColType, null);
                return info;
            }

            private static (ColumnType, TFDataType) GetOutputTypes(TFGraph graph, string columnName)
            {
                var tfoutput = new TFOutput(graph[columnName]);
                var shape = graph.GetTensorShape(tfoutput);

                int[] dims = new int[shape.NumDimensions];
                for (int k = 0; k < shape.NumDimensions; k++)
                    dims[k] = (int)(shape[k] == -1 ? 1 : shape[k]);

                var type = TensorflowUtils.Tf2MlNetType(tfoutput.OutputType);
                return (new VectorType(type, dims), tfoutput.OutputType);
            }

            private static (string[], int[], bool[], TFShape[], TFDataType[]) GetInputMetaData(TFGraph graph, string[] source, ISchema inputSchema)
            {
                var tfShapes = new TFShape[source.Length];
                var tfTypes = new TFDataType[source.Length];
                var colNames = new string[source.Length];
                var inputColIndices = new int[source.Length];
                var isInputVector = new bool[source.Length];
                for (int j = 0; j < source.Length; j++)
                {
                    colNames[j] = source[j];
                    if (!inputSchema.TryGetColumnIndex(colNames[j], out inputColIndices[j]))
                        throw Contracts.Except($"Column '{colNames[j]}' does not exist");

                    isInputVector[j] = inputSchema.GetColumnType(inputColIndices[j]).IsVector;

                    var tfoutput = new TFOutput(graph[colNames[j]]);

                    if (!TensorflowUtils.IsTypeSupported(tfoutput.OutputType))
                        throw Contracts.Except($"Input type '{tfoutput.OutputType}' of input column '{colNames[j]}' is not supported in Tensorflow");

                    tfShapes[j] = graph.GetTensorShape(tfoutput);
                    tfTypes[j] = tfoutput.OutputType;

                    var l = new long[tfShapes[j].NumDimensions];
                    for (int ishape = 0; ishape < tfShapes[j].NumDimensions; ishape++)
                    {
                        l[ishape] = tfShapes[j][ishape] == -1 ? 1 : tfShapes[j][ishape];
                    }
                    tfShapes[j] = new TFShape(l);
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

        public const string Summary = "Transforms the data using the tenorflow model.";
        public const string UserName = "TensorflowTransform";
        public const string ShortName = "TFTransform";
        private const string RegistrationName = "TensorflowTransform";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="modelFile">This is the frozen tensorflow model file. https://www.tensorflow.org/mobile/prepare_models </param>
        /// <param name="name">Name of the output column. Keep it same as in the Tensorflow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the Tensorflow model.</param>
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
