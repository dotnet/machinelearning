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

[assembly: LoadableClass(AgTransform.Summary, typeof(IDataTransform), typeof(AgTransform),
    typeof(AgTransform.Arguments), typeof(SignatureDataTransform), AgTransform.UserName, AgTransform.ShortName)]

// This is for de-serialization from a binary model file.
[assembly: LoadableClass(typeof(AgTransform.AgMapper), null, typeof(SignatureLoadRowMapper),
    "", AgTransform.AgMapper.LoaderSignature)]

[assembly: EntryPointModule(typeof(AgTransform))]

namespace Microsoft.ML.Transforms
{
    public static class AgTransform
    {
        internal sealed class AgMapper : IRowMapper
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

            private const int BatchSize = 1;
            public const string LoaderSignature = "AgMapper";
            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "TENSFLOW",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature);
            }

            public AgMapper(IHostEnvironment env, ISchema inputSchema, string[] inputColNames, string outputColName)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("TensorFlowMapper");
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckNonEmpty(inputColNames, nameof(inputColNames));
                _host.CheckNonEmpty(outputColName, nameof(outputColName));

                _session = LoadTFSessionSavedModel();

                _outputColName = outputColName;
                _outputColType = new VectorType(NumberType.R4, 3);

                _inputColIndices = new int[inputColNames.Length];
                var colNames = new string[inputColNames.Length];
                for (int i = 0; i < inputColNames.Length; i++)
                {
                    colNames[i] = inputColNames[i];
                    if (!inputSchema.TryGetColumnIndex(colNames[i], out _inputColIndices[i]))
                        throw Contracts.Except($"Column '{colNames[i]}' does not exist");
                }
            }

            public AgMapper(IHostEnvironment env, ISchema inputSchema, byte[] modelBytes, string[] inputColNames, string outputColName)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("TensorFlowMapper");
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckNonEmpty(modelBytes, nameof(modelBytes));
                _host.CheckNonEmpty(inputColNames, nameof(inputColNames));
                _host.CheckNonEmpty(outputColName, nameof(outputColName));

                _session = LoadTFSession(modelBytes, null);
                _host.CheckValue(_session.Graph[outputColName], nameof(outputColName), "Output does not exist in the model");
                _host.Check(inputColNames.All(name => _session.Graph[name] != null), "One of the input does not exist in the model");

                _outputColName = outputColName;
                (_outputColType, _tfOutputType) = GetOutputTypes(_session.Graph, _outputColName);
                (_inputColNames, _inputColIndices, _isVectorInput, _tfInputShapes, _tfInputTypes) = GetInputMetaData(_session.Graph, inputColNames, inputSchema);
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

            private static (ColumnType, TFDataType) GetOutputTypes(TFGraph graph, string columnName)
            {
                Contracts.AssertValue(graph);
                Contracts.AssertNonEmpty(columnName);
                Contracts.AssertValue(graph[columnName]);

                var tfoutput = new TFOutput(graph[columnName]);
                var shape = graph.GetTensorShape(tfoutput);

                int[] dims = shape.ToIntArray().Skip(shape[0] == -1 ? BatchSize : 0).ToArray();
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

            public static AgMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
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

                return new AgMapper(env, schema, data, source, outputColName);
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

                ctx.SaveNonEmptyString(_outputColName);
            }

            private TFSession LoadTFSessionSavedModel()
            {
                var sessionOptions = new TFSessionOptions();
                var exportDir = @"D:\CS231N-Convolutional-Neural-Networks-for-Visual-Recognition\assignments_details\09042018\save_load\hellotensor_2";
                var tags = new string[] { "serve" };
                var graph = new TFGraph();
                var metaGraphDef = new TFBuffer();

                var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, graph, metaGraphDef);
                return session;
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                var getters = new Delegate[1];
                disposer = null;
                using (var ch = _host.Start("CreateGetters"))
                {
                    if (activeOutput(0))
                        getters[0] = MakeGetter<float>(input);
                    ch.Done();
                    return getters;
                }
            }
            private Delegate MakeGetter<T>(IRow input)
            {
                _host.AssertValue(input);
                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    var values = dst.Values;
                    if (Utils.Size(values) < _outputColType.VectorSize)
                        values = new T[_outputColType.VectorSize];

                    dst = new VBuffer<T>(values.Length, values, dst.Indices);
                };
                return valuegetter;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => activeOutput(0) && _inputColIndices.Any(i => i == col);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                return new[] { new RowMapperColumnInfo(_outputColName, _outputColType, null) };
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the output", ShortName = "output", SortOrder = 2)]
            public string OutputColumn;
        }

        public const string Summary = "Computes sum.";
        public const string UserName = "AgTransform";
        public const string ShortName = "AgTransform";
        private const string RegistrationName = "AgTransform";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column. Keep it same as in the TensorFlow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the TensorFlow model.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string name, params string[] source)
        {
            return Create(env, new Arguments() { InputColumns = source, OutputColumn = name }, input);
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

            var mapper = new AgMapper(host, input.Schema, args.InputColumns, args.OutputColumn);
            return new RowToRowMapperTransform(host, input, mapper);
        }
    }
}