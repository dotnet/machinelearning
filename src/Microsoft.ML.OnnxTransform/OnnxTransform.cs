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
using Microsoft.ML.OnnxScoring;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using OnnxShape = System.Collections.Generic.List<long>;
using Microsoft.ML.Core.Data;

[assembly: LoadableClass(OnnxTransform.Summary, typeof(IDataTransform), typeof(OnnxTransform),
    typeof(OnnxTransform.Arguments), typeof(SignatureDataTransform), OnnxTransform.UserName, OnnxTransform.ShortName)]

[assembly: LoadableClass(OnnxTransform.Summary, typeof(IDataTransform), typeof(OnnxTransform),
    null, typeof(SignatureLoadDataTransform), OnnxTransform.UserName, OnnxTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(OnnxTransform), null, typeof(SignatureLoadModel),
    OnnxTransform.UserName, OnnxTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(OnnxTransform), null, typeof(SignatureLoadRowMapper),
    OnnxTransform.UserName, OnnxTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(OnnxTransform))]

namespace Microsoft.ML.OnnxScoring
{
    public sealed class OnnxTransform : ITransformer, ICanSaveModel
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Path to the onnx model file.", ShortName = "model", SortOrder = 0)]
            public string ModelFile;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Name of the input column.", SortOrder = 1)]
            public string InputColumn;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Name of the output column.", SortOrder = 2)]
            public string OutputColumn;
        }

        private readonly IHost _host;
        private readonly Arguments _args;
        internal readonly OnnxModel Model;
        private const string RegistrationName = "OnnxTransform";

        internal const string Summary = "Transforms the data using the Onnx model.";
        internal const string UserName = "OnnxTransform";
        internal const string ShortName = "Onnx";
        internal const string LoaderSignature = "OnnxTransform";

        public readonly string Input;
        public readonly string Output;
        public readonly ColumnType OutputType;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ONNXSCOR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
	        loaderAssemblyName: typeof(OnnxTransform).Assembly.FullName);
        }

        // Factory method for SignatureDataTransform
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
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

            var inputColumn = ctx.LoadNonEmptyString();
            var outputColumn = ctx.LoadNonEmptyString();
            var args = new Arguments() { InputColumn = inputColumn, OutputColumn = outputColumn };

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
            _host.CheckNonWhiteSpace(args.InputColumn, nameof(args.InputColumn));
            _host.CheckNonWhiteSpace(args.OutputColumn, nameof(args.OutputColumn));

            if (modelBytes == null)
            {
                _host.CheckNonWhiteSpace(args.ModelFile, nameof(args.ModelFile));
                _host.CheckUserArg(File.Exists(args.ModelFile), nameof(args.ModelFile));
                Model = new OnnxModel(args.ModelFile);
            }
            else
                Model = OnnxModel.CreateFromBytes(modelBytes);

            Input = args.InputColumn;
            Output = args.OutputColumn;

            var outputNodeInfo = Model.GetOutputsInfo().Where(x => x.Name == args.OutputColumn).First();
            var type = OnnxUtils.OnnxToMlNetType(outputNodeInfo.Type);
            var shape = outputNodeInfo.Shape;
            var dims = shape.Count > 0 ? shape.Skip(shape[0] < 0 ? 1 : 0).Select( x => (int) x ).ToArray() : new[] { 0 };
            OutputType = new VectorType(type, dims);
            _args = args;
        }

        public OnnxTransform(IHostEnvironment env, string modelFile, string inputColumn, string outputColumn)
            : this(env, new Arguments() { ModelFile = modelFile, InputColumn = inputColumn, OutputColumn = outputColumn })
        {
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            if (!inputSchema.TryGetColumnIndex(Input, out int srcCol))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", Input);

            var transform = Transform(new EmptyDataView(_host, inputSchema));
            return transform.Schema;
        }

        private IRowMapper MakeRowMapper(ISchema schema) => new Mapper(_host, this, schema);

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
            ctx.SaveNonEmptyString(_args.InputColumn);
            ctx.SaveNonEmptyString(_args.OutputColumn);
        }

        public bool IsRowToRowMapper => true;

        public IRowToRowMapper GetRowToRowMapper(ISchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            return MakeDataTransform(new EmptyDataView(_host, inputSchema));
        }

        private sealed class Mapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly OnnxTransform _parent;

            private readonly Type _outputItemRawType;
            private readonly ColumnType _outputColType;
            private readonly string _outputColName;

            private readonly IdvToTensorAdapter _idvToTensorAdapter;

            public Mapper(IHostEnvironment env, OnnxTransform parent, ISchema inputSchema)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(Mapper));
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckValue(parent, nameof(parent));

                _parent = parent;
                var model = _parent.Model;
                _idvToTensorAdapter = new IdvToTensorAdapter(inputSchema, parent._args.InputColumn,
                                            model.ModelInfo.InputsInfo[0]);

                // TODO: Remove assumption below
                // Assume first output dimension is 1
                var outputNodeInfo = model.ModelInfo.OutputsInfo[0];
                int[] dims = outputNodeInfo.Shape.Skip(1).Select(x => (int)x).ToArray();
                var outputItemType = OnnxUtils.OnnxToMlNetType(outputNodeInfo.Type);
                _outputColType = new VectorType(outputItemType, dims);
                _outputColName = _parent.Output;
                _outputItemRawType = outputItemType.RawType;
                _host.Assert(_outputItemRawType == _outputColType.ItemType.RawType);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                var info = new RowMapperColumnInfo[1];
                info[0] = new RowMapperColumnInfo(_outputColName, _outputColType, null);
                return info;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => activeOutput(0) && (_idvToTensorAdapter.IdvColumnIndex == col);
            }

            public void Save(ModelSaveContext ctx)
            {
                _parent.Save(ctx);
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                var getters = new Delegate[1];
                using (var ch = _host.Start("CreateGetters"))
                {
                    if (activeOutput(0))
                        getters[0] = Utils.MarshalInvoke(MakeGetter<int>, _outputItemRawType, input);

                    ch.Done();
                    return getters;
                }
            }

            private Delegate MakeGetter<T>(IRow input)
            {
                _host.AssertValue(input);
                _host.Assert(typeof(T) == _outputItemRawType);

                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    _idvToTensorAdapter.InitializeValueGetters(input);
                    var inputTensors = new List<Tensor> { _idvToTensorAdapter.GetTensor() };
                    var outputTensors = _parent.Model.Run(inputTensors);
                    Contracts.Assert(outputTensors.Count() > 0);

                    var values = dst.Values;
                    if (Utils.Size(values) < _outputColType.VectorSize)
                        values = new T[_outputColType.VectorSize];

                    OnnxUtils.CopyTo(outputTensors[0], values);
                    dst = new VBuffer<T>(values.Length, values, dst.Indices);
                };

                return valuegetter;
            }
        }
    }
    public sealed class OnnxEstimator : TrivialEstimator<OnnxTransform>
    {
        public OnnxEstimator(IHostEnvironment env, string modelFile, string input, string output)
           : this(env, new OnnxTransform(env, modelFile, input, output))
        {
        }

        public OnnxEstimator(IHostEnvironment env, OnnxTransform transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(OnnxTransform)), transformer)
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);

            var input = Transformer.Input;
            if (!inputSchema.TryFindColumn(input, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
            if (!(col.Kind == SchemaShape.Column.VectorKind.VariableVector || col.Kind == SchemaShape.Column.VectorKind.Vector))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, nameof(VectorType), col.GetTypeString());
            var inputNodeInfo = Transformer.Model.GetInputsInfo().Where(x => x.Name == input).First();
            var expectedType = OnnxUtils.OnnxToMlNetType(inputNodeInfo.Type);
            if (col.ItemType != expectedType)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, expectedType.ToString(), col.ItemType.ToString());

            resultDic[Transformer.Output] = new SchemaShape.Column(Transformer.Output,
                Transformer.OutputType.IsKnownSizeVector ? SchemaShape.Column.VectorKind.Vector
                : SchemaShape.Column.VectorKind.VariableVector, Transformer.OutputType.ItemType, false);

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
                return new OnnxEstimator(env, _modelFile, inputNames[outCol.Input], outputNames[outCol]);
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
