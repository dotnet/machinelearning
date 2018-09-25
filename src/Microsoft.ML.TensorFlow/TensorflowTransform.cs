// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TensorFlow;

[assembly: LoadableClass(TensorFlowTransform.Summary, typeof(IDataTransform), typeof(TensorFlowTransform),
    typeof(TensorFlowTransform.Arguments), typeof(SignatureDataTransform), TensorFlowTransform.UserName, TensorFlowTransform.ShortName)]

[assembly: LoadableClass(TensorFlowTransform.Summary, typeof(IDataTransform), typeof(TensorFlowTransform), null, typeof(SignatureLoadDataTransform),
    TensorFlowTransform.UserName, TensorFlowTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(TensorFlowTransform), null, typeof(SignatureLoadModel),
    TensorFlowTransform.UserName, TensorFlowTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TensorFlowTransform), null, typeof(SignatureLoadRowMapper),
    TensorFlowTransform.UserName, TensorFlowTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(TensorFlowTransform))]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="TensorflowTransform"]/*' />
    public sealed class TensorFlowTransform : ITransformer, ICanSaveModel
    {
        public class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "TensorFlow model used by the transform. Please see https://www.tensorflow.org/mobile/prepare_models for more details.", SortOrder = 0)]
            public string Model;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the outputs", ShortName = "outputs", SortOrder = 2)]
            public string[] OutputColumns;
        }

        public sealed class TrainingArguments
        {
            [Argument(ArgumentType.Required, HelpText = "Directory for saving intermediate model state and temporary files.", SortOrder = 3)]
            public string WrokingDir;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Training labels.", ShortName = "OptimizationOp", SortOrder = 4)]
            public string LabeLColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the optimization operation in the TensorFlow graph.", ShortName = "OptimizationOp", SortOrder = 4)]
            public string OptimizationOperation;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the operation in the TensorFlow graph to compute training loss (Optional)", ShortName = "LossOp", SortOrder = 5)]
            public string LossOperation;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the operation in the TensorFlow graph to compute performance metric during training (Optional)", ShortName = "MetricOp", SortOrder = 6)]
            public string MetricOperation;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Number of samples to use for training.", SortOrder = 7)]
            public int BatchSize = 64;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Number of training iterations.", SortOrder = 7)]
            public int Epoch = 10;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the operation in the TensorFlow graph which sets optimizer learning rate (Optional).", SortOrder = 8)]
            public string LearningRateOperation;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Learning rate to use during optimization.", SortOrder = 8)]
            public float LearningRate = 0.01f;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Shuffle data before each iteration.", SortOrder = 8)]
            public bool Shuffle = true;
        }

        private readonly IHost _host;
        private readonly string _savedModelPath;
        private readonly bool _isTemporarySavedModel;
        private const string RegistrationName = "TensorFlowTransform";

        internal readonly TFSession Session;
        internal readonly ColumnType[] OutputTypes;
        internal readonly TFDataType[] TFOutputTypes;
        internal readonly TFDataType[] TFInputTypes;
        internal readonly TFShape[] TFInputShapes;
        internal TFGraph Graph => Session.Graph;

        public readonly string[] Inputs;
        public readonly string[] Outputs;

        public static int BatchSize = 1;
        internal const string Summary = "Transforms the data using the TensorFlow model.";
        internal const string UserName = "TensorFlowTransform";
        internal const string ShortName = "TFTransform";
        internal const string LoaderSignature = "TensorFlowTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TENSFLOW",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002,  // Added Support for Multiple Outputs and SavedModel.
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TensorFlowTransform).Assembly.FullName);
        }

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="model">Path to the TensorFlow model. </param>
        /// <param name="names">Name of the output column(s). Keep it same as in the Tensorflow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the Tensorflow model.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string model, string[] names, string[] source)
        {
            return new TensorFlowTransform(env, TensorFlowUtils.GetSession(env, model), source, names, TensorFlowUtils.IsSavedModel(env, model) ? model : null, false).MakeDataTransform(input);
        }

        /// <summary>
        /// Utility method to create <see cref="TensorFlowTransform"/> with retraining capabilities.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="model">Path to the TensorFlow model. </param>
        /// <param name="names">Name of the output column(s). Keep it same as in the Tensorflow model.</param>
        /// <param name="source">Name of the input column(s). Keep it same as in the Tensorflow model.</param>
        /// <param name="args">Argument object containing training related parameters.</param>
        /// <returns></returns>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string model, string[] names, string[] source, TrainingArguments args)
        {
            // Shuffling fails because ShuffleTransform is not RowToRowMapper
            //if(args.Shuffle)
            //{
            //    input = new ShuffleTransform(env, new ShuffleTransform.Arguments(), input);
            //}
            return new TensorFlowTransform(env, model, names, source, args, input).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static TensorFlowTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            GetModelInfo(env, ctx, out string[] inputs, out string[] outputs, out bool isFrozen);
            if (isFrozen)
            {
                byte[] modelBytes = null;
                if (!ctx.TryLoadBinaryStream("TFModel", r => modelBytes = r.ReadByteArray()))
                    throw env.ExceptDecode();
                return new TensorFlowTransform(env, TensorFlowUtils.LoadTFSession(env, modelBytes), inputs, outputs, null, false);
            }

            var tempDirPath = Path.GetFullPath(Path.Combine(Path.GetTempPath(), RegistrationName + "_" + Guid.NewGuid()));
            TensorFlowUtils.CreateFolderWithAclIfNotExists(env, tempDirPath);
            try
            {
                var load = ctx.TryLoadBinaryStream("TFSavedModel", br =>
                {
                    int count = br.ReadInt32();
                    for (int n = 0; n < count; n++)
                    {
                        string relativeFile = br.ReadString();
                        long fileLength = br.ReadInt64();

                        string fullFilePath = Path.Combine(tempDirPath, relativeFile);
                        string fullFileDir = Path.GetDirectoryName(fullFilePath);
                        if (fullFileDir != tempDirPath)
                        {
                            TensorFlowUtils.CreateFolderWithAclIfNotExists(env, fullFileDir);
                        }
                        using (var fs = new FileStream(fullFilePath, FileMode.Create, FileAccess.Write))
                        {
                            long actualRead = br.BaseStream.CopyRange(fs, fileLength);
                            env.Assert(actualRead == fileLength);
                        }
                    }
                });

                return new TensorFlowTransform(env, TensorFlowUtils.GetSession(env, tempDirPath), inputs, outputs, tempDirPath, true);
            }
            catch (Exception)
            {
                TensorFlowUtils.DeleteFolderWithRetries(env, tempDirPath);
                throw;
            }
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));
            env.CheckValue(args.InputColumns, nameof(args.InputColumns));
            env.CheckValue(args.OutputColumns, nameof(args.OutputColumns));

            return new TensorFlowTransform(env, TensorFlowUtils.GetSession(env, args.Model), args.InputColumns, args.OutputColumns, TensorFlowUtils.IsSavedModel(env, args.Model) ? args.Model : null, false).MakeDataTransform(input);
        }

        internal TensorFlowTransform(IHostEnvironment env, string model, string[] names, string[] source, TrainingArguments args, IDataView input)
            : this(env, TensorFlowUtils.GetSession(env, model), source, names, TensorFlowUtils.IsSavedModel(env, model) ? model : null, false)
        {

            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));
            env.CheckNonEmpty(args.WrokingDir, nameof(args.WrokingDir));
            env.CheckNonEmpty(args.OptimizationOperation, nameof(args.OptimizationOperation));

            TrainCore(args, input);
        }

        private void TrainCore(TrainingArguments args, IDataView input)
        {
            var inputsForTraining = new string[Inputs.Length + 1];
            var inputColIndices = new int[inputsForTraining.Length];
            var isInputVector = new bool[inputsForTraining.Length];
            var tfInputTypes = new TFDataType[inputsForTraining.Length];
            var tfInputShapes = new TFShape[inputsForTraining.Length];

            for (int i = 0; i < Inputs.Length; i++)
            {
                inputsForTraining[i] = Inputs[i];
            }

            inputsForTraining[inputsForTraining.Length - 1] = args.LabeLColumn;

            var inputSchema = input.Schema;
            for (int i = 0; i < inputsForTraining.Length; i++)
            {
                if (!inputSchema.TryGetColumnIndex(inputsForTraining[i], out inputColIndices[i]))
                    throw _host.Except($"Column {inputsForTraining[i]} doesn't exist");

                var type = inputSchema.GetColumnType(inputColIndices[i]);
                isInputVector[i] = type.IsVector;

                var tfInput = new TFOutput(Graph[inputsForTraining[i]]);
                tfInputTypes[i] = tfInput.OutputType;
                tfInputShapes[i] = Graph.GetTensorShape(tfInput);
                if (tfInputShapes[i].NumDimensions != -1)
                {
                    var newShape = new long[tfInputShapes[i].NumDimensions];
                    newShape[0] = tfInputShapes[i][0] == -1 ? args.BatchSize : tfInputShapes[i][0];

                    for (int j = 1; j < tfInputShapes[i].NumDimensions; j++)
                        newShape[j] = tfInputShapes[i][j];
                    tfInputShapes[i] = new TFShape(newShape);
                }

                var expectedType = TensorFlowUtils.Tf2MlNetType(tfInputTypes[i]);
                if (type.ItemType != expectedType)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputsForTraining[i], expectedType.ToString(), type.ToString());
            }

            var fetchList = new List<string>();
            if (args.LossOperation != null)
                fetchList.Add(args.LossOperation);
            if (args.MetricOperation != null)
                fetchList.Add(args.MetricOperation);

            for (int epoch = 0; epoch < args.Epoch; epoch++)
            {
                using (var cursor = input.GetRowCursor(a => true))
                {
                    var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, tfInputTypes, tfInputShapes);

                    float loss=0;
                    float metric=0;
                    int rows = 0;
                    bool isDataLeft = false;
                    using (var ch = _host.Start("Training TensorFlow model..."))
                    {
                        while (cursor.MoveNext())
                        {
                            rows++;
                            for (int i = 0; i < inputColIndices.Length; i++)
                            {
                                isDataLeft = true;
                                srcTensorGetters[i].BufferTrainingData();
                            }

                            if ((rows % args.BatchSize) == 0)
                            {
                                isDataLeft = false;
                                (loss, metric) = TrainBatch(inputColIndices, inputsForTraining, srcTensorGetters, fetchList, args);
                            }
                        }
                        if (isDataLeft)
                        {
                            isDataLeft = false;
                            ch.Warning("Not training on the last batch. The batch size is less than {0}.", args.BatchSize);
                        }
                        ch.Info("Loss: {0}, Metric: {1}", fetchList.Count > 0 ? loss.ToString("#.####") : "None",
                                fetchList.Count > 1 ? metric.ToString("#.####") : "None");
                    }
                }
            }
        }

        private (float loss, float metric) TrainBatch(int[] inputColIndices,
            string[] inputsForTraining,
            ITensorValueGetter[] srcTensorGetters,
            List<string> fetchList,
            TrainingArguments args)
        {
            float loss = 0;
            float metric = 0;
            var runner = Session.GetRunner();
            for (int i = 0; i < inputColIndices.Length; i++)
            {
                var inputName = inputsForTraining[i];
                runner.AddInput(inputName, srcTensorGetters[i].GetBufferedBatchTensor());
            }
            if (args.LearningRateOperation != null)
                runner.AddInput(args.LearningRateOperation, new TFTensor(args.LearningRate));
            runner.AddTarget(args.OptimizationOperation);

            if (fetchList.Count > 0)
                runner.Fetch(fetchList.ToArray());

            var tensor = runner.Run();
            loss = tensor.Length > 0 ? (float)tensor[0].GetValue() : 0.0f;
            metric = tensor.Length > 1 ? (float)tensor[1].GetValue() : 0.0f;

            return (loss, metric);
        }

        private static ITensorValueGetter CreateTensorValueGetter<T>(IRow input, bool isVector, int colIndex, TFShape tfShape)
        {
            if (isVector)
                return new TensorValueGetterVec<T>(input, colIndex, tfShape);
            else
                return new TensorValueGetter<T>(input, colIndex, tfShape);
        }

        private static ITensorValueGetter CreateTensorValueGetter(IRow input, TFDataType tfType, bool isVector, int colIndex, TFShape tfShape)
        {
            var type = TFTensor.TypeFromTensorType(tfType);
            Contracts.AssertValue(type);
            return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type, input, isVector, colIndex, tfShape);
        }

        private static ITensorValueGetter[] GetTensorValueGetters(IRow input,
            int[] inputColIndices,
            bool[] isInputVector,
            TFDataType[] tfInputTypes,
            TFShape[] tfInputShapes)
        {
            var srcTensorGetters = new ITensorValueGetter[inputColIndices.Length];
            for (int i = 0; i < inputColIndices.Length; i++)
            {
                int colIndex = inputColIndices[i];
                srcTensorGetters[i] = CreateTensorValueGetter(input, tfInputTypes[i], isInputVector[i], colIndex, tfInputShapes[i]);
            }
            return srcTensorGetters;
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static void GetModelInfo(IHostEnvironment env, ModelLoadContext ctx, out string[] inputs, out string[] outputs, out bool isFrozen)
        {
            isFrozen = true;
            bool isNonFrozenModelSupported = ctx.Header.ModelVerReadable >= 0x00010002;
            if (isNonFrozenModelSupported)
                isFrozen = ctx.Reader.ReadBoolByte();

            var numInputs = ctx.Reader.ReadInt32();
            env.CheckDecode(numInputs > 0);
            inputs = new string[numInputs];
            for (int j = 0; j < inputs.Length; j++)
                inputs[j] = ctx.LoadNonEmptyString();

            bool isMultiOutput = ctx.Header.ModelVerReadable >= 0x00010002;
            var numOutputs = 1;
            if (isMultiOutput)
                numOutputs = ctx.Reader.ReadInt32();

            env.CheckDecode(numOutputs > 0);
            outputs = new string[numOutputs];
            for (int j = 0; j < outputs.Length; j++)
                outputs[j] = ctx.LoadNonEmptyString();
        }

        internal TensorFlowTransform(IHostEnvironment env, TFSession session, string[] inputs, string[] outputs, string savedModelPath, bool isTemporarySavedModel)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(RegistrationName));
            _host.CheckValue(session, nameof(session));
            _host.CheckNonEmpty(inputs, nameof(inputs));
            _host.CheckNonEmpty(outputs, nameof(outputs));

            Session = session;
            _savedModelPath = savedModelPath;
            _isTemporarySavedModel = isTemporarySavedModel;

            foreach (var input in inputs)
            {
                _host.CheckNonWhiteSpace(input, nameof(inputs));
                if (Session.Graph[input] == null)
                    throw _host.ExceptParam(nameof(inputs), $"Input column '{input}' does not exist in the model");
                var tfInput = new TFOutput(Session.Graph[input]);
                if (!TensorFlowUtils.IsTypeSupported(tfInput.OutputType))
                    throw _host.ExceptParam(nameof(session), $"Input type '{tfInput.OutputType}' of input column '{input}' is not supported in TensorFlow");
            }

            var newNames = new HashSet<string>();
            foreach (var output in outputs)
            {
                _host.CheckNonWhiteSpace(output, nameof(outputs));
                if (!newNames.Add(output))
                    throw _host.ExceptParam(nameof(outputs), $"Output column '{output}' specified multiple times");
                if (Session.Graph[output] == null)
                    throw _host.ExceptParam(nameof(outputs), $"Output column '{output}' does not exist in the model");
            }

            Inputs = inputs;
            TFInputTypes = new TFDataType[Inputs.Length];
            TFInputShapes = new TFShape[Inputs.Length];
            for (int i = 0; i < Inputs.Length; i++)
            {
                var tfInput = new TFOutput(Graph[Inputs[i]]);
                TFInputTypes[i] = tfInput.OutputType;
                TFInputShapes[i] = Graph.GetTensorShape(tfInput);
                if (TFInputShapes[i].NumDimensions != -1)
                {
                    var newShape = new long[TFInputShapes[i].NumDimensions];
                    newShape[0] = TFInputShapes[i][0] == -1 ? BatchSize : TFInputShapes[i][0];

                    for (int j = 1; j < TFInputShapes[i].NumDimensions; j++)
                        newShape[j] = TFInputShapes[i][j];
                    TFInputShapes[i] = new TFShape(newShape);
                }
            }

            Outputs = outputs;
            OutputTypes = new ColumnType[Outputs.Length];
            TFOutputTypes = new TFDataType[Outputs.Length];
            for (int i = 0; i < Outputs.Length; i++)
            {
                var tfOutput = new TFOutput(Graph[Outputs[i]]);
                var shape = Graph.GetTensorShape(tfOutput);
                int[] dims = shape.NumDimensions > 0 ? shape.ToIntArray().Skip(shape[0] == -1 ? 1 : 0).ToArray() : new[] { 0 };
                var type = TensorFlowUtils.Tf2MlNetType(tfOutput.OutputType);
                OutputTypes[i] = new VectorType(type, dims);
                TFOutputTypes[i] = tfOutput.OutputType;
            }
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            foreach (var input in Inputs)
            {
                if (!inputSchema.TryGetColumnIndex(input, out int srcCol))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
            }
            return Transform(new EmptyDataView(_host, inputSchema)).Schema;
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

            // *** Binary format ***
            // byte: indicator for frozen models
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            var isFrozen = string.IsNullOrEmpty(_savedModelPath);
            ctx.Writer.WriteBoolByte(isFrozen);
            if (isFrozen)
            {
                var buffer = new TFBuffer();
                Session.Graph.ToGraphDef(buffer);
                ctx.SaveBinaryStream("TFModel", w =>
                {
                    w.WriteByteArray(buffer.ToArray());
                });
            }
            else
            {
                ctx.SaveBinaryStream("TFSavedModel", w =>
                {
                    string[] modelFilePaths = Directory.GetFiles(_savedModelPath, "*", SearchOption.AllDirectories);
                    w.Write(modelFilePaths.Length);

                    foreach (var fullPath in modelFilePaths)
                    {
                        var relativePath = fullPath.Substring(_savedModelPath.Length + 1);
                        w.Write(relativePath);

                        using (var fs = new FileStream(fullPath, FileMode.Open))
                        {
                            long fileLength = fs.Length;
                            w.Write(fileLength);
                            long actualWritten = fs.CopyRange(w.BaseStream, fileLength);
                            _host.Assert(actualWritten == fileLength);
                        }
                    }
                });
            }
            _host.AssertNonEmpty(Inputs);
            ctx.Writer.Write(Inputs.Length);
            foreach (var colName in Inputs)
                ctx.SaveNonEmptyString(colName);

            _host.AssertNonEmpty(Outputs);
            ctx.Writer.Write(Outputs.Length);
            foreach (var colName in Outputs)
                ctx.SaveNonEmptyString(colName);
        }

        ~TensorFlowTransform()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            // Ensure that the Session is not null and it's handle is not Zero, as it may have already been disposed/finalized.
            // Technically we shouldn't be calling this if disposing == false, since we're running in finalizer
            // and the GC doesn't guarantee ordering of finalization of managed objects, but we have to make sure
            // that the Session is closed before deleting our temporary directory.
            try
            {
                if (Session?.Handle != IntPtr.Zero)
                {
                    Session.CloseSession();
                    Session.Dispose();
                }
            }
            finally
            {
                if (!string.IsNullOrEmpty(_savedModelPath) && _isTemporarySavedModel)
                {
                    TensorFlowUtils.DeleteFolderWithRetries(_host, _savedModelPath);
                }
            }
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
            private readonly ISchema _schema;
            private readonly TensorFlowTransform _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly TFShape[] _fullySpecifiedShapes;

            public Mapper(IHostEnvironment env, TensorFlowTransform parent, ISchema inputSchema)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(Mapper));
                _host.CheckValue(inputSchema, nameof(inputSchema));
                _host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _schema = inputSchema;
                _inputColIndices = new int[_parent.Inputs.Length];
                _isInputVector = new bool[_parent.Inputs.Length];
                _fullySpecifiedShapes = new TFShape[_parent.Inputs.Length];
                for (int i = 0; i < _parent.Inputs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.Inputs[i], out _inputColIndices[i]))
                        throw _host.Except($"Column {_parent.Inputs[i]} doesn't exist");

                    var type = inputSchema.GetColumnType(_inputColIndices[i]);
                    if (type.IsVector && type.VectorSize == 0)
                        throw _host.Except($"Variable length input columns not supported");

                    _isInputVector[i] = type.IsVector;
                    var expectedType = TensorFlowUtils.Tf2MlNetType(_parent.TFInputTypes[i]);
                    if (type.ItemType != expectedType)
                        throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i], expectedType.ToString(), type.ToString());
                    var originalShape = _parent.TFInputShapes[i];
                    var shape = originalShape.ToIntArray();

                    var colTypeDims = Enumerable.Range(0, type.AsVector.DimCount + 1).Select(d => d == 0 ? 1 : (long)type.AsVector.GetDim(d - 1)).ToArray();
                    if (shape == null)
                        _fullySpecifiedShapes[i] = new TFShape(colTypeDims);
                    else if (type.AsVector.DimCount == 1)
                    {
                        // If the column is one dimension we make sure that the total size of the TF shape matches.
                        // Compute the total size of the known dimensions of the shape.
                        int valCount = shape.Where(x => x > 0).Aggregate((x, y) => x * y);
                        // The column length should be divisible by this, so that the other dimensions can be integral.
                        if (type.ValueCount % valCount != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {type.ValueCount}.");

                        // If the shape is multi-dimensional, we should be able to create the length of the vector by plugging
                        // in a single value for the unknown shapes. E.g., if the shape is [?,?,3], then there should exist a value
                        // d such that d*d*3 is equal to the length of the input column.
                        var d = originalShape.NumDimensions > 2 ? Math.Pow(type.ValueCount / valCount, 1.0 / (originalShape.NumDimensions - 2)) : 1;
                        if (originalShape.NumDimensions > 2 && d - (int)d != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {type.ValueCount}.");

                        // Fill in the unknown dimensions.
                        var l = new long[originalShape.NumDimensions];
                        for (int ishape = 0; ishape < originalShape.NumDimensions; ishape++)
                            l[ishape] = originalShape[ishape] == -1 ? (int)d : originalShape[ishape];
                        _fullySpecifiedShapes[i] = new TFShape(l);
                    }
                    else
                    {
                        if (shape.Select((dim, j) => dim != -1 && dim != colTypeDims[j]).Any(b => b))
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {originalShape.ToString()}, but input data is {type.AsVector.ToString()}.");

                        // Fill in the unknown dimensions.
                        var l = new long[originalShape.NumDimensions];
                        for (int ishape = 0; ishape < originalShape.NumDimensions; ishape++)
                            l[ishape] = originalShape[ishape] == -1 ? colTypeDims[ishape] : originalShape[ishape];
                        _fullySpecifiedShapes[i] = new TFShape(l);
                    }
                }
            }

            public void Save(ModelSaveContext ctx)
            {
                _parent.Save(ctx);
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

            private Delegate[] MakeGetters(IRow input, Func<int, bool> activeOutput)
            {
                _host.AssertValue(input);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();

                var valueGetters = new Delegate[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                {
                    if (activeOutput(i))
                    {
                        var type = TFTensor.TypeFromTensorType(_parent.TFOutputTypes[i]);
                        _host.Assert(type == _parent.OutputTypes[i].ItemType.RawType);
                        var srcTensorGetters = GetTensorValueGetters(input, _inputColIndices, _isInputVector, _parent.TFInputTypes, _fullySpecifiedShapes);
                        valueGetters[i] = Utils.MarshalInvoke(MakeGetter<int>, type, input, i, srcTensorGetters, activeOutputColNames, outputCache);
                    }
                }
                return valueGetters;
            }

            private Delegate MakeGetter<T>(IRow input, int iinfo, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                _host.AssertValue(input);
                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                    var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                    var tensorSize = tensor.Shape.Where(x => x > 0).Aggregate((x, y) => x * y);

                    var values = dst.Values;
                    if (Utils.Size(values) < tensorSize)
                        values = new T[tensorSize];

                    TensorFlowUtils.FetchData<T>(tensor.Data, values);
                    dst = new VBuffer<T>(values.Length, values, dst.Indices);
                };
                return valuegetter;
            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    var runner = _parent.Session.GetRunner();
                    for (int i = 0; i < _inputColIndices.Length; i++)
                    {
                        var inputName = _parent.Inputs[i];
                        runner.AddInput(inputName, srcTensorGetters[i].GetTensor());
                    }

                    var tensors = runner.Fetch(activeOutputColNames).Run();
                    Contracts.Assert(tensors.Length > 0);

                    for (int j = 0; j < tensors.Length; j++)
                        outputCache.Outputs[activeOutputColNames[j]] = tensors[j];

                    outputCache.Position = position;
                }
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                using (var ch = _host.Start("CreateGetters"))
                {
                    var getters = MakeGetters(input, activeOutput);
                    ch.Done();
                    return getters;
                }
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            public RowMapperColumnInfo[] GetOutputColumns()
            {
                var info = new RowMapperColumnInfo[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                    info[i] = new RowMapperColumnInfo(_parent.Outputs[i], _parent.OutputTypes[i], null);
                return info;
            }
        }

        [TlcModule.EntryPoint(Name = "Transforms.TensorFlowScorer",
            Desc = Summary,
            UserName = UserName,
            ShortName = ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.TensorFlow/doc.xml' path='doc/members/member[@name=""TensorflowTransform""]/*' />" })]
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

        private interface ITensorValueGetter
        {
            TFTensor GetTensor();

            void BufferTrainingData();

            TFTensor GetBufferedBatchTensor();
        }

        private class TensorValueGetter<T> : ITensorValueGetter
        {
            private readonly ValueGetter<T> _srcgetter;
            private readonly List<T> _bufferedData;
            private readonly TFShape _tfShape;

            public TensorValueGetter(IRow input, int colIndex, TFShape tfShape)
            {
                _srcgetter = input.GetGetter<T>(colIndex);
                _tfShape = tfShape;
                _bufferedData = new List<T>();
            }

            public TFTensor GetTensor()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                return TFTensor.CreateScalar(scalar);
            }

            public void BufferTrainingData()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                _bufferedData.Add(scalar);
            }

            public TFTensor GetBufferedBatchTensor()
            {
                var tensor = TFTensor.Create(_bufferedData.ToArray(), _bufferedData.Count, _tfShape);
                _bufferedData.Clear();
                return tensor;
            }
        }

        private class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private readonly TFShape _tfShape;
            private VBuffer<T> _vBuffer;
            private VBuffer<T> _vBufferDense;
            private readonly List<T> _bufferedData;

            public TensorValueGetterVec(IRow input, int colIndex, TFShape tfShape)
            {
                _srcgetter = input.GetGetter<VBuffer<T>>(colIndex);
                _tfShape = tfShape;
                _vBuffer = default;
                _vBufferDense = default;
                _bufferedData = new List<T>();
            }

            public TFTensor GetTensor()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyToDense(ref _vBufferDense);
                return TFTensor.Create(_vBufferDense.Values, _vBufferDense.Length, _tfShape);
            }

            public void BufferTrainingData()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyToDense(ref _vBufferDense);
                _bufferedData.AddRange(_vBufferDense.DenseValues());
            }

            public TFTensor GetBufferedBatchTensor()
            {
                var tensor = TFTensor.Create(_bufferedData.ToArray(), _bufferedData.Count, _tfShape);
                _bufferedData.Clear();
                return tensor;
            }
        }
    }

    public sealed class TensorFlowEstimator : TrivialEstimator<TensorFlowTransform>
    {
        public TensorFlowEstimator(IHostEnvironment env, string model, string[] inputs, string[] outputs)
           : this(env, new TensorFlowTransform(env, TensorFlowUtils.GetSession(env, model), inputs, outputs, TensorFlowUtils.IsSavedModel(env, model) ? model : null, false))
        {
        }

        public TensorFlowEstimator(IHostEnvironment env, TensorFlowTransform transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TensorFlowTransform)), transformer)
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
                if (!(col.Kind == SchemaShape.Column.VectorKind.Vector))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, nameof(VectorType), col.GetTypeString());
                var expectedType = TensorFlowUtils.Tf2MlNetType(Transformer.TFInputTypes[i]);
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

    public static class TensorFlowStaticExtensions
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
                return new TensorFlowEstimator(env, _modelFile, new[] { inputNames[outCol.Input] }, new[] { outputNames[outCol] });
            }
        }

        // REVIEW: this method only covers one use case of using TensorFlow models: consuming one
        // input and producing one output of floats.
        // We could consider selectively adding some more extensions to enable common scenarios.
        /// <summary>
        /// Run a TensorFlow model on the input column and extract one output column.
        /// The inputs and outputs are matched to TensorFlow graph nodes by name.
        /// </summary>
        public static Vector<float> ApplyTensorFlowGraph(this Vector<float> input, string modelFile)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckNonEmpty(modelFile, nameof(modelFile));
            return new OutColumn(input, modelFile);
        }
    }
}
