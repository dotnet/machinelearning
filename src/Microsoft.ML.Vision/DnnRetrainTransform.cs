// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TensorFlow;
using Microsoft.ML.Transforms;
using NumSharp;
using Tensorflow;
using static Microsoft.ML.TensorFlow.TensorFlowUtils;
using static Tensorflow.Binding;
using Utils = Microsoft.ML.Internal.Utilities.Utils;

[assembly: LoadableClass(DnnRetrainTransformer.Summary, typeof(IDataTransform), typeof(DnnRetrainTransformer),
    typeof(DnnRetrainEstimator.Options), typeof(SignatureDataTransform), DnnRetrainTransformer.UserName, DnnRetrainTransformer.ShortName)]

[assembly: LoadableClass(DnnRetrainTransformer.Summary, typeof(IDataTransform), typeof(DnnRetrainTransformer), null, typeof(SignatureLoadDataTransform),
    DnnRetrainTransformer.UserName, DnnRetrainTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(DnnRetrainTransformer), null, typeof(SignatureLoadModel),
    DnnRetrainTransformer.UserName, DnnRetrainTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(DnnRetrainTransformer), null, typeof(SignatureLoadRowMapper),
    DnnRetrainTransformer.UserName, DnnRetrainTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// <see cref="ITransformer" /> for the <see cref="DnnRetrainEstimator"/>.
    /// </summary>
    internal sealed class DnnRetrainTransformer : RowToRowTransformerBase, IDisposable
    {
        private bool _isDisposed;

        private readonly IHostEnvironment _env;
        private readonly string _modelLocation;
        private readonly bool _isTemporarySavedModel;
        private readonly bool _addBatchDimensionInput;
        private readonly Session _session;
        private readonly DataViewType[] _outputTypes;
        private readonly TF_DataType[] _tfOutputTypes;
        private readonly TF_DataType[] _tfInputTypes;
        private readonly TensorShape[] _tfInputShapes;
        private readonly (Operation, int)[] _tfInputOperations;
        private readonly (Operation, int)[] _tfOutputOperations;
        private readonly TF_Output[] _tfInputNodes;
        private readonly TF_Output[] _tfOutputNodes;
        private Graph Graph => _session.graph;
        private readonly Dictionary<string, string> _idvToTfMapping;
        private readonly string[] _inputs;
        private readonly string[] _outputs;

        internal const string Summary = "Re-Trains Dnn models.";
        internal const string UserName = "DnnRtTransform";
        internal const string ShortName = "DnnRtTransform";
        internal const string LoaderSignature = "DnnRtTransform";

        internal static class DefaultModelFileNames
        {
            public const string VariablesFolder = "variables";
            public const string Index = "variables.index";
            public const string Data = "variables.data-00000-of-00001";
            public const string Graph = "saved_model.pb";
            public const string TmpMlnetModel = "mlnet_model";
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DNNTRANS",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00000001,
                verReadableCur: 0x00000001,
                verWeCanReadBack: 0x00000001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DnnRetrainTransformer).Assembly.FullName);
        }

        // Factory method for SignatureLoadModel.
        private static DnnRetrainTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // byte: indicator for adding batch dimension in input
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            // stream: tensorFlow model.

            GetModelInfo(env, ctx, out string[] inputs, out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput);

            if (isFrozen)
            {
                byte[] modelBytes = null;
                if (!ctx.TryLoadBinaryStream("TFModel", r => modelBytes = r.ReadByteArray()))
                    throw env.ExceptDecode();

                return new DnnRetrainTransformer(env, TensorFlowUtils.LoadTFSession(env, modelBytes), outputs, inputs,
                    null, false, addBatchDimensionInput, 1);
            }

            var tempDirPath = Path.GetFullPath(Path.Combine(((IHostEnvironmentInternal)env).TempFilePath, nameof(DnnRetrainTransformer) + "_" + Guid.NewGuid()));
            CreateFolderWithAclIfNotExists(env, tempDirPath);
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
                            CreateFolderWithAclIfNotExists(env, fullFileDir);
                        }
                        using (var fs = new FileStream(fullFilePath, FileMode.Create, FileAccess.Write))
                        {
                            long actualRead = br.BaseStream.CopyRange(fs, fileLength);
                            env.Assert(actualRead == fileLength);
                        }
                    }
                });

                return new DnnRetrainTransformer(env, GetSession(env, tempDirPath), outputs, inputs, tempDirPath, true,
                    addBatchDimensionInput, 1);
            }
            catch (Exception)
            {
                DeleteFolderWithRetries(env, tempDirPath);
                throw;
            }
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, DnnRetrainEstimator.Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.InputColumns, nameof(options.InputColumns));
            env.CheckValue(options.OutputColumns, nameof(options.OutputColumns));

            return new DnnRetrainTransformer(env, options, input).MakeDataTransform(input);
        }

        internal DnnRetrainTransformer(IHostEnvironment env, DnnRetrainEstimator.Options options, IDataView input)
            : this(env, options, LoadDnnModel(env, options.ModelLocation), input)
        {
        }

        internal DnnRetrainTransformer(IHostEnvironment env, DnnRetrainEstimator.Options options, ML.TensorFlow.TensorFlowSessionWrapper tensorFlowModel, IDataView input, IDataView validationSet = null)
            : this(env, tensorFlowModel.Session, options.OutputColumns, options.InputColumns,
                  options.ModelLocation, false, options.AddBatchDimensionInputs, options.BatchSize)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            CheckTrainingParameters(options);

            if (!IsSavedModel(env, options.ModelLocation))
                throw env.ExceptNotSupp("TensorFlowTransform: Re-Training of TensorFlow model is only supported for un-frozen model.");

            TrainCore(options, input, validationSet);
        }

        private void CheckTrainingParameters(DnnRetrainEstimator.Options options)
        {
            Host.CheckNonWhiteSpace(options.LabelColumn, nameof(options.LabelColumn));
            Host.CheckNonWhiteSpace(options.OptimizationOperation, nameof(options.OptimizationOperation));
            if (_session.graph.OperationByName(options.OptimizationOperation) == null)
                throw Host.ExceptParam(nameof(options.OptimizationOperation), $"Optimization operation '{options.OptimizationOperation}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.TensorFlowLabel, nameof(options.TensorFlowLabel));
            if (_session.graph.OperationByName(options.TensorFlowLabel) == null)
                throw Host.ExceptParam(nameof(options.TensorFlowLabel), $"'{options.TensorFlowLabel}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.SaveLocationOperation, nameof(options.SaveLocationOperation));
            if (_session.graph.OperationByName(options.SaveLocationOperation) == null)
                throw Host.ExceptParam(nameof(options.SaveLocationOperation), $"'{options.SaveLocationOperation}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.SaveOperation, nameof(options.SaveOperation));
            if (_session.graph.OperationByName(options.SaveOperation) == null)
                throw Host.ExceptParam(nameof(options.SaveOperation), $"'{options.SaveOperation}' does not exist in the model");

            if (options.LossOperation != null)
            {
                Host.CheckNonWhiteSpace(options.LossOperation, nameof(options.LossOperation));
                if (_session.graph.OperationByName(options.LossOperation) == null)
                    throw Host.ExceptParam(nameof(options.LossOperation), $"'{options.LossOperation}' does not exist in the model");
            }

            if (options.MetricOperation != null)
            {
                Host.CheckNonWhiteSpace(options.MetricOperation, nameof(options.MetricOperation));
                if (_session.graph.OperationByName(options.MetricOperation) == null)
                    throw Host.ExceptParam(nameof(options.MetricOperation), $"'{options.MetricOperation}' does not exist in the model");
            }

            if (options.LearningRateOperation != null)
            {
                Host.CheckNonWhiteSpace(options.LearningRateOperation, nameof(options.LearningRateOperation));
                if (_session.graph.OperationByName(options.LearningRateOperation) == null)
                    throw Host.ExceptParam(nameof(options.LearningRateOperation), $"'{options.LearningRateOperation}' does not exist in the model");
            }
        }

        private (int, bool, TF_DataType, TensorShape) GetTrainingInputInfo(DataViewSchema inputSchema, string columnName, string tfNodeName, int batchSize)
        {
            if (!inputSchema.TryGetColumnIndex(columnName, out int inputColIndex))
                throw Host.Except($"Column {columnName} doesn't exist");

            var type = inputSchema[inputColIndex].Type;
            var isInputVector = type is VectorDataViewType;

            (Operation inputTensor, int index) = GetOperationFromName(tfNodeName, _session);
            var tfInput = new TF_Input(inputTensor, index);
            var tfInputType = inputTensor.OpType == "Placeholder" ? inputTensor.OutputType(index) :
                inputTensor.InputType(index);
            var tfInputShape = ((Tensor)inputTensor).TensorShape;

            var numInputDims = tfInputShape != null ? tfInputShape.ndim : -1;
            if (isInputVector && (tfInputShape == null || (numInputDims == 0)))
            {
                var vecType = (VectorDataViewType)type;
                var colTypeDims = new int[vecType.Dimensions.Length + 1];
                colTypeDims[0] = -1;
                for (int indexLocal = 0; indexLocal < vecType.Dimensions.Length; indexLocal += 1)
                    colTypeDims[indexLocal + 1] = vecType.Dimensions[indexLocal];

                tfInputShape = new TensorShape(colTypeDims);
            }
            if (numInputDims != -1)
            {
                var newShape = new int[numInputDims];
                var dims = tfInputShape.dims;
                newShape[0] = dims[0] == 0 || dims[0] == -1 ? batchSize : dims[0];

                for (int j = 1; j < numInputDims; j++)
                    newShape[j] = dims[j];
                tfInputShape = new TensorShape(newShape);
            }

            var expectedType = Tf2MlNetType(tfInputType);
            var actualType = type.GetItemType().RawType;
            if (type is KeyDataViewType && actualType == typeof(UInt32))
                actualType = typeof(Int64);

            if (actualType != expectedType.RawType)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", columnName, expectedType.ToString(), type.ToString());

            return (inputColIndex, isInputVector, tfInputType, tfInputShape);
        }

        private void TrainCore(DnnRetrainEstimator.Options options, IDataView input, IDataView validationSet)
        {
            var inputsForTraining = new string[_inputs.Length + 1];
            var inputColIndices = new int[inputsForTraining.Length];
            var isInputVector = new bool[inputsForTraining.Length];
            var tfInputTypes = new TF_DataType[inputsForTraining.Length];
            var tfInputShapes = new TensorShape[inputsForTraining.Length];

            for (int i = 0; i < _inputs.Length; i++)
                inputsForTraining[i] = _idvToTfMapping[_inputs[i]];

            var inputSchema = input.Schema;
            for (int i = 0; i < inputsForTraining.Length - 1; i++)
                (inputColIndices[i], isInputVector[i], tfInputTypes[i], tfInputShapes[i]) =
                    GetTrainingInputInfo(inputSchema, _inputs[i], inputsForTraining[i], options.BatchSize);

            var index = inputsForTraining.Length - 1;
            inputsForTraining[index] = options.TensorFlowLabel;

            (inputColIndices[index], isInputVector[index], tfInputTypes[index], tfInputShapes[index]) =
                    GetTrainingInputInfo(inputSchema, options.LabelColumn, inputsForTraining[index], options.BatchSize);

            // Create graph inputs.
            Operation labelOp;
            int labelOpIdx;
            (labelOp, labelOpIdx) = GetOperationFromName(options.TensorFlowLabel, _session);
            TF_Output[] tfInputs;
            if (!string.IsNullOrEmpty(options.LearningRateOperation))
                tfInputs = new TF_Output[_tfInputNodes.Length + 2]; //Inputs + Label + Learning Rate.
            else
                tfInputs = new TF_Output[_tfInputNodes.Length + 1]; //Inputs + Label.

            Array.Copy(_tfInputNodes, tfInputs, _tfInputNodes.Length);

            tfInputs[_tfInputNodes.Length] = new TF_Output(labelOp, labelOpIdx);
            var lr = GetOperationFromName(options.LearningRateOperation, _session);
            tfInputs[_tfInputNodes.Length + 1] = new TF_Output(lr.Item1, lr.Item2);

            // Create graph operations.
            IntPtr[] ops = null;
            if (options.OptimizationOperation != null)
                ops = new[] { c_api.TF_GraphOperationByName(Graph, options.OptimizationOperation) };

            // Instantiate the graph.
            string[] outputs = null;
            if (options.LossOperation != null && options.MetricOperation != null)
                outputs = new[] { options.LossOperation, options.MetricOperation };
            else if (options.LossOperation != null)
                outputs = new[] { options.LossOperation };
            else if (options.MetricOperation != null)
                outputs = new[] { options.MetricOperation };

            Runner runner = new Runner(_session, new[] { options.LearningRateOperation }.Concat(inputsForTraining).ToArray(),
                outputs, new[] { options.OptimizationOperation }).AddInput(new Tensor(options.LearningRate), 0);

            var cols = input.Schema.Where(c => inputColIndices.Contains(c.Index));

            for (int epoch = 0; epoch < options.Epoch; epoch++)
            {
                using (var cursor = input.GetRowCursor(cols))
                {
                    var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, tfInputTypes, tfInputShapes);
                    bool isDataLeft = false;
                    using (var ch = Host.Start("Training TensorFlow model..."))
                    using (var pch = Host.StartProgressChannel("TensorFlow training progress..."))
                    {
                        float loss = 0;
                        float metric = 0;
                        pch.SetHeader(new ProgressHeader(new[] { "Loss", "Metric" }, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                        while (cursor.MoveNext())
                        {
                            for (int i = 0; i < inputsForTraining.Length; i++)
                            {
                                isDataLeft = true;
                                srcTensorGetters[i].BufferTrainingData();
                            }

                            if (((cursor.Position + 1) % options.BatchSize) == 0)
                            {
                                isDataLeft = false;
                                var (l, m) = ExecuteGraphAndRetrieveMetrics(inputsForTraining, srcTensorGetters, runner);
                                loss += l;
                                metric += m;
                            }
                        }
                        if (isDataLeft)
                        {
                            isDataLeft = false;
                            ch.Warning("Not training on the last batch. The batch size is less than {0}.", options.BatchSize);
                        }
                        pch.Checkpoint(new double?[] { loss, metric });
                    }
                }
            }

            UpdateModelOnDisk(options.ModelLocation, options);
        }

        private (float loss, float metric) ExecuteGraphAndRetrieveMetrics(
            string[] inputs,
            ITensorValueGetter[] srcTensorGetters,
            Runner runner)
        {
            float loss = 0.0f;
            float metric = 0.0f;
            for (int i = 0; i < inputs.Length; i++)
                runner.AddInput(srcTensorGetters[i].GetBufferedBatchTensor(), i + 1);

            Tensor[] tensor = runner.Run();
            if (tensor.Length > 0 && tensor[0] != IntPtr.Zero)
            {
                tensor[0].ToScalar<float>(ref loss);
                tensor[0].Dispose();
            }

            if (tensor.Length > 1 && tensor[1] != IntPtr.Zero)
            {
                tensor[1].ToScalar<float>(ref metric);
                tensor[1].Dispose();
            }

            return (loss, metric);
        }

        /// <summary>
        /// Updates the model on the disk.
        /// After retraining Session and Graphs are both up-to-date
        /// However model on disk is not which is used to serialzed to ML.Net stream
        /// </summary>
        private void UpdateModelOnDisk(string modelDir, DnnRetrainEstimator.Options options)
        {
            try
            {
                // Save the model on disk
                var path = Path.Combine(modelDir, DefaultModelFileNames.TmpMlnetModel);
                //var input = GetOperationFromName(options.SaveLocationOperation, Session);
                var runner = new Runner(_session, new[] { options.SaveLocationOperation },
                    null, new[] { options.SaveOperation }).AddInput(new Tensor(path), 0);

                runner.Run();
                // Preserve original files
                var variablesPath = Path.Combine(modelDir, DefaultModelFileNames.VariablesFolder);
                var archivePath = Path.Combine(variablesPath + "-" + Guid.NewGuid().ToString());
                Directory.CreateDirectory(archivePath);
                foreach (var f in Directory.GetFiles(variablesPath))
                    File.Copy(f, Path.Combine(archivePath, Path.GetFileName(f)));

                string[] modelFilePaths = null;

                // There are two ways parameters are saved depending on
                // either `saver_def = tf.train.Saver().as_saver_def()` was called in Python before `tf.saved_model.simple_save` or not.
                // If `saver_def = tf.train.Saver().as_saver_def()` was called files are saved in top directory.
                // If not then temporary directory is created in current directory which starts with `mlnet_model`
                // and files are saved there.
                var tmpParamDir = Directory.GetDirectories(modelDir, DefaultModelFileNames.TmpMlnetModel + "*");
                if (tmpParamDir != null && tmpParamDir.Length > 0)
                    modelFilePaths = Directory.GetFiles(tmpParamDir[0]);
                else
                    modelFilePaths = Directory.GetFiles(modelDir, DefaultModelFileNames.TmpMlnetModel + "*");

                foreach (var file in modelFilePaths)
                {
                    if (file.EndsWith(".data-00000-of-00001"))
                    {
                        var destination = Path.Combine(variablesPath, DefaultModelFileNames.Data);
                        if (File.Exists(destination))
                            File.Delete(destination);
                        Directory.Move(file, destination);
                    }
                    if (file.EndsWith(".index"))
                    {
                        var destination = Path.Combine(variablesPath, DefaultModelFileNames.Index);
                        if (File.Exists(destination))
                            File.Delete(destination);
                        Directory.Move(file, destination);
                    }
                }

                if (tmpParamDir != null && tmpParamDir.Length > 0)
                    DeleteFolderWithRetries(Host, tmpParamDir[0]);
            }
            catch (Exception e)
            {
                throw Host.ExceptIO(e, "Error serializing TensorFlow retrained model to disk.");
            }
        }

        private static ITensorValueGetter CreateTensorValueGetter<T>(DataViewRow input, bool isVector, int colIndex, TensorShape tfShape, bool keyType = false)
        {
            if (isVector)
                return new TensorValueGetterVec<T>(input, colIndex, tfShape);
            return new TensorValueGetter<T>(input, colIndex, tfShape, keyType);
        }

        private static ITensorValueGetter CreateTensorValueGetter(DataViewRow input, TF_DataType tfType, bool isVector, int colIndex, TensorShape tfShape)
        {
            var type = Tf2MlNetType(tfType);
            if (input.Schema[colIndex].Type is KeyDataViewType && type.RawType == typeof(Int64))
                return Utils.MarshalInvoke(CreateTensorValueGetter<int>, typeof(UInt32), input, isVector, colIndex, tfShape, true);

            return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type.RawType, input, isVector, colIndex, tfShape, false);
        }

        private static ITensorValueGetter[] GetTensorValueGetters(
            DataViewRow input,
            int[] inputColIndices,
            bool[] isInputVector,
            TF_DataType[] tfInputTypes,
            TensorShape[] tfInputShapes)
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static void GetModelInfo(IHostEnvironment env, ModelLoadContext ctx, out string[] inputs,
            out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput)
        {
            isFrozen = ctx.Reader.ReadBoolByte();
            addBatchDimensionInput = ctx.Reader.ReadBoolByte();

            var numInputs = ctx.Reader.ReadInt32();
            env.CheckDecode(numInputs > 0);
            inputs = new string[numInputs];
            for (int j = 0; j < inputs.Length; j++)
                inputs[j] = ctx.LoadNonEmptyString();

            var numOutputs = ctx.Reader.ReadInt32();
            env.CheckDecode(numOutputs > 0);
            outputs = new string[numOutputs];
            for (int j = 0; j < outputs.Length; j++)
                outputs[j] = ctx.LoadNonEmptyString();
        }

        internal DnnRetrainTransformer(IHostEnvironment env, Session session, string[] outputColumnNames,
            string[] inputColumnNames, string modelLocation, bool isTemporarySavedModel,
            bool addBatchDimensionInput, int batchSize)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(DnnRetrainTransformer)))

        {
            Host.CheckValue(session, nameof(session));
            Host.CheckNonEmpty(inputColumnNames, nameof(inputColumnNames));
            Host.CheckNonEmpty(outputColumnNames, nameof(outputColumnNames));

            _env = env;
            _session = session;
            _modelLocation = Path.IsPathRooted(modelLocation) ? modelLocation : Path.Combine(Directory.GetCurrentDirectory(), modelLocation);
            _isTemporarySavedModel = isTemporarySavedModel;
            _addBatchDimensionInput = addBatchDimensionInput;
            _inputs = inputColumnNames;
            _outputs = outputColumnNames;
            _idvToTfMapping = new Dictionary<string, string>();

            foreach (var x in _inputs)
                _idvToTfMapping[x] = x;

            foreach (var x in _outputs)
                _idvToTfMapping[x] = x;

            (_tfOutputTypes, _outputTypes, _tfOutputOperations) = GetOutputInfo(Host, _session, _outputs);

            (_tfInputTypes, _tfInputShapes, _tfInputOperations) = GetInputInfo(Host, _session, _inputs.Select(x => _idvToTfMapping[x]).ToArray(), batchSize);

            _tfInputNodes = new TF_Output[_inputs.Length];
            _tfOutputNodes = new TF_Output[_outputs.Length];

            for (int index = 0; index < _tfInputOperations.Length; index += 1)
                _tfInputNodes[index] = new TF_Output(_tfInputOperations[index].Item1, _tfInputOperations[index].Item2);

            for (int index = 0; index < _tfOutputOperations.Length; index += 1)
                _tfOutputNodes[index] = new TF_Output(_tfOutputOperations[index].Item1, _tfOutputOperations[index].Item2);
        }

        private static (Operation, int) GetOperationFromName(string operation, Session session)
        {
            var p = operation.IndexOf(':');

            if (p != -1 && p != operation.Length - 1)
            {
                var op = operation.Substring(0, p);
                if (int.TryParse(operation.Substring(p + 1), out var idx))
                {

                    return (session.graph.OperationByName(op), idx);
                }
            }
            return (session.graph.OperationByName(operation), 0);
        }

        internal static (TF_DataType[] tfInputTypes, TensorShape[] tfInputShapes, (Operation, int)[]) GetInputInfo(IHost host, Session session, string[] inputs, int batchSize = 1)
        {
            var tfInputTypes = new TF_DataType[inputs.Length];
            var tfInputShapes = new TensorShape[inputs.Length];
            var tfInputOperations = new (Operation, int)[inputs.Length];

            int index = 0;
            foreach (var input in inputs)
            {
                host.CheckNonWhiteSpace(input, nameof(inputs));
                (Operation inputTensor, int inputTensorIndex) = GetOperationFromName(input, session);

                if (inputTensor == null)
                    throw host.ExceptParam(nameof(inputs), $"Input column '{input}' does not exist in the model");

                TF_DataType tfInputType = string.Compare(inputTensor.OpType, "PlaceHolder", true) == 0 ? inputTensor.OutputType(inputTensorIndex) : inputTensor.InputType(index);
                if (!IsTypeSupported(tfInputType))
                    throw host.ExceptParam(nameof(session), $"Input type '{tfInputType}' of input column '{input}' is not supported in TensorFlow");

                tfInputTypes[index] = tfInputType;
                tfInputShapes[index] = ((Tensor)inputTensor).TensorShape;
                tfInputOperations[index] = (inputTensor, inputTensorIndex);
                index++;
            }

            return (tfInputTypes, tfInputShapes, tfInputOperations);
        }

        internal static TensorShape GetTensorShape(TF_Output output, Graph graph, Status status = null)
        {
            if (graph == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(graph));

            var cstatus = status == null ? new Status() : status;
            var n = c_api.TF_GraphGetTensorNumDims(graph, output, cstatus.Handle);

            cstatus.Check();

            if (n == -1)
                return new TensorShape(new int[0]);

            var dims = new long[n];
            c_api.TF_GraphGetTensorShape(graph, output, dims, dims.Length, cstatus.Handle);
            cstatus.Check();
            return new TensorShape(dims.Select(x => (int)x).ToArray());
        }

        internal static (TF_DataType[] tfOutputTypes, DataViewType[] outputTypes, (Operation, int)[]) GetOutputInfo(IHost host, Session session, string[] outputs)
        {
            var tfOutputTypes = new TF_DataType[outputs.Length];
            var outputTypes = new DataViewType[outputs.Length];
            var newNames = new HashSet<string>();
            var tfOutputOperations = new (Operation, int)[outputs.Length];

            for (int i = 0; i < outputs.Length; i++)
            {
                host.CheckNonWhiteSpace(outputs[i], nameof(outputs));
                if (!newNames.Add(outputs[i]))
                    throw host.ExceptParam(nameof(outputs), $"Output column '{outputs[i]}' specified multiple times");

                (Tensor outputTensor, int outputIndex) = GetOperationFromName(outputs[i], session);
                if (outputTensor == null)
                    throw host.ExceptParam(nameof(outputs), $"Output column '{outputs[i]}' does not exist in the model");

                var tfOutputType = ((Operation)outputTensor).OutputType(outputIndex);
                var shape = GetTensorShape(new TF_Output((Operation)outputTensor, outputIndex), session.graph);

                // The transformer can only retrieve the output as fixed length vector with shape of kind [-1, d1, d2, d3, ...]
                // i.e. the first dimension (if unknown) is assumed to be batch dimension.
                // If there are other dimension that are unknown the transformer will return a variable length vector.
                // This is the work around in absence of reshape transformer.
                int[] dims = shape.ndim > 0 ? shape.dims.Skip(shape.dims[0] == -1 ? 1 : 0).ToArray() : new[] { 0 };
                for (int j = 0; j < dims.Length; j++)
                    dims[j] = dims[j] == -1 ? 0 : dims[j];
                if (dims == null || dims.Length == 0)
                {
                    dims = new[] { 1 };
                    outputTypes[i] = Tf2MlNetType(tfOutputType);
                }
                else
                {
                    var type = Tf2MlNetType(tfOutputType);
                    outputTypes[i] = new VectorDataViewType(type, dims);
                }

                tfOutputTypes[i] = tfOutputType;
                tfOutputOperations[i] = (outputTensor, outputIndex);
            }

            return (tfOutputTypes, outputTypes, tfOutputOperations);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // byte: indicator for adding batch dimension in input
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            // stream: tensorFlow model.
            var isFrozen = !IsSavedModel(_env, _modelLocation);
            ctx.Writer.WriteBoolByte(isFrozen);
            ctx.Writer.WriteBoolByte(_addBatchDimensionInput);

            Host.AssertNonEmpty(_inputs);
            ctx.Writer.Write(_inputs.Length);
            foreach (var colName in _inputs)
                ctx.SaveNonEmptyString(colName);

            Host.AssertNonEmpty(_outputs);
            ctx.Writer.Write(_outputs.Length);
            foreach (var colName in _outputs)
                ctx.SaveNonEmptyString(colName);

            ctx.SaveBinaryStream("TFSavedModel", w =>
            {
                // only these files need to be saved.
                string[] modelFilePaths =
                {
                    Path.Combine(_modelLocation, DefaultModelFileNames.Graph),
                    Path.Combine(_modelLocation, DefaultModelFileNames.VariablesFolder, DefaultModelFileNames.Data),
                    Path.Combine(_modelLocation, DefaultModelFileNames.VariablesFolder, DefaultModelFileNames.Index),
                };

                w.Write(modelFilePaths.Length);

                foreach (var fullPath in modelFilePaths)
                {
                    var relativePath = fullPath.Substring(_modelLocation.Length + 1);
                    w.Write(relativePath);

                    using (var fs = new FileStream(fullPath, FileMode.Open))
                    {
                        long fileLength = fs.Length;
                        w.Write(fileLength);
                        long actualWritten = fs.CopyRange(w.BaseStream, fileLength);
                        Host.Assert(actualWritten == fileLength);
                    }
                }
            });
        }

        public void Dispose()
        {
            if (_isDisposed)
                return;

            // Ensure that the Session is not null and it's handle is not Zero, as it may have already been disposed/finalized.
            // Technically we shouldn't be calling this if disposing == false, since we're running in finalizer
            // and the GC doesn't guarantee ordering of finalization of managed objects, but we have to make sure
            // that the Session is closed before deleting our temporary directory.
            try
            {
                if (_session != null && _session != IntPtr.Zero)
                {
                    if (_session.graph != null)
                        _session.graph.Dispose();
                    _session.close();
                }
            }
            finally
            {
                if (IsSavedModel(_env, _modelLocation) && _isTemporarySavedModel)
                {
                    DeleteFolderWithRetries(Host, _modelLocation);
                }

                _isDisposed = true;
            }
        }

        private sealed class Mapper : MapperBase
        {
            private readonly DnnRetrainTransformer _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly TensorShape[] _fullySpecifiedShapes;
            private readonly ConcurrentBag<Runner> _runners;

            public Mapper(DnnRetrainTransformer parent, DataViewSchema inputSchema) :
                   base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                Host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _inputColIndices = new int[_parent._inputs.Length];
                _isInputVector = new bool[_parent._inputs.Length];
                _fullySpecifiedShapes = new TensorShape[_parent._inputs.Length];
                for (int i = 0; i < _parent._inputs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent._inputs[i], out _inputColIndices[i]))
                        throw Host.ExceptSchemaMismatch(nameof(InputSchema), "source", _parent._inputs[i]);

                    var type = inputSchema[_inputColIndices[i]].Type;
                    if (type is VectorDataViewType vecType && vecType.Size == 0)
                        throw Host.Except("Variable length input columns not supported");

                    _isInputVector[i] = type is VectorDataViewType;
                    if (!_isInputVector[i])
                        throw Host.Except("Non-vector columns are not supported and should be loaded as vector columns of size 1");
                    vecType = (VectorDataViewType)type;
                    var expectedType = Tf2MlNetType(_parent._tfInputTypes[i]);
                    if (type.GetItemType() != expectedType)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent._inputs[i], expectedType.ToString(), type.ToString());
                    var originalShape = _parent._tfInputShapes[i];
                    var shape = originalShape.dims;

                    var colTypeDims = vecType.Dimensions.Select(dim => (int)dim).ToArray();
                    if (shape == null || (shape.Length == 0))
                        _fullySpecifiedShapes[i] = new TensorShape(colTypeDims);
                    else
                    {
                        // If the column is one dimension we make sure that the total size of the TF shape matches.
                        // Compute the total size of the known dimensions of the shape.
                        int valCount = 1;
                        int numOfUnkDim = 0;
                        foreach (var s in shape)
                        {
                            if (s > 0)
                                valCount *= s;
                            else
                                numOfUnkDim++;
                        }
                        // The column length should be divisible by this, so that the other dimensions can be integral.
                        int typeValueCount = type.GetValueCount();
                        if (typeValueCount % valCount != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent._inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // If the shape is multi-dimensional, we should be able to create the length of the vector by plugging
                        // in a single value for the unknown shapes. For example, if the shape is [?,?,3], then there should exist a value
                        // d such that d*d*3 is equal to the length of the input column.
                        var d = numOfUnkDim > 0 ? Math.Pow(typeValueCount / valCount, 1.0 / numOfUnkDim) : 0;
                        if (d - (int)d != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent._inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // Fill in the unknown dimensions.
                        var originalShapeDims = originalShape.dims;
                        var originalShapeNdim = originalShape.ndim;
                        var l = new int[originalShapeNdim];
                        for (int ishape = 0; ishape < originalShapeNdim; ishape++)
                            l[ishape] = originalShapeDims[ishape] == -1 ? (int)d : originalShapeDims[ishape];
                        _fullySpecifiedShapes[i] = new TensorShape(l);
                    }

                    if (_parent._addBatchDimensionInput)
                    {
                        var l = new int[_fullySpecifiedShapes[i].ndim + 1];
                        l[0] = 1;
                        for (int ishape = 1; ishape < l.Length; ishape++)
                            l[ishape] = _fullySpecifiedShapes[i].dims[ishape - 1];
                        _fullySpecifiedShapes[i] = new TensorShape(l);
                    }
                }

                _runners = new ConcurrentBag<Runner>();
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            private class OutputCache
            {
                public long Position;
                public Dictionary<string, Tensor> Outputs;
                public OutputCache()
                {
                    Position = -1;
                    Outputs = new Dictionary<string, Tensor>();
                }
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent._outputs.Where((x, i) => activeOutput(i)).ToArray();

                var type = Tf2MlNetType(_parent._tfOutputTypes[iinfo]).RawType;
                Host.Assert(type == _parent._outputTypes[iinfo].GetItemType().RawType);
                var srcTensorGetters = GetTensorValueGetters(input, _inputColIndices, _isInputVector, _parent._tfInputTypes, _fullySpecifiedShapes);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, iinfo, srcTensorGetters, activeOutputColNames, outputCache);
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache) where T : unmanaged
            {
                Host.AssertValue(input);

                if (_parent._outputTypes[iinfo].IsStandardScalar())
                {
                    ValueGetter<T> valuegetter = (ref T dst) =>
                    {
                        UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                        var tensor = outputCache.Outputs[_parent._outputs[iinfo]];
                        tensor.ToScalar<T>(ref dst);
                    };
                    return valuegetter;
                }
                else
                {
                    if (_parent._tfOutputTypes[iinfo] == TF_DataType.TF_STRING)
                    {
                        ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                        {
                            UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                            var tensor = outputCache.Outputs[_parent._outputs[iinfo]];
                            var tensorSize = tensor.TensorShape.dims.Where(x => x > 0).Aggregate((x, y) => x * y);

                            var editor = VBufferEditor.Create(ref dst, (int)tensorSize);
                            FetchStringData(tensor, editor.Values);
                            dst = editor.Commit();
                        };
                        return valuegetter;
                    }
                    else
                    {
                        ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                        {
                            UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                            var tensor = outputCache.Outputs[_parent._outputs[iinfo]];
                            var tensorSize = tensor.TensorShape.dims.Where(x => x > 0).Aggregate((x, y) => x * y);

                            var editor = VBufferEditor.Create(ref dst, (int)tensorSize);

                            tensor.CopyTo<T>(editor.Values);
                            dst = editor.Commit();
                        };
                        return valuegetter;
                    }
                }
            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    if (_parent.Graph.graph_key != tf.get_default_graph().graph_key)
                        _parent._session.graph.as_default();

                    Runner runner = new Runner(_parent._session,
                        _parent._inputs.Select(x => _parent._idvToTfMapping[x]).ToArray(),
                        _parent._outputs.Select(x => _parent._idvToTfMapping[x]).ToArray());

                    // Feed the inputs.
                    for (int i = 0; i < _parent._inputs.Length; i++)
                        runner.AddInput(srcTensorGetters[i].GetTensor(), 0);

                    // Execute the graph.
                    var tensors = runner.Run();
                    Contracts.Assert(tensors.Length > 0);

                    for (int j = 0; j < activeOutputColNames.Length; j++)
                        outputCache.Outputs[activeOutputColNames[j]] = tensors[j];

                    outputCache.Position = position;
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent._outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[_parent._outputs.Length];
                for (int i = 0; i < _parent._outputs.Length; i++)
                    info[i] = new DataViewSchema.DetachedColumn(_parent._outputs[i], _parent._outputTypes[i], null);
                return info;
            }
        }

        private interface ITensorValueGetter
        {
            Tensor GetTensor();

            void BufferTrainingData();

            Tensor GetBufferedBatchTensor();
        }

        private class TensorValueGetter<T> : ITensorValueGetter
        {
            private readonly ValueGetter<T> _srcgetter;
            private readonly T[] _bufferedData;
            private readonly Int64[] _bufferedDataLong;
            private readonly TensorShape _tfShape;
            private int _position;
            private readonly bool _keyType;
            private readonly long[] _dims;

            public TensorValueGetter(DataViewRow input, int colIndex, TensorShape tfShape, bool keyType = false)
            {
                _srcgetter = input.GetGetter<T>(input.Schema[colIndex]);
                _tfShape = tfShape;
                long size = 0;
                _position = 0;
                if (tfShape.dims.Length != 0)
                {
                    size = 1;
                    foreach (var dim in tfShape.dims)
                        size *= dim;
                    _dims = _tfShape.dims.Select(x => (long)x).ToArray();
                }
                if (keyType)
                    _bufferedDataLong = new long[size];
                else
                    _bufferedData = new T[size];
                _keyType = keyType;
            }

            public Tensor GetTensor()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                if (_keyType)
                {
                    var tensor = new Tensor(new[] { Convert.ToInt64(scalar) - 1 });
                    tensor.set_shape(_tfShape);
                    return tensor;
                }
                else
                {
                    var tensor = new Tensor(new[] { scalar });
                    tensor.set_shape(_tfShape);
                    return tensor;
                }
            }

            public void BufferTrainingData()
            {
                if (_keyType)
                {
                    var scalar = default(T);
                    _srcgetter(ref scalar);
                    _bufferedDataLong[_position++] = Convert.ToInt64(scalar) - 1;
                }
                else
                {
                    var scalar = default(T);
                    _srcgetter(ref scalar);
                    _bufferedData[_position++] = scalar;
                }
            }

            public Tensor GetBufferedBatchTensor()
            {
                if (_keyType)
                {
                    var tensor = new Tensor(_bufferedDataLong, _dims, TF_DataType.TF_INT64);
                    _position = 0;
                    return tensor;
                }
                else
                {
                    var tensor = TensorFlowUtils.CastDataAndReturnAsTensor(_bufferedData, _tfShape);
                    _position = 0;
                    return tensor;
                }
            }
        }

        private class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private readonly TensorShape _tfShape;
            private VBuffer<T> _vBuffer;
            private T[] _denseData;
            private T[] _bufferedData;
            private int _position;
            private readonly long[] _dims;
            private readonly long _bufferedDataSize;

            public TensorValueGetterVec(DataViewRow input, int colIndex, TensorShape tfShape)
            {
                _srcgetter = input.GetGetter<VBuffer<T>>(input.Schema[colIndex]);
                _tfShape = tfShape;
                _vBuffer = default;
                _denseData = default;

                long size = 0;
                _position = 0;
                if (tfShape.dims.Length != 0)
                {
                    size = 1;
                    foreach (var dim in tfShape.dims)
                        size *= dim;
                }
                _bufferedData = new T[size];
                _bufferedDataSize = size;
                if (_tfShape.dims != null)
                    _dims = _tfShape.dims.Select(x => (long)x).ToArray();
            }

            public Tensor GetTensor()
            {
                _srcgetter(ref _vBuffer);

                // _denseData.Length can be greater than _vBuffer.Length sometime after
                // Utils.EnsureSize is executed. Use _vBuffer.Length to access the elements in _denseData.
                // This is done to reduce memory allocation every time tensor is created.
                _denseData = new T[_vBuffer.Length];
                _vBuffer.CopyTo(_denseData);
                return TensorFlowUtils.CastDataAndReturnAsTensor(_denseData, _tfShape);
            }

            public void BufferTrainingData()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyTo(_bufferedData, _position);
                _position += _vBuffer.Length;
            }

            public Tensor GetBufferedBatchTensor()
            {
                _position = 0;
                var tensor = TensorFlowUtils.CastDataAndReturnAsTensor(_bufferedData, _tfShape);
                _bufferedData = new T[_bufferedDataSize];
                return tensor;
            }
        }
    }

    internal sealed class DnnRetrainEstimator : IEstimator<DnnRetrainTransformer>
    {
        /// <summary>
        /// The options for the <see cref="DnnRetrainTransformer"/>.
        /// </summary>
        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// Location of the TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "TensorFlow model used by the transform. Please see https://www.tensorflow.org/mobile/prepare_models for more details.", SortOrder = 0)]
            public string ModelLocation;

            /// <summary>
            /// The names of the model inputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            /// <summary>
            /// The names of the requested model outputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the outputs", ShortName = "outputs", SortOrder = 2)]
            public string[] OutputColumns;

            /// <summary>
            /// The name of the label column in <see cref="IDataView"/> that will be mapped to label node in TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Training labels.", ShortName = "label", SortOrder = 4)]
            public string LabelColumn;

            /// <summary>
            /// The name of the label in TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "TensorFlow label node.", ShortName = "TFLabel", SortOrder = 5)]
            public string TensorFlowLabel;

            /// <summary>
            /// Name of the operation in TensorFlow graph that is used for optimizing parameters in the graph.
            /// Usually it is the name specified in the minimize method of optimizer in python
            /// e.g. optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name = "SGDOptimizer").
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the optimization operation in the TensorFlow graph.", ShortName = "OptimizationOp", SortOrder = 6)]
            public string OptimizationOperation;

            /// <summary>
            /// The name of the operation in the TensorFlow graph to compute training loss (Optional).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the operation in the TensorFlow graph to compute training loss (Optional)", ShortName = "LossOp", SortOrder = 7)]
            public string LossOperation;

            /// <summary>
            /// The name of the operation in the TensorFlow graph to compute performance metric during training (Optional).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the operation in the TensorFlow graph to compute performance metric during training (Optional)", ShortName = "MetricOp", SortOrder = 8)]
            public string MetricOperation;

            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of samples to use for mini-batch training.", SortOrder = 9)]
            public int BatchSize = 64;

            /// <summary>
            /// Number of training iterations.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of training iterations.", SortOrder = 10)]
            public int Epoch = 5;

            /// <summary>
            /// The name of the operation in the TensorFlow graph which sets optimizer learning rate (Optional).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the operation in the TensorFlow graph which sets optimizer learning rate (Optional).", SortOrder = 11)]
            public string LearningRateOperation;

            /// <summary>
            /// Learning rate to use during optimization.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate to use during optimization.", SortOrder = 12)]
            public float LearningRate = 0.01f;

            /// <summary>
            /// Name of the input in TensorFlow graph that specify the location for saving/restoring models to/from disk.
            /// This parameter is set by different kinds of 'Savers' in TensorFlow and users don't have control over this.
            /// Therefore, its highly unlikely that this parameter is changed from its default value of 'save/Const'.
            /// Please change it cautiously if you need to.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input in TensorFlow graph that specify the location for saving/restoring models from disk.", SortOrder = 13)]
            public string SaveLocationOperation = "save/Const";

            /// <summary>
            /// Name of the operation in TensorFlow graph that is used for saving/restoring models to/from disk.
            /// This parameter is set by different kinds of 'Savers' in TensorFlow and users don't have control over this.
            /// Therefore, its highly unlikely that this parameter is changed from its default value of 'save/control_dependency'.
            /// Please change it cautiously if you need to.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input in TensorFlow graph that specify the location for saving/restoring models from disk.", SortOrder = 14)]
            public string SaveOperation = "save/control_dependency";

            /// <summary>
            /// Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
            /// </summary>
            /// <remarks>
            /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.
            /// In this case, there is no way to induce shape from the model's inputs or input data.
            /// </remarks>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].", SortOrder = 16)]
            public bool AddBatchDimensionInputs = false;
        }

        private readonly IHost _host;
        private readonly Options _options;
        private readonly ML.TensorFlow.TensorFlowSessionWrapper _tensorFlowModel;
        private readonly TF_DataType[] _tfInputTypes;
        private readonly DataViewType[] _outputTypes;
        private DnnRetrainTransformer _transformer;

        internal DnnRetrainEstimator(IHostEnvironment env, Options options, ML.TensorFlow.TensorFlowSessionWrapper tensorFlowModel)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(DnnRetrainEstimator));
            _options = options;
            _tensorFlowModel = tensorFlowModel;
            var inputTuple = DnnRetrainTransformer.GetInputInfo(_host, tensorFlowModel.Session, options.InputColumns);
            _tfInputTypes = inputTuple.tfInputTypes;
            _outputTypes = DnnRetrainTransformer.GetOutputInfo(_host, tensorFlowModel.Session, options.OutputColumns).outputTypes;
        }

        private static Options CreateArguments(ML.TensorFlow.TensorFlowSessionWrapper tensorFlowModel, string[] outputColumnNames, string[] inputColumnName, bool addBatchDimensionInput)
        {
            var options = new Options();
            options.ModelLocation = tensorFlowModel.ModelPath;
            options.InputColumns = inputColumnName;
            options.OutputColumns = outputColumnNames;
            options.AddBatchDimensionInputs = addBatchDimensionInput;
            return options;
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            for (var i = 0; i < _options.InputColumns.Length; i++)
            {
                var input = _options.InputColumns[i];
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                if (!(col.Kind == SchemaShape.Column.VectorKind.Vector))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector", col.GetTypeString());
                var expectedType = Tf2MlNetType(_tfInputTypes[i]);
                if (col.ItemType != expectedType)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, expectedType.ToString(), col.ItemType.ToString());
            }
            for (var i = 0; i < _options.OutputColumns.Length; i++)
            {
                resultDic[_options.OutputColumns[i]] = new SchemaShape.Column(_options.OutputColumns[i],
                    _outputTypes[i].IsKnownSizeVector() ? SchemaShape.Column.VectorKind.Vector
                    : SchemaShape.Column.VectorKind.VariableVector, _outputTypes[i].GetItemType(), false);
            }
            return new SchemaShape(resultDic.Values);
        }

        /// <summary>
        /// Trains and returns a <see cref="DnnRetrainTransformer"/>.
        /// </summary>
        public DnnRetrainTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            if (_transformer == null)
                _transformer = new DnnRetrainTransformer(_host, _options, _tensorFlowModel, input);

            // Validate input schema.
            _transformer.GetOutputSchema(input.Schema);
            return _transformer;
        }
    }
}
