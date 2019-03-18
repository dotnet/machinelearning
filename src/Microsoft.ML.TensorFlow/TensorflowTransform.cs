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
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TensorFlow;

[assembly: LoadableClass(TensorFlowTransformer.Summary, typeof(IDataTransform), typeof(TensorFlowTransformer),
    typeof(TensorFlowEstimator.Options), typeof(SignatureDataTransform), TensorFlowTransformer.UserName, TensorFlowTransformer.ShortName)]

[assembly: LoadableClass(TensorFlowTransformer.Summary, typeof(IDataTransform), typeof(TensorFlowTransformer), null, typeof(SignatureLoadDataTransform),
    TensorFlowTransformer.UserName, TensorFlowTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(TensorFlowTransformer), null, typeof(SignatureLoadModel),
    TensorFlowTransformer.UserName, TensorFlowTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TensorFlowTransformer), null, typeof(SignatureLoadRowMapper),
    TensorFlowTransformer.UserName, TensorFlowTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(TensorFlowTransformer))]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="TensorflowTransformer"]/*' />
    public sealed class TensorFlowTransformer : RowToRowTransformerBase
    {
        private readonly string _savedModelPath;
        private readonly bool _isTemporarySavedModel;
        private readonly bool _addBatchDimensionInput;
        internal readonly TFSession Session;
        internal readonly DataViewType[] OutputTypes;
        internal readonly TFDataType[] TFOutputTypes;
        internal readonly TFDataType[] TFInputTypes;
        internal readonly TFShape[] TFInputShapes;
        internal TFGraph Graph => Session.Graph;

        internal readonly string[] Inputs;
        internal readonly string[] Outputs;

        internal static int BatchSize = 1;
        internal const string Summary = "Transforms the data using the TensorFlow model.";
        internal const string UserName = "TensorFlowTransform";
        internal const string ShortName = "TFTransform";
        internal const string LoaderSignature = "TensorFlowTransform";

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
                modelSignature: "TENSFLOW",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002,  // Added Support for Multiple Outputs and SavedModel.
                verWrittenCur: 0x00010003,  // Added Support for adding batch dimension in inputs.
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TensorFlowTransformer).Assembly.FullName);
        }

        /// <summary>
        /// Transform for scoring Tensorflow models. Input data column names/types must exactly match
        /// all model input names. Only the output columns specified will be generated.
        /// If the model is already loaded please <see cref="TensorFlowTransformer(IHostEnvironment, TensorFlowModel, string, string, bool)"/> to avoid reloading of model.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="outputColumnName">The output columns to generate. Names must match model specifications. Data types are inferred from model.</param>
        /// <param name="inputColumnName">The name of the input data column. Must match model input name. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        internal TensorFlowTransformer(IHostEnvironment env, string modelFile, string outputColumnName, string inputColumnName = null, bool addBatchDimensionInput = false)
            : this(env, TensorFlowUtils.GetSession(env, modelFile), new[] { outputColumnName }, new[] { inputColumnName ?? outputColumnName }, TensorFlowUtils.IsSavedModel(env, modelFile) ? modelFile : null, false, addBatchDimensionInput)
        {
        }

        /// <summary>
        /// Transform for scoring Tensorflow models. Input data column names/types must exactly match
        /// all model input names. Only the output columns specified will be generated.
        /// If the model is already loaded please <see cref="TensorFlowTransformer(IHostEnvironment, TensorFlowModel, string[], string[], bool)"/> to avoid reloading of model.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelFile">Model file path.</param>
        /// <param name="inputColumnNames">The name of the input data columns. Must match model's input names.</param>
        /// <param name="outputColumnNames">The output columns to generate. Names must match model specifications. Data types are inferred from model.</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        internal TensorFlowTransformer(IHostEnvironment env, string modelFile, string[] outputColumnNames, string[] inputColumnNames, bool addBatchDimensionInput = false)
            : this(env, TensorFlowUtils.GetSession(env, modelFile), outputColumnNames, inputColumnNames, TensorFlowUtils.IsSavedModel(env, modelFile) ? modelFile : null, false, addBatchDimensionInput)
        {
        }

        /// <summary>
        /// Transform for scoring Tensorflow models. Input data column names/types must exactly match
        /// all model input names. Only the output columns specified will be generated.
        /// This convenience method avoids reloading of TensorFlow model.
        /// It is useful in a situation where user has already loaded TensorFlow model using <see cref="TensorFlowUtils.LoadTensorFlowModel(IHostEnvironment, string)"/> for inspecting model schema.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="tfModelInfo"> <see cref="TensorFlowModel"/> object created with <see cref="TensorFlowUtils.LoadTensorFlowModel(IHostEnvironment, string)"/>.</param>
        /// <param name="outputColumnName">The output columns to generate. Names must match model specifications. Data types are inferred from model.</param>
        /// <param name="inputColumnName">The name of the input data columns. Must match model's input names. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        internal TensorFlowTransformer(IHostEnvironment env, TensorFlowModel tfModelInfo, string outputColumnName, string inputColumnName = null, bool addBatchDimensionInput = false)
            : this(env, tfModelInfo.Session, new[] { outputColumnName }, new[] { inputColumnName ?? outputColumnName }, TensorFlowUtils.IsSavedModel(env, tfModelInfo.ModelPath) ? tfModelInfo.ModelPath : null, false, addBatchDimensionInput)
        {
        }

        /// <summary>
        /// Transform for scoring Tensorflow models. Input data column names/types must exactly match
        /// all model input names. Only the output columns specified will be generated.
        /// This convenience method avoids reloading of TensorFlow model.
        /// It is useful in a situation where user has already loaded TensorFlow model using <see cref="TensorFlowUtils.LoadTensorFlowModel(IHostEnvironment, string)"/> for inspecting model schema.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="tfModelInfo"> <see cref="TensorFlowModel"/> object created with <see cref="TensorFlowUtils.LoadTensorFlowModel(IHostEnvironment, string)"/>.</param>
        /// <param name="inputColumnNames">The name of the input data columns. Must match model's input names.</param>
        /// <param name="outputColumnNames">The output columns to generate. Names must match model specifications. Data types are inferred from model.</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        internal TensorFlowTransformer(IHostEnvironment env, TensorFlowModel tfModelInfo, string[] outputColumnNames, string[] inputColumnNames, bool addBatchDimensionInput = false)
            : this(env, tfModelInfo.Session, outputColumnNames, inputColumnNames, TensorFlowUtils.IsSavedModel(env, tfModelInfo.ModelPath) ? tfModelInfo.ModelPath : null, false, addBatchDimensionInput)
        {
        }

        // Factory method for SignatureLoadModel.
        private static TensorFlowTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // byte: indicator for adding batch dimension in input
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            GetModelInfo(env, ctx, out string[] inputs, out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput);
            if (isFrozen)
            {
                byte[] modelBytes = null;
                if (!ctx.TryLoadBinaryStream("TFModel", r => modelBytes = r.ReadByteArray()))
                    throw env.ExceptDecode();
                return new TensorFlowTransformer(env, TensorFlowUtils.LoadTFSession(env, modelBytes), outputs, inputs, null, false, addBatchDimensionInput);
            }

            var tempDirPath = Path.GetFullPath(Path.Combine(Path.GetTempPath(), nameof(TensorFlowTransformer) + "_" + Guid.NewGuid()));
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

                return new TensorFlowTransformer(env, TensorFlowUtils.GetSession(env, tempDirPath), outputs, inputs, tempDirPath, true, addBatchDimensionInput);
            }
            catch (Exception)
            {
                TensorFlowUtils.DeleteFolderWithRetries(env, tempDirPath);
                throw;
            }
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, TensorFlowEstimator.Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.InputColumns, nameof(options.InputColumns));
            env.CheckValue(options.OutputColumns, nameof(options.OutputColumns));

            return new TensorFlowTransformer(env, options, input).MakeDataTransform(input);
        }

        internal TensorFlowTransformer(IHostEnvironment env, TensorFlowEstimator.Options options, IDataView input)
            : this(env, options, TensorFlowUtils.LoadTensorFlowModel(env, options.ModelLocation), input)
        {
        }

        internal TensorFlowTransformer(IHostEnvironment env, TensorFlowEstimator.Options options, TensorFlowModel tensorFlowModel, IDataView input)
            : this(env, tensorFlowModel.Session, options.OutputColumns, options.InputColumns, TensorFlowUtils.IsSavedModel(env, options.ModelLocation) ? options.ModelLocation : null, false, options.AddBatchDimensionInputs)
        {

            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));

            if (options.ReTrain)
            {
                env.CheckValue(input, nameof(input));

                CheckTrainingParameters(options);

                if (!TensorFlowUtils.IsSavedModel(env, options.ModelLocation))
                    throw env.ExceptNotSupp("TensorFlowTransform: Re-Training of TensorFlow model is only supported for un-frozen model.");
                TrainCore(options, input);
            }
        }

        private void CheckTrainingParameters(TensorFlowEstimator.Options options)
        {
            Host.CheckNonWhiteSpace(options.LabelColumn, nameof(options.LabelColumn));
            Host.CheckNonWhiteSpace(options.OptimizationOperation, nameof(options.OptimizationOperation));
            if (Session.Graph[options.OptimizationOperation] == null)
                throw Host.ExceptParam(nameof(options.OptimizationOperation), $"Optimization operation '{options.OptimizationOperation}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.TensorFlowLabel, nameof(options.TensorFlowLabel));
            if (Session.Graph[options.TensorFlowLabel] == null)
                throw Host.ExceptParam(nameof(options.TensorFlowLabel), $"'{options.TensorFlowLabel}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.SaveLocationOperation, nameof(options.SaveLocationOperation));
            if (Session.Graph[options.SaveLocationOperation] == null)
                throw Host.ExceptParam(nameof(options.SaveLocationOperation), $"'{options.SaveLocationOperation}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.SaveOperation, nameof(options.SaveOperation));
            if (Session.Graph[options.SaveOperation] == null)
                throw Host.ExceptParam(nameof(options.SaveOperation), $"'{options.SaveOperation}' does not exist in the model");

            if (options.LossOperation != null)
            {
                Host.CheckNonWhiteSpace(options.LossOperation, nameof(options.LossOperation));
                if (Session.Graph[options.LossOperation] == null)
                    throw Host.ExceptParam(nameof(options.LossOperation), $"'{options.LossOperation}' does not exist in the model");
            }

            if (options.MetricOperation != null)
            {
                Host.CheckNonWhiteSpace(options.MetricOperation, nameof(options.MetricOperation));
                if (Session.Graph[options.MetricOperation] == null)
                    throw Host.ExceptParam(nameof(options.MetricOperation), $"'{options.MetricOperation}' does not exist in the model");
            }

            if (options.LearningRateOperation != null)
            {
                Host.CheckNonWhiteSpace(options.LearningRateOperation, nameof(options.LearningRateOperation));
                if (Session.Graph[options.LearningRateOperation] == null)
                    throw Host.ExceptParam(nameof(options.LearningRateOperation), $"'{options.LearningRateOperation}' does not exist in the model");
            }
        }

        private (int, bool, TFDataType, TFShape) GetTrainingInputInfo(DataViewSchema inputSchema, string columnName, string tfNodeName, int batchSize)
        {
            if (!inputSchema.TryGetColumnIndex(columnName, out int inputColIndex))
                throw Host.Except($"Column {columnName} doesn't exist");

            var type = inputSchema[inputColIndex].Type;
            var isInputVector = type is VectorType;

            var tfInput = new TFOutput(Graph[tfNodeName]);
            var tfInputType = tfInput.OutputType;
            var tfInputShape = Graph.GetTensorShape(tfInput);
            if (tfInputShape.NumDimensions != -1)
            {
                var newShape = new long[tfInputShape.NumDimensions];
                newShape[0] = tfInputShape[0] == -1 ? batchSize : tfInputShape[0];

                for (int j = 1; j < tfInputShape.NumDimensions; j++)
                    newShape[j] = tfInputShape[j];
                tfInputShape = new TFShape(newShape);
            }

            var expectedType = TensorFlowUtils.Tf2MlNetType(tfInputType);
            if (type.GetItemType() != expectedType)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", columnName, expectedType.ToString(), type.ToString());

            return (inputColIndex, isInputVector, tfInputType, tfInputShape);
        }

        private void TrainCore(TensorFlowEstimator.Options options, IDataView input)
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

            var inputSchema = input.Schema;
            for (int i = 0; i < inputsForTraining.Length - 1; i++)
            {
                (inputColIndices[i], isInputVector[i], tfInputTypes[i], tfInputShapes[i]) =
                    GetTrainingInputInfo(inputSchema, inputsForTraining[i], inputsForTraining[i], options.BatchSize);
            }

            var index = inputsForTraining.Length - 1;
            inputsForTraining[index] = options.TensorFlowLabel;
            (inputColIndices[index], isInputVector[index], tfInputTypes[index], tfInputShapes[index]) =
                    GetTrainingInputInfo(inputSchema, options.LabelColumn, inputsForTraining[index], options.BatchSize);

            var fetchList = new List<string>();
            if (options.LossOperation != null)
                fetchList.Add(options.LossOperation);
            if (options.MetricOperation != null)
                fetchList.Add(options.MetricOperation);

            var cols = input.Schema.Where(c => inputColIndices.Contains(c.Index));
            for (int epoch = 0; epoch < options.Epoch; epoch++)
            {
                using (var cursor = input.GetRowCursor(cols))
                {
                    var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, tfInputTypes, tfInputShapes);

                    float loss = 0;
                    float metric = 0;
                    bool isDataLeft = false;
                    using (var ch = Host.Start("Training TensorFlow model..."))
                    using (var pch = Host.StartProgressChannel("TensorFlow training progress..."))
                    {
                        pch.SetHeader(new ProgressHeader(new[] { "Loss", "Metric" }, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                        while (cursor.MoveNext())
                        {
                            for (int i = 0; i < inputColIndices.Length; i++)
                            {
                                isDataLeft = true;
                                srcTensorGetters[i].BufferTrainingData();
                            }

                            if (((cursor.Position + 1) % options.BatchSize) == 0)
                            {
                                isDataLeft = false;
                                var (l, m) = TrainBatch(inputColIndices, inputsForTraining, srcTensorGetters, fetchList, options);
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

        private (float loss, float metric) TrainBatch(int[] inputColIndices,
            string[] inputsForTraining,
            ITensorValueGetter[] srcTensorGetters,
            List<string> fetchList,
            TensorFlowEstimator.Options options)
        {
            float loss = 0;
            float metric = 0;
            var runner = Session.GetRunner();
            for (int i = 0; i < inputColIndices.Length; i++)
            {
                var inputName = inputsForTraining[i];
                runner.AddInput(inputName, srcTensorGetters[i].GetBufferedBatchTensor());
            }

            if (options.LearningRateOperation != null)
                runner.AddInput(options.LearningRateOperation, new TFTensor(options.LearningRate));
            runner.AddTarget(options.OptimizationOperation);

            if (fetchList.Count > 0)
                runner.Fetch(fetchList.ToArray());

            var tensor = runner.Run();
            loss = tensor.Length > 0 ? (float)tensor[0].GetValue() : 0.0f;
            metric = tensor.Length > 1 ? (float)tensor[1].GetValue() : 0.0f;

            return (loss, metric);
        }

        /// <summary>
        /// Updates the model on the disk.
        /// After retraining Session and Graphs are both up-to-date
        /// However model on disk is not which is used to serialzed to ML.Net stream
        /// </summary>
        private void UpdateModelOnDisk(string modelDir, TensorFlowEstimator.Options options)
        {
            try
            {
                // Save the model on disk
                var path = Path.Combine(modelDir, DefaultModelFileNames.TmpMlnetModel);
                Session.GetRunner().AddInput(options.SaveLocationOperation, TFTensor.CreateString(Encoding.UTF8.GetBytes(path)))
                        .AddTarget(options.SaveOperation).Run();

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
                    TensorFlowUtils.DeleteFolderWithRetries(Host, tmpParamDir[0]);
            }
            catch (Exception e)
            {
                throw Host.ExceptIO(e, "Error serializing TensorFlow retrained model to disk.");
            }
        }

        private static ITensorValueGetter CreateTensorValueGetter<T>(DataViewRow input, bool isVector, int colIndex, TFShape tfShape)
        {
            if (isVector)
                return new TensorValueGetterVec<T>(input, colIndex, tfShape);
            return new TensorValueGetter<T>(input, colIndex, tfShape);
        }

        private static ITensorValueGetter CreateTensorValueGetter(DataViewRow input, TFDataType tfType, bool isVector, int colIndex, TFShape tfShape)
        {
            var type = TFTensor.TypeFromTensorType(tfType);
            Contracts.AssertValue(type);
            return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type, input, isVector, colIndex, tfShape);
        }

        private static ITensorValueGetter[] GetTensorValueGetters(DataViewRow input,
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static void GetModelInfo(IHostEnvironment env, ModelLoadContext ctx, out string[] inputs, out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput)
        {
            isFrozen = true;
            bool isNonFrozenModelSupported = ctx.Header.ModelVerReadable >= 0x00010002;
            if (isNonFrozenModelSupported)
                isFrozen = ctx.Reader.ReadBoolByte();

            addBatchDimensionInput = false;
            bool isAddingBatchDimensionSupported = ctx.Header.ModelVerReadable >= 0x00010003;
            if (isAddingBatchDimensionSupported)
                addBatchDimensionInput = ctx.Reader.ReadBoolByte();

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

        internal TensorFlowTransformer(IHostEnvironment env, TFSession session, string[] outputColumnNames, string[] inputColumnNames, string savedModelPath, bool isTemporarySavedModel, bool addBatchDimensionInput) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TensorFlowTransformer)))

        {
            Host.CheckValue(session, nameof(session));
            Host.CheckNonEmpty(inputColumnNames, nameof(inputColumnNames));
            Host.CheckNonEmpty(outputColumnNames, nameof(outputColumnNames));

            Session = session;
            _savedModelPath = savedModelPath;
            _isTemporarySavedModel = isTemporarySavedModel;
            _addBatchDimensionInput = addBatchDimensionInput;
            Inputs = inputColumnNames;
            Outputs = outputColumnNames;

            (TFInputTypes, TFInputShapes) = GetInputInfo(Host, Session, Inputs);
            (TFOutputTypes, OutputTypes) = GetOutputInfo(Host, Session, Outputs);
        }

        internal static (TFDataType[] tfInputTypes, TFShape[] tfInputShapes) GetInputInfo(IHost host, TFSession session, string[] inputs)
        {
            var tfInputTypes = new TFDataType[inputs.Length];
            var tfInputShapes = new TFShape[inputs.Length];

            foreach (var input in inputs)
            {
                host.CheckNonWhiteSpace(input, nameof(inputs));
                if (session.Graph[input] == null)
                    throw host.ExceptParam(nameof(inputs), $"Input column '{input}' does not exist in the model");
                var tfInput = new TFOutput(session.Graph[input]);
                if (!TensorFlowUtils.IsTypeSupported(tfInput.OutputType))
                    throw host.ExceptParam(nameof(session), $"Input type '{tfInput.OutputType}' of input column '{input}' is not supported in TensorFlow");
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                var tfInput = new TFOutput(session.Graph[inputs[i]]);
                tfInputTypes[i] = tfInput.OutputType;
                tfInputShapes[i] = session.Graph.GetTensorShape(tfInput);
            }
            return (tfInputTypes, tfInputShapes);
        }

        internal static (TFDataType[] tfOutputTypes, DataViewType[] outputTypes) GetOutputInfo(IHost host, TFSession session, string[] outputs)
        {
            var tfOutputTypes = new TFDataType[outputs.Length];
            var outputTypes = new DataViewType[outputs.Length];
            var newNames = new HashSet<string>();
            foreach (var output in outputs)
            {
                host.CheckNonWhiteSpace(output, nameof(outputs));
                if (!newNames.Add(output))
                    throw host.ExceptParam(nameof(outputs), $"Output column '{output}' specified multiple times");
                if (session.Graph[output] == null)
                    throw host.ExceptParam(nameof(outputs), $"Output column '{output}' does not exist in the model");
            }

            for (int i = 0; i < outputs.Length; i++)
            {
                var tfOutput = new TFOutput(session.Graph[outputs[i]]);
                var shape = session.Graph.GetTensorShape(tfOutput);

                // The transformer can only retreive the output as fixed length vector with shape of kind [-1, d1, d2, d3, ...]
                // i.e. the first dimension (if unknown) is assumed to be batch dimension.
                // If there are other dimension that are unknown the transformer will return a variable length vector.
                // This is the work around in absence of reshape transformer.
                int[] dims = shape.NumDimensions > 0 ? shape.ToIntArray().Skip(shape[0] == -1 ? 1 : 0).ToArray() : new[] { 0 };
                for (int j = 0; j < dims.Length; j++)
                    dims[j] = dims[j] == -1 ? 0 : dims[j];
                var type = TensorFlowUtils.Tf2MlNetType(tfOutput.OutputType);
                outputTypes[i] = new VectorType(type, dims);
                tfOutputTypes[i] = tfOutput.OutputType;
            }

            return (tfOutputTypes, outputTypes);
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
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            var isFrozen = string.IsNullOrEmpty(_savedModelPath);
            ctx.Writer.WriteBoolByte(isFrozen);
            ctx.Writer.WriteBoolByte(_addBatchDimensionInput);
            if (isFrozen)
            {
                var buffer = new TFBuffer();
                Session.Graph.ToGraphDef(buffer);
                ctx.SaveBinaryStream("TFModel", w =>
                {
                    w.WriteByteArray(buffer.ToSpan());
                });
            }
            else
            {
                ctx.SaveBinaryStream("TFSavedModel", w =>
                {
                    // only these files need to be saved.
                    string[] modelFilePaths =
                    {
                        Path.Combine(_savedModelPath, DefaultModelFileNames.Graph),
                        Path.Combine(_savedModelPath, DefaultModelFileNames.VariablesFolder, DefaultModelFileNames.Data),
                        Path.Combine(_savedModelPath, DefaultModelFileNames.VariablesFolder, DefaultModelFileNames.Index),
                    };

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
                            Host.Assert(actualWritten == fileLength);
                        }
                    }
                });
            }
            Host.AssertNonEmpty(Inputs);
            ctx.Writer.Write(Inputs.Length);
            foreach (var colName in Inputs)
                ctx.SaveNonEmptyString(colName);

            Host.AssertNonEmpty(Outputs);
            ctx.Writer.Write(Outputs.Length);
            foreach (var colName in Outputs)
                ctx.SaveNonEmptyString(colName);
        }

        ~TensorFlowTransformer()
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
                    TensorFlowUtils.DeleteFolderWithRetries(Host, _savedModelPath);
                }
            }
        }

        private sealed class Mapper : MapperBase
        {
            private readonly TensorFlowTransformer _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly TFShape[] _fullySpecifiedShapes;

            public Mapper(TensorFlowTransformer parent, DataViewSchema inputSchema) :
                   base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                Host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _inputColIndices = new int[_parent.Inputs.Length];
                _isInputVector = new bool[_parent.Inputs.Length];
                _fullySpecifiedShapes = new TFShape[_parent.Inputs.Length];
                for (int i = 0; i < _parent.Inputs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.Inputs[i], out _inputColIndices[i]))
                        throw Host.ExceptSchemaMismatch(nameof(InputSchema), "source", _parent.Inputs[i]);

                    var type = inputSchema[_inputColIndices[i]].Type;
                    if (type is VectorType vecType && vecType.Size == 0)
                        throw Host.Except("Variable length input columns not supported");

                    _isInputVector[i] = type is VectorType;
                    if (!_isInputVector[i]) // Temporary pending fix of issue #1542. In its current state, the below code would fail anyway with a naked exception if this check was not here.
                        throw Host.Except("Non-vector columns not supported");
                    vecType = (VectorType)type;
                    var expectedType = TensorFlowUtils.Tf2MlNetType(_parent.TFInputTypes[i]);
                    if (type.GetItemType() != expectedType)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i], expectedType.ToString(), type.ToString());
                    var originalShape = _parent.TFInputShapes[i];
                    var shape = originalShape.ToIntArray();

                    var colTypeDims = vecType.Dimensions.Select(dim => (long)dim).ToArray();
                    if (shape == null)
                        _fullySpecifiedShapes[i] = new TFShape(colTypeDims);
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
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // If the shape is multi-dimensional, we should be able to create the length of the vector by plugging
                        // in a single value for the unknown shapes. For example, if the shape is [?,?,3], then there should exist a value
                        // d such that d*d*3 is equal to the length of the input column.
                        var d = numOfUnkDim > 0 ? Math.Pow(typeValueCount / valCount, 1.0 / numOfUnkDim) : 0;
                        if (d - (int)d != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // Fill in the unknown dimensions.
                        var l = new long[originalShape.NumDimensions];
                        for (int ishape = 0; ishape < originalShape.NumDimensions; ishape++)
                            l[ishape] = originalShape[ishape] == -1 ? (int)d : originalShape[ishape];
                        _fullySpecifiedShapes[i] = new TFShape(l);
                    }

                    if (_parent._addBatchDimensionInput)
                    {
                        var l = new long[_fullySpecifiedShapes[i].NumDimensions + 1];
                        l[0] = 1L;
                        for (int ishape = 1; ishape < l.Length; ishape++)
                            l[ishape] = _fullySpecifiedShapes[i][ishape-1];
                        _fullySpecifiedShapes[i] = new TFShape(l);
                    }
                }
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

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

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();

                var type = TFTensor.TypeFromTensorType(_parent.TFOutputTypes[iinfo]);
                Host.Assert(type == _parent.OutputTypes[iinfo].GetItemType().RawType);
                var srcTensorGetters = GetTensorValueGetters(input, _inputColIndices, _isInputVector, _parent.TFInputTypes, _fullySpecifiedShapes);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, iinfo, srcTensorGetters, activeOutputColNames, outputCache);
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                Host.AssertValue(input);
                if (_parent.TFOutputTypes[iinfo] == TFDataType.String)
                {
                    ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                    {
                        UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                        var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                        var tensorSize = tensor.Shape.Where(x => x > 0).Aggregate((x, y) => x * y);

                        var editor = VBufferEditor.Create(ref dst, (int)tensorSize);
                        TensorFlowUtils.FetchStringData(tensor, editor.Values);
                        dst = editor.Commit();
                    };
                    return valuegetter;
                }
                else
                {
                    ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                    {
                        UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                        var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                        var tensorSize = tensor.Shape.Where(x => x > 0).Aggregate((x, y) => x * y);

                        var editor = VBufferEditor.Create(ref dst, (int)tensorSize);
                        TensorFlowUtils.FetchData<T>(tensor.Data, editor.Values);
                        dst = editor.Commit();
                    };
                    return valuegetter;
                }
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

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                    info[i] = new DataViewSchema.DetachedColumn(_parent.Outputs[i], _parent.OutputTypes[i], null);
                return info;
            }
        }

        [TlcModule.EntryPoint(Name = "Transforms.TensorFlowScorer",
            Desc = Summary,
            UserName = UserName,
            ShortName = ShortName)]
        internal static CommonOutputs.TransformOutput TensorFlowScorer(IHostEnvironment env, TensorFlowEstimator.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "TensorFlow", input);
            var view = Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
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
            private readonly T[] _bufferedData;
            private readonly TFShape _tfShape;
            private int _position;

            public TensorValueGetter(DataViewRow input, int colIndex, TFShape tfShape)
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
                }
                _bufferedData = new T[size];
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
                _bufferedData[_position++] = scalar;
            }

            public TFTensor GetBufferedBatchTensor()
            {
                var tensor = TFTensor.Create(_bufferedData, _bufferedData.Length, _tfShape);
                _position = 0;
                return tensor;
            }
        }

        private class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private readonly TFShape _tfShape;
            private VBuffer<T> _vBuffer;
            private T[] _denseData;
            private readonly T[] _bufferedData;
            private int _position;

            public TensorValueGetterVec(DataViewRow input, int colIndex, TFShape tfShape)
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
            }

            public TFTensor GetTensor()
            {
                _srcgetter(ref _vBuffer);

                // _denseData.Length can be greater than _vBuffer.Length sometime after
                // Utils.EnsureSize is exectued. Use _vBuffer.Length to access the elements in _denseData.
                // This is done to reduce memory allocation every time tensor is created.
                Utils.EnsureSize(ref _denseData, _vBuffer.Length, keepOld: false);
                _vBuffer.CopyTo(_denseData);

                return TFTensor.Create(_denseData, _vBuffer.Length, _tfShape);
            }

            public void BufferTrainingData()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyTo(_bufferedData, _position);
                _position += _vBuffer.Length;
            }

            public TFTensor GetBufferedBatchTensor()
            {
                var tensor = TFTensor.Create(_bufferedData, _bufferedData.Length, _tfShape);
                _position = 0;
                return tensor;
            }
        }
    }

    /// <summary>
    /// Estimator for the <see cref="TensorFlowTransformer"/>.
    /// </summary>
    public sealed class TensorFlowEstimator : IEstimator<TensorFlowTransformer>
    {
        /// <summary>
        /// The options for the <see cref="TensorFlowTransformer"/>.
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
            /// Name of the input in TensorFlow graph that specifiy the location for saving/restoring models to/from disk.
            /// This parameter is set by different kinds of 'Savers' in TensorFlow and users don't have control over this.
            /// Therefore, its highly unlikely that this parameter is changed from its default value of 'save/Const'.
            /// Please change it cautiously if you need to.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input in TensorFlow graph that specifiy the location for saving/restoring models from disk.", SortOrder = 13)]
            public string SaveLocationOperation = "save/Const";

            /// <summary>
            /// Name of the operation in TensorFlow graph that is used for saving/restoring models to/from disk.
            /// This parameter is set by different kinds of 'Savers' in TensorFlow and users don't have control over this.
            /// Therefore, its highly unlikely that this parameter is changed from its default value of 'save/control_dependency'.
            /// Please change it cautiously if you need to.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input in TensorFlow graph that specifiy the location for saving/restoring models from disk.", SortOrder = 14)]
            public string SaveOperation = "save/control_dependency";

            /// <summary>
            /// Needed for command line to specify if retraining is requested.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Retrain TensorFlow model.", SortOrder = 15)]
            public bool ReTrain = false;

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
        private readonly TensorFlowModel _tensorFlowModel;
        private readonly TFDataType[] _tfInputTypes;
        private readonly DataViewType[] _outputTypes;
        private TensorFlowTransformer _transformer;

        [BestFriend]
        internal TensorFlowEstimator(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, string modelLocation, bool addBatchDimensionInput)
            : this(env, outputColumnNames, inputColumnNames, TensorFlowUtils.LoadTensorFlowModel(env, modelLocation), addBatchDimensionInput)
        {
        }

        internal TensorFlowEstimator(IHostEnvironment env, string[] outputColumnNames, string[] inputColumnNames, TensorFlowModel tensorFlowModel, bool addBatchDimensionInput)
            : this(env, CreateArguments(tensorFlowModel, outputColumnNames, inputColumnNames, addBatchDimensionInput), tensorFlowModel)
        {
        }

        internal TensorFlowEstimator(IHostEnvironment env, Options options)
            : this(env, options, TensorFlowUtils.LoadTensorFlowModel(env, options.ModelLocation))
        {
        }

        internal TensorFlowEstimator(IHostEnvironment env, Options options, TensorFlowModel tensorFlowModel)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(TensorFlowEstimator));
            _options = options;
            _tensorFlowModel = tensorFlowModel;
            var inputTuple = TensorFlowTransformer.GetInputInfo(_host, tensorFlowModel.Session, options.InputColumns);
            _tfInputTypes = inputTuple.tfInputTypes;
            var outputTuple = TensorFlowTransformer.GetOutputInfo(_host, tensorFlowModel.Session, options.OutputColumns);
            _outputTypes = outputTuple.outputTypes;
        }

        private static Options CreateArguments(TensorFlowModel tensorFlowModel, string[] outputColumnNames, string[] inputColumnName, bool addBatchDimensionInput)
        {
            var options = new Options();
            options.ModelLocation = tensorFlowModel.ModelPath;
            options.InputColumns = inputColumnName;
            options.OutputColumns = outputColumnNames;
            options.ReTrain = false;
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
                var expectedType = TensorFlowUtils.Tf2MlNetType(_tfInputTypes[i]);
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
        /// Trains and returns a <see cref="TensorFlowTransformer"/>.
        /// </summary>
        public TensorFlowTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            if (_transformer == null)
            {
                _transformer = _options.ReTrain ? new TensorFlowTransformer(_host, _options, _tensorFlowModel, input) :
                    new TensorFlowTransformer(_host, _tensorFlowModel.Session, _options.OutputColumns, _options.InputColumns,
                    TensorFlowUtils.IsSavedModel(_host, _options.ModelLocation) ? _options.ModelLocation : null, false, _options.AddBatchDimensionInputs);
            }
            // Validate input schema.
            _transformer.GetOutputSchema(input.Schema);
            return _transformer;
        }
    }
}
