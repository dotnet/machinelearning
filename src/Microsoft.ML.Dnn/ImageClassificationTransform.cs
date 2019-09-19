// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Google.Protobuf;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Dnn;
using Tensorflow;
using Tensorflow.Summaries;
using static Microsoft.ML.Data.TextLoader;
using static Microsoft.ML.Transforms.Dnn.DnnUtils;
using static Microsoft.ML.Transforms.ImageClassificationEstimator;
using static Tensorflow.Binding;
using Architecture = Microsoft.ML.Transforms.ImageClassificationEstimator.Architecture;

[assembly: LoadableClass(ImageClassificationTransformer.Summary, typeof(IDataTransform), typeof(ImageClassificationTransformer),
    typeof(ImageClassificationEstimator.Options), typeof(SignatureDataTransform), ImageClassificationTransformer.UserName, ImageClassificationTransformer.ShortName)]

[assembly: LoadableClass(ImageClassificationTransformer.Summary, typeof(IDataTransform), typeof(ImageClassificationTransformer), null, typeof(SignatureLoadDataTransform),
    ImageClassificationTransformer.UserName, ImageClassificationTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageClassificationTransformer), null, typeof(SignatureLoadModel),
    ImageClassificationTransformer.UserName, ImageClassificationTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageClassificationTransformer), null, typeof(SignatureLoadRowMapper),
    ImageClassificationTransformer.UserName, ImageClassificationTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// <see cref="ITransformer" /> for the <see cref="ImageClassificationEstimator"/>.
    /// </summary>
    public sealed class ImageClassificationTransformer : RowToRowTransformerBase
    {
        private readonly IHostEnvironment _env;
        private readonly bool _addBatchDimensionInput;
        private Session _session;
        private Tensor _bottleneckTensor;
        private Operation _trainStep;
        private Tensor _softMaxTensor;
        private Tensor _crossEntropy;
        private Tensor _labelTensor;
        private Tensor _evaluationStep;
        private Tensor _prediction;
        private Tensor _bottleneckInput;
        private Tensor _jpegData;
        private Tensor _resizedImage;
        private string _jpegDataTensorName;
        private string _resizedImageTensorName;
        private string _inputTensorName;
        private readonly int _classCount;
        private readonly string _checkpointPath;
        private readonly string _bottleneckOperationName;
        private Graph Graph => _session.graph;
        private readonly string[] _inputs;
        private readonly string[] _outputs;
        private readonly string _labelColumnName;
        private readonly string _finalModelPrefix;
        private readonly Architecture _arch;
        private readonly string _scoreColumnName;
        private readonly string _predictedLabelColumnName;
        private readonly float _learningRate;
        private readonly string _softmaxTensorName;
        private readonly string _predictionTensorName;
        internal const string Summary = "Trains Dnn models.";
        internal const string UserName = "ImageClassificationTransform";
        internal const string ShortName = "ImgClsTrans";
        internal const string LoaderSignature = "ImageClassificationTrans";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGTRANS",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00000001,
                verReadableCur: 0x00000001,
                verWeCanReadBack: 0x00000001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImageClassificationTransformer).Assembly.FullName);
        }

        // Factory method for SignatureLoadModel.
        private static ImageClassificationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
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

            GetModelInfo(env, ctx, out string[] inputs, out string[] outputs, out bool addBatchDimensionInput,
                out string labelColumn, out string checkpointName, out Architecture arch, out string scoreColumnName,
                out string predictedColumnName, out float learningRate, out int classCount, out string predictionTensorName, out string softMaxTensorName,
                out string jpegDataTensorName, out string resizeTensorName);

            byte[] modelBytes = null;
            if (!ctx.TryLoadBinaryStream("TFModel", r => modelBytes = r.ReadByteArray()))
                throw env.ExceptDecode();

            return new ImageClassificationTransformer(env, DnnUtils.LoadTFSession(env, modelBytes), outputs, inputs,
                null, addBatchDimensionInput, 1, labelColumn, checkpointName, arch,
                scoreColumnName, predictedColumnName, learningRate, null, classCount, true, predictionTensorName,
                softMaxTensorName, jpegDataTensorName, resizeTensorName);

        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, ImageClassificationEstimator.Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.InputColumns, nameof(options.InputColumns));
            env.CheckValue(options.OutputColumns, nameof(options.OutputColumns));

            return new ImageClassificationTransformer(env, options, input).MakeDataTransform(input);
        }

        internal ImageClassificationTransformer(IHostEnvironment env, ImageClassificationEstimator.Options options, IDataView input)
            : this(env, options, DnnUtils.LoadDnnModel(env, options.ModelLocation), input)
        {
        }

        internal ImageClassificationTransformer(IHostEnvironment env, ImageClassificationEstimator.Options options, DnnModel tensorFlowModel, IDataView input)
            : this(env, tensorFlowModel.Session, options.OutputColumns, options.InputColumns,
                  options.ModelLocation, null, options.BatchSize,
                  options.LabelColumn, options.FinalModelPrefix, options.Arch, options.ScoreColumnName,
                  options.PredictedLabelColumnName, options.LearningRate, input.Schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            CheckTrainingParameters(options);
            var imageProcessor = new ImageProcessor(this);
            if (!options.ReuseTrainSetBottleneckCachedValues || !File.Exists(options.TrainSetBottleneckCachedValuesFilePath))
                CacheFeaturizedImagesToDisk(input, options.LabelColumn, options.InputColumns[0], imageProcessor,
                    _inputTensorName, _bottleneckTensor.name, options.TrainSetBottleneckCachedValuesFilePath,
                    ImageClassificationMetrics.Dataset.Train, options.MetricsCallback);

            if (options.ValidationSet != null &&
                    (!options.ReuseTrainSetBottleneckCachedValues || !File.Exists(options.ValidationSetBottleneckCachedValuesFilePath)))
                CacheFeaturizedImagesToDisk(options.ValidationSet, options.LabelColumn, options.InputColumns[0],
                    imageProcessor, _inputTensorName, _bottleneckTensor.name, options.ValidationSetBottleneckCachedValuesFilePath,
                    ImageClassificationMetrics.Dataset.Validation, options.MetricsCallback);

            TrainAndEvaluateClassificationLayer(options.TrainSetBottleneckCachedValuesFilePath, options, options.ValidationSetBottleneckCachedValuesFilePath);
        }

        private void CheckTrainingParameters(ImageClassificationEstimator.Options options)
        {
            Host.CheckNonWhiteSpace(options.LabelColumn, nameof(options.LabelColumn));
            Host.CheckNonWhiteSpace(options.TensorFlowLabel, nameof(options.TensorFlowLabel));

            if (_session.graph.OperationByName(_labelTensor.name.Split(':')[0]) == null)
                throw Host.ExceptParam(nameof(options.TensorFlowLabel), $"'{options.TensorFlowLabel}' does not exist in the model");
        }

        private (Tensor, Tensor) AddJpegDecoding(int height, int width, int depth)
        {
            // height, width, depth
            var inputDim = (height, width, depth);
            var jpegData = tf.placeholder(tf.@string, name: "DecodeJPGInput");
            var decodedImage = tf.image.decode_jpeg(jpegData, channels: inputDim.Item3);
            // Convert from full range of uint8 to range [0,1] of float32.
            var decodedImageAsFloat = tf.image.convert_image_dtype(decodedImage, tf.float32);
            var decodedImage4d = tf.expand_dims(decodedImageAsFloat, 0);
            var resizeShape = tf.stack(new int[] { inputDim.Item1, inputDim.Item2 });
            var resizeShapeAsInt = tf.cast(resizeShape, dtype: tf.int32);
            var resizedImage = tf.image.resize_bilinear(decodedImage4d, resizeShapeAsInt, false, "ResizeTensor");
            return (jpegData, resizedImage);
        }

        private sealed class ImageProcessor
        {
            private Runner _imagePreprocessingRunner;

            public ImageProcessor(ImageClassificationTransformer transformer)
            {
                _imagePreprocessingRunner = new Runner(transformer._session);
                _imagePreprocessingRunner.AddInput(transformer._jpegDataTensorName);
                _imagePreprocessingRunner.AddOutputs(transformer._resizedImageTensorName);
            }

            public Tensor ProcessImage(string path)
            {
                var imageTensor = new Tensor(File.ReadAllBytes(path), TF_DataType.TF_STRING);
                var processedTensor = _imagePreprocessingRunner.AddInput(imageTensor, 0).Run()[0];
                imageTensor.Dispose();
                return processedTensor;
            }
        }

        private void CacheFeaturizedImagesToDisk(IDataView input, string labelColumnName, string imagepathColumnName,
            ImageProcessor imageProcessor, string inputTensorName, string outputTensorName, string cacheFilePath,
            ImageClassificationMetrics.Dataset dataset, ImageClassificationMetricsCallback metricsCallback)
        {
            var labelColumn = input.Schema[labelColumnName];

            if (labelColumn.Type.RawType != typeof(UInt32))
                throw Host.ExceptSchemaMismatch(nameof(labelColumn), "Label",
                    labelColumnName, typeof(uint).ToString(),
                    labelColumn.Type.RawType.ToString());

            var imagePathColumn = input.Schema[imagepathColumnName];
            Runner runner = new Runner(_session);
            runner.AddOutputs(outputTensorName);

            using (TextWriter writer = File.CreateText(cacheFilePath))
            using (var cursor = input.GetRowCursor(input.Schema.Where(c => c.Index == labelColumn.Index || c.Index == imagePathColumn.Index)))
            {
                var labelGetter = cursor.GetGetter<uint>(labelColumn);
                var imagePathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(imagePathColumn);
                UInt32 label = UInt32.MaxValue;
                ReadOnlyMemory<char> imagePath = default;
                runner.AddInput(inputTensorName);
                ImageClassificationMetrics metrics = new ImageClassificationMetrics();
                metrics.Bottleneck = new BottleneckMetrics();
                metrics.Bottleneck.DatasetUsed = dataset;
                while (cursor.MoveNext())
                {
                    labelGetter(ref label);
                    imagePathGetter(ref imagePath);
                    var imagePathStr = imagePath.ToString();
                    var imageTensor = imageProcessor.ProcessImage(imagePathStr);
                    runner.AddInput(imageTensor, 0);
                    var featurizedImage = runner.Run()[0]; // Reuse memory?
                    writer.WriteLine(label - 1 + "," + string.Join(",", featurizedImage.ToArray<float>()));
                    featurizedImage.Dispose();
                    imageTensor.Dispose();
                    metrics.Bottleneck.Index++;
                    metrics.Bottleneck.Name = imagePathStr;
                    metricsCallback?.Invoke(metrics);
                }
            }
        }

        private IDataView GetShuffledData(string path)
        {
            return new RowShufflingTransformer(
                _env,
                new RowShufflingTransformer.Options
                {
                    ForceShuffle = true,
                    ForceShuffleSource = true
                },
                new TextLoader(
                    _env,
                    new TextLoader.Options
                    {
                        Separators = new[] { ',' },
                        Columns = new[]
                        {
                                        new Column("Label", DataKind.Int64, 0),
                                        new Column("Features", DataKind.Single, new [] { new Range(1, null) }),
                        },
                    },
                    new MultiFileSource(path))
                    .Load(new MultiFileSource(path)));
        }

        private void TrainAndEvaluateClassificationLayer(string trainBottleneckFilePath, ImageClassificationEstimator.Options options,
            string validationSetBottleneckFilePath)
        {
            int batchSize = options.BatchSize;
            int epochs = options.Epoch;
            bool evaluateOnly = !string.IsNullOrEmpty(validationSetBottleneckFilePath);
            ImageClassificationMetricsCallback statisticsCallback = options.MetricsCallback;
            var trainingSet = GetShuffledData(trainBottleneckFilePath);
            IDataView validationSet = null;
            if (options.ValidationSet != null && !string.IsNullOrEmpty(validationSetBottleneckFilePath))
                validationSet = GetShuffledData(validationSetBottleneckFilePath);

            long label = long.MaxValue;
            VBuffer<float> features = default;
            ReadOnlySpan<float> featureValues = default;
            var featureColumn = trainingSet.Schema[1];
            int featureLength = featureColumn.Type.GetVectorSize();
            float[] featureBatch = new float[featureLength * batchSize];
            var featureBatchHandle = GCHandle.Alloc(featureBatch, GCHandleType.Pinned);
            IntPtr featureBatchPtr = featureBatchHandle.AddrOfPinnedObject();
            int featureBatchSizeInBytes = sizeof(float) * featureBatch.Length;
            long[] labelBatch = new long[batchSize];
            var labelBatchHandle = GCHandle.Alloc(labelBatch, GCHandleType.Pinned);
            IntPtr labelBatchPtr = labelBatchHandle.AddrOfPinnedObject();
            int labelBatchSizeInBytes = sizeof(long) * labelBatch.Length;
            var labelTensorShape = _labelTensor.TensorShape.dims.Select(x => (long)x).ToArray();
            labelTensorShape[0] = batchSize;
            int batchIndex = 0;
            var runner = new Runner(_session);
            var testEvalRunner = new Runner(_session);
            testEvalRunner.AddOutputs(_evaluationStep.name);
            testEvalRunner.AddOutputs(_crossEntropy.name);

            Runner validationEvalRunner = null;
            if (validationSet != null)
            {
                validationEvalRunner = new Runner(_session);
                validationEvalRunner.AddOutputs(_evaluationStep.name);
                validationEvalRunner.AddInput(_bottleneckInput.name).AddInput(_labelTensor.name);
            }

            runner.AddOperation(_trainStep);
            var featureTensorShape = _bottleneckInput.TensorShape.dims.Select(x => (long)x).ToArray();
            featureTensorShape[0] = batchSize;

            Saver trainSaver = null;
            FileWriter trainWriter = null;
            Tensor merged = tf.summary.merge_all();
            trainWriter = tf.summary.FileWriter(Path.Combine(Directory.GetCurrentDirectory(), "train"), _session.graph);
            trainSaver = tf.train.Saver();
            trainSaver.save(_session, _checkpointPath);

            runner.AddInput(_bottleneckInput.name).AddInput(_labelTensor.name);
            testEvalRunner.AddInput(_bottleneckInput.name).AddInput(_labelTensor.name);
            Dictionary<long, int> classStatsTrain = new Dictionary<long, int>();
            Dictionary<long, int> classStatsValidate = new Dictionary<long, int>();
            for (int index = 0; index < _classCount; index += 1)
                classStatsTrain[index] = classStatsValidate[index] = 0;

            ImageClassificationMetrics metrics = new ImageClassificationMetrics();
            metrics.Train = new TrainMetrics();
            for (int epoch = 0; epoch < epochs; epoch += 1)
            {
                metrics.Train.Accuracy = 0;
                metrics.Train.CrossEntropy = 0;
                metrics.Train.BatchProcessedCount = 0;
                using (var cursor = trainingSet.GetRowCursor(trainingSet.Schema.ToArray(), new Random()))
                {
                    var labelGetter = cursor.GetGetter<long>(trainingSet.Schema[0]);
                    var featuresGetter = cursor.GetGetter<VBuffer<float>>(featureColumn);
                    while (cursor.MoveNext())
                    {
                        labelGetter(ref label);
                        featuresGetter(ref features);
                        classStatsTrain[label]++;

                        if (featureValues == default)
                            featureValues = features.GetValues();

                        // Buffer the values.
                        for (int index = 0; index < featureLength; index += 1)
                            featureBatch[batchIndex * featureLength + index] = featureValues[index];

                        labelBatch[batchIndex] = label;
                        batchIndex += 1;
                        // Train.
                        if (batchIndex == batchSize)
                        {
                            runner.AddInput(new Tensor(featureBatchPtr, featureTensorShape, TF_DataType.TF_FLOAT, featureBatchSizeInBytes), 0)
                                .AddInput(new Tensor(labelBatchPtr, labelTensorShape, TF_DataType.TF_INT64, labelBatchSizeInBytes), 1)
                                .Run();

                            metrics.Train.BatchProcessedCount += 1;

                            if (options.TestOnTrainSet && statisticsCallback != null)
                            {
                                var outputTensors = testEvalRunner
                                    .AddInput(new Tensor(featureBatchPtr, featureTensorShape, TF_DataType.TF_FLOAT, featureBatchSizeInBytes), 0)
                                    .AddInput(new Tensor(labelBatchPtr, labelTensorShape, TF_DataType.TF_INT64, labelBatchSizeInBytes), 1)
                                    .Run();

                                metrics.Train.Accuracy += outputTensors[0].ToArray<float>()[0];
                                metrics.Train.CrossEntropy += outputTensors[1].ToArray<float>()[0];

                                outputTensors[0].Dispose();
                                outputTensors[1].Dispose();
                            }

                            batchIndex = 0;
                        }
                    }

                    if (options.TestOnTrainSet && statisticsCallback != null)
                    {
                        metrics.Train.Epoch = epoch;
                        metrics.Train.Accuracy /= metrics.Train.BatchProcessedCount;
                        metrics.Train.CrossEntropy /= metrics.Train.BatchProcessedCount;
                        metrics.Train.DatasetUsed = ImageClassificationMetrics.Dataset.Train;
                        statisticsCallback(metrics);
                    }
                }

                if (validationSet == null)
                    continue;

                batchIndex = 0;
                metrics.Train.BatchProcessedCount = 0;
                metrics.Train.Accuracy = 0;
                metrics.Train.CrossEntropy = 0;
                using (var cursor = validationSet.GetRowCursor(validationSet.Schema.ToArray(), new Random()))
                {
                    var labelGetter = cursor.GetGetter<long>(validationSet.Schema[0]);
                    var featuresGetter = cursor.GetGetter<VBuffer<float>>(featureColumn);
                    while (cursor.MoveNext())
                    {
                        labelGetter(ref label);
                        featuresGetter(ref features);
                        classStatsValidate[label]++;
                        // Buffer the values.
                        for (int index = 0; index < featureLength; index += 1)
                            featureBatch[batchIndex * featureLength + index] = featureValues[index];

                        labelBatch[batchIndex] = label;
                        batchIndex += 1;
                        // Evaluate.
                        if (batchIndex == batchSize)
                        {
                            var outputTensors = validationEvalRunner
                                .AddInput(new Tensor(featureBatchPtr, featureTensorShape, TF_DataType.TF_FLOAT, featureBatchSizeInBytes), 0)
                                .AddInput(new Tensor(labelBatchPtr, labelTensorShape, TF_DataType.TF_INT64, labelBatchSizeInBytes), 1)
                                .Run();

                            metrics.Train.Accuracy += outputTensors[0].ToArray<float>()[0];
                            metrics.Train.BatchProcessedCount += 1;
                            batchIndex = 0;

                            outputTensors[0].Dispose();
                        }
                    }

                    if (statisticsCallback != null)
                    {
                        metrics.Train.Epoch = epoch;
                        metrics.Train.Accuracy /= metrics.Train.BatchProcessedCount;
                        metrics.Train.DatasetUsed = ImageClassificationMetrics.Dataset.Validation;
                        statisticsCallback(metrics);
                    }
                }
            }

            trainSaver.save(_session, _checkpointPath);
            UpdateTransferLearningModelOnDisk(options, _classCount);
        }

        private (Session, Tensor, Tensor, Tensor) BuildEvaluationSession(ImageClassificationEstimator.Options options, int classCount)
        {
            var evalGraph = DnnUtils.LoadMetaGraph(options.ModelLocation);
            var evalSess = tf.Session(graph: evalGraph);
            Tensor evaluationStep = null;
            Tensor prediction = null;
            Tensor bottleneckTensor = evalGraph.OperationByName(_bottleneckOperationName);
            evalGraph.as_default();
            var (_, _, groundTruthInput, finalTensor) = AddFinalRetrainOps(classCount, options.LabelColumn,
                    options.ScoreColumnName, options.LearningRate, bottleneckTensor, false);

                tf.train.Saver().restore(evalSess, _checkpointPath);
                (evaluationStep, prediction) = AddEvaluationStep(finalTensor, groundTruthInput);
                (_jpegData, _resizedImage) = AddJpegDecoding(299, 299, 3);
            return (evalSess, _labelTensor, evaluationStep, prediction);
        }

        private (Tensor, Tensor) AddEvaluationStep(Tensor resultTensor, Tensor groundTruthTensor)
        {
            Tensor evaluationStep = null;
            Tensor correctPrediction = null;

            tf_with(tf.name_scope("accuracy"), scope =>
            {
                tf_with(tf.name_scope("correct_prediction"), delegate
                {
                    _prediction = tf.argmax(resultTensor, 1);
                    correctPrediction = tf.equal(_prediction, groundTruthTensor);
                });

                tf_with(tf.name_scope("accuracy"), delegate
                {
                    evaluationStep = tf.reduce_mean(tf.cast(correctPrediction, tf.float32));
                });
            });

            tf.summary.scalar("accuracy", evaluationStep);
            return (evaluationStep, _prediction);
        }

        private void UpdateTransferLearningModelOnDisk(ImageClassificationEstimator.Options options, int classCount)
        {
            var (sess, _, _, _) = BuildEvaluationSession(options, classCount);
            var graph = sess.graph;
            var outputGraphDef = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), new string[] { _softMaxTensor.name.Split(':')[0], _prediction.name.Split(':')[0], _jpegData.name.Split(':')[0], _resizedImage.name.Split(':')[0] });

            string frozenModelPath = _checkpointPath + ".pb";
            File.WriteAllBytes(_checkpointPath + ".pb", outputGraphDef.ToByteArray());
            _session.graph.Dispose();
            _session.Dispose();
            _session = LoadTFSessionByModelFilePath(_env, frozenModelPath, false);
        }

        private void VariableSummaries(RefVariable var)
        {
            tf_with(tf.name_scope("summaries"), delegate
            {
                var mean = tf.reduce_mean(var);
                tf.summary.scalar("mean", mean);
                Tensor stddev = null;
                tf_with(tf.name_scope("stddev"), delegate
                {
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)));
                });
                tf.summary.scalar("stddev", stddev);
                tf.summary.scalar("max", tf.reduce_max(var));
                tf.summary.scalar("min", tf.reduce_min(var));
                tf.summary.histogram("histogram", var);
            });
        }

        private (Operation, Tensor, Tensor, Tensor) AddFinalRetrainOps(int classCount, string labelColumn,
            string scoreColumnName, float learningRate, Tensor bottleneckTensor, bool isTraining)
        {
            var bottleneckTensorDims = bottleneckTensor.TensorShape.dims;
            var (batch_size, bottleneck_tensor_size) = (bottleneckTensorDims[0], bottleneckTensorDims[1]);
            tf_with(tf.name_scope("input"), scope =>
            {
                if (isTraining)
                {
                    _bottleneckInput = tf.placeholder_with_default(
                        bottleneckTensor,
                        shape: bottleneckTensorDims ,
                        name: "BottleneckInputPlaceholder");
                }

                _labelTensor = tf.placeholder(tf.int64, new TensorShape(batch_size), name: labelColumn);
            });

            string layerName = "final_retrain_ops";
            Tensor logits = null;
            tf_with(tf.name_scope(layerName), scope =>
            {
                RefVariable layerWeights = null;
                tf_with(tf.name_scope("weights"), delegate
                {
                    var initialValue = tf.truncated_normal(new int[] { bottleneck_tensor_size, classCount }, stddev: 0.001f);
                    layerWeights = tf.Variable(initialValue, name: "final_weights");
                    VariableSummaries(layerWeights);
                });

                RefVariable layerBiases = null;
                tf_with(tf.name_scope("biases"), delegate
                {
                    TensorShape shape = new TensorShape(classCount);
                    layerBiases = tf.Variable(tf.zeros(shape), name: "final_biases");
                    VariableSummaries(layerBiases);
                });

                tf_with(tf.name_scope("Wx_plus_b"), delegate
                {
                    var matmul = tf.matmul(isTraining ? _bottleneckInput : bottleneckTensor, layerWeights);
                    logits = matmul + layerBiases;
                    tf.summary.histogram("pre_activations", logits);
                });
            });

            _softMaxTensor = tf.nn.softmax(logits, name: scoreColumnName);

            tf.summary.histogram("activations", _softMaxTensor);
            if (!isTraining)
                return (null, null, _labelTensor, _softMaxTensor);

            Tensor crossEntropyMean = null;
            tf_with(tf.name_scope("cross_entropy"), delegate
            {
                crossEntropyMean = tf.losses.sparse_softmax_cross_entropy(
                    labels: _labelTensor, logits: logits);
            });

            tf.summary.scalar("cross_entropy", crossEntropyMean);

            tf_with(tf.name_scope("train"), delegate
            {
                var optimizer = tf.train.GradientDescentOptimizer(learningRate);
                _trainStep = optimizer.minimize(crossEntropyMean);
            });

            return (_trainStep, crossEntropyMean, _labelTensor, _softMaxTensor);
        }

        private void AddTransferLearningLayer(string labelColumn,
            string scoreColumnName, float learningRate, int classCount)
        {
            _bottleneckTensor = Graph.OperationByName(_bottleneckOperationName);
            (_trainStep, _crossEntropy, _labelTensor, _softMaxTensor) =
                    AddFinalRetrainOps(classCount, labelColumn, scoreColumnName, learningRate, _bottleneckTensor, true);

        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static void GetModelInfo(IHostEnvironment env, ModelLoadContext ctx, out string[] inputs,
            out string[] outputs, out bool addBatchDimensionInput,
            out string labelColumn, out string checkpointName, out Architecture arch,
            out string scoreColumnName, out string predictedColumnName, out float learningRate, out int classCount, out string predictionTensorName, out string softMaxTensorName,
            out string jpegDataTensorName, out string resizeTensorName)
        {
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

            labelColumn = ctx.Reader.ReadString();
            checkpointName = ctx.Reader.ReadString();
            arch = (Architecture)ctx.Reader.ReadInt32();
            scoreColumnName = ctx.Reader.ReadString();
            predictedColumnName = ctx.Reader.ReadString();
            learningRate = ctx.Reader.ReadFloat();
            classCount = ctx.Reader.ReadInt32();
            predictionTensorName = ctx.Reader.ReadString();
            softMaxTensorName = ctx.Reader.ReadString();
            jpegDataTensorName = ctx.Reader.ReadString();
            resizeTensorName = ctx.Reader.ReadString();
        }

        internal ImageClassificationTransformer(IHostEnvironment env, Session session, string[] outputColumnNames,
            string[] inputColumnNames, string modelLocation,
            bool? addBatchDimensionInput, int batchSize, string labelColumnName, string finalModelPrefix, Architecture arch,
            string scoreColumnName, string predictedLabelColumnName, float learningRate, DataViewSchema inputSchema, int? classCount = null, bool loadModel = false,
            string predictionTensorName = null, string softMaxTensorName = null, string jpegDataTensorName = null, string resizeTensorName = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageClassificationTransformer)))

        {
            Host.CheckValue(session, nameof(session));
            Host.CheckNonEmpty(inputColumnNames, nameof(inputColumnNames));
            Host.CheckNonEmpty(outputColumnNames, nameof(outputColumnNames));

            _env = env;
            _session = session;
            _addBatchDimensionInput = addBatchDimensionInput ?? arch == Architecture.ResnetV2101;
            _inputs = inputColumnNames;
            _outputs = outputColumnNames;
            _labelColumnName = labelColumnName;
            _finalModelPrefix = finalModelPrefix;
            _arch = arch;
            _scoreColumnName = scoreColumnName;
            _predictedLabelColumnName = predictedLabelColumnName;
            _learningRate = learningRate;
            _softmaxTensorName = softMaxTensorName;
            _predictionTensorName = predictionTensorName;
            _jpegDataTensorName = jpegDataTensorName;
            _resizedImageTensorName = resizeTensorName;

            if (classCount == null)
            {
                var labelColumn = inputSchema.GetColumnOrNull(labelColumnName).Value;
                var labelType = labelColumn.Type;
                var labelCount = labelType.GetKeyCount();
                if (labelCount <= 0)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", (string)labelColumn.Name, "Key", (string)labelType.ToString());

                _classCount = labelCount == 1 ? 2 : (int)labelCount;
            }
            else
                _classCount = classCount.Value;

            _checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), finalModelPrefix + modelLocation);

            // Configure bottleneck tensor based on the model.
            if (arch == ImageClassificationEstimator.Architecture.ResnetV2101)
            {
                _bottleneckOperationName = "resnet_v2_101/SpatialSqueeze";
                _inputTensorName = "input";
            }
            else if (arch == ImageClassificationEstimator.Architecture.InceptionV3)
            {
                _bottleneckOperationName = "module_apply_default/hub_output/feature_vector/SpatialSqueeze";
                _inputTensorName = "Placeholder";
            }

            _outputs = new[] { scoreColumnName, predictedLabelColumnName };

            if (loadModel == false)
            {
                (_jpegData, _resizedImage) = AddJpegDecoding(299, 299, 3);
                _jpegDataTensorName = _jpegData.name;
                _resizedImageTensorName = _resizedImage.name;

                // Add transfer learning layer.
                AddTransferLearningLayer(labelColumnName, scoreColumnName, learningRate, _classCount);

                // Initialize the variables.
                new Runner(_session).AddOperation(tf.global_variables_initializer()).Run();

                // Add evaluation layer.
                (_evaluationStep, _) = AddEvaluationStep(_softMaxTensor, _labelTensor);
                _softmaxTensorName = _softMaxTensor.name;
                _predictionTensorName = _prediction.name;
            }
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            ctx.Writer.WriteBoolByte(_addBatchDimensionInput);

            Host.AssertNonEmpty(_inputs);
            ctx.Writer.Write(_inputs.Length);
            foreach (var colName in _inputs)
                ctx.SaveNonEmptyString(colName);

            Host.AssertNonEmpty(_outputs);
            ctx.Writer.Write(_outputs.Length);
            foreach (var colName in _outputs)
                ctx.SaveNonEmptyString(colName);

            ctx.Writer.Write(_labelColumnName);
            ctx.Writer.Write(_finalModelPrefix);
            ctx.Writer.Write((int)_arch);
            ctx.Writer.Write(_scoreColumnName);
            ctx.Writer.Write(_predictedLabelColumnName);
            ctx.Writer.Write(_learningRate);
            ctx.Writer.Write(_classCount);
            ctx.Writer.Write(_predictionTensorName);
            ctx.Writer.Write(_softmaxTensorName);
            ctx.Writer.Write(_jpegDataTensorName);
            ctx.Writer.Write(_resizedImageTensorName);
            Status status = new Status();
            var buffer = _session.graph.ToGraphDef(status);
            ctx.SaveBinaryStream("TFModel", w =>
            {
                w.WriteByteArray(buffer.MemoryBlock.ToArray());
            });
            status.Check(true);
        }

        ~ImageClassificationTransformer()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            // Ensure that the Session is not null and it's handle is not Zero, as it may have already been disposed/finalized.
            // Technically we shouldn't be calling this if disposing == false, since we're running in finalizer
            // and the GC doesn't guarantee ordering of finalization of managed objects, but we have to make sure
            // that the Session is closed before deleting our temporary directory.
            if (_session != null && _session != IntPtr.Zero)
            {
                _session.close();
            }
        }

        private sealed class Mapper : MapperBase
        {
            private readonly ImageClassificationTransformer _parent;
            private readonly int[] _inputColIndices;

            public Mapper(ImageClassificationTransformer parent, DataViewSchema inputSchema) :
                   base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                Host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _inputColIndices = new int[1];
                if (!inputSchema.TryGetColumnIndex(_parent._inputs[0], out _inputColIndices[0]))
                    throw Host.ExceptSchemaMismatch(nameof(InputSchema), "source", _parent._inputs[0]);
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            private class OutputCache
            {
                public long Position;
                private ValueGetter<ReadOnlyMemory<char>> _imagePathGetter;
                private ReadOnlyMemory<char> _imagePath;
                private Runner _runner;
                private ImageProcessor _imageProcessor;
                public UInt32 PredictedLabel { get; set; }
                public float[] ClassProbabilities { get; set; }
                private DataViewRow _inputRow;

                public OutputCache(DataViewRow input, ImageClassificationTransformer transformer)
                {
                    _imagePath = default;
                    _imagePathGetter = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[transformer._inputs[0]]);
                    _runner = new Runner(transformer._session);
                    _runner.AddInput(transformer._inputTensorName);
                    _runner.AddOutputs(transformer._softmaxTensorName);
                    _runner.AddOutputs(transformer._predictionTensorName);
                    _imageProcessor = new ImageProcessor(transformer);
                    _inputRow = input;
                    Position = -1;
                }

                public void UpdateCacheIfNeeded()
                {
                    lock (this)
                    {
                        if (_inputRow.Position != Position)
                        {
                            Position = _inputRow.Position;
                            _imagePathGetter(ref _imagePath);
                            var processedTensor = _imageProcessor.ProcessImage(_imagePath.ToString());
                            var outputTensor = _runner.AddInput(processedTensor, 0).Run();
                            ClassProbabilities = outputTensor[0].ToArray<float>();
                            PredictedLabel = (UInt32)outputTensor[1].ToArray<long>()[0];
                            outputTensor[0].Dispose();
                            outputTensor[1].Dispose();
                            processedTensor.Dispose();
                        }
                    }
                }
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                _parent._session.graph.as_default();
                Host.AssertValue(input);
                var cache = new OutputCache(input, _parent);

                if (iinfo == 0)
                {
                    ValueGetter<VBuffer<float>> valuegetter = (ref VBuffer<float> dst) =>
                    {
                        cache.UpdateCacheIfNeeded();
                        var editor = VBufferEditor.Create(ref dst, cache.ClassProbabilities.Length);
                        new Span<float>(cache.ClassProbabilities, 0, cache.ClassProbabilities.Length).CopyTo(editor.Values);
                        dst = editor.Commit();
                    };
                    return valuegetter;
                }
                else
                {
                    ValueGetter<UInt32> valuegetter = (ref UInt32 dst) =>
                    {
                        cache.UpdateCacheIfNeeded();
                        dst = cache.PredictedLabel;
                    };

                    return valuegetter;
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent._outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[_parent._outputs.Length];
                info[0] = new DataViewSchema.DetachedColumn(_parent._outputs[0], new VectorDataViewType(NumberDataViewType.Single, _parent._classCount), null);
                info[1] = new DataViewSchema.DetachedColumn(_parent._outputs[1], NumberDataViewType.UInt32, null);
                return info;
            }
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="ImageClassificationTransformer"]/*' />
    public sealed class ImageClassificationEstimator : IEstimator<ImageClassificationTransformer>
    {
        /// <summary>
        /// Image classification model.
        /// </summary>
        public enum Architecture
        {
            ResnetV2101,
            InceptionV3
        };

        /// <summary>
        /// Backend DNN training framework.
        /// </summary>
        public enum DnnFramework
        {
            Tensorflow
        };

        /// <summary>
        /// Callback that returns DNN statistics during training phase.
        /// </summary>
        public delegate void ImageClassificationMetricsCallback(ImageClassificationMetrics metrics);

        /// <summary>
        /// DNN training metrics.
        /// </summary>
        public sealed class TrainMetrics
        {
            /// <summary>
            /// Indicates the dataset on which metrics are being reported.
            /// <see cref="ImageClassificationMetrics.Dataset"/>
            /// </summary>
            public ImageClassificationMetrics.Dataset DatasetUsed { get; set; }

            /// <summary>
            /// The number of batches processed in an epoch.
            /// </summary>
            public int BatchProcessedCount { get; set; }

            /// <summary>
            /// The training epoch index for which this metric is reported.
            /// </summary>
            public int Epoch { get; set; }

            /// <summary>
            /// Accuracy of the batch on this <see cref="Epoch"/>. Higher the better.
            /// </summary>
            public float Accuracy { get; set; }

            /// <summary>
            /// Cross-Entropy (loss) of the batch on this <see cref="Epoch"/>. Lower
            /// the better.
            /// </summary>
            public float CrossEntropy { get; set; }

            /// <summary>
            /// String representation of the metrics.
            /// </summary>
            public override string ToString()
            {
                if (DatasetUsed == ImageClassificationMetrics.Dataset.Train)
                    return $"Phase: Training, Dataset used: {DatasetUsed.ToString(),10}, Batch Processed Count: {BatchProcessedCount,3}, " +
                        $"Epoch: {Epoch,3}, Accuracy: {Accuracy,10}, Cross-Entropy: {CrossEntropy,10}";
                else
                    return $"Phase: Training, Dataset used: {DatasetUsed.ToString(),10}, Batch Processed Count: {BatchProcessedCount,3}, " +
                        $"Epoch: {Epoch,3}, Accuracy: {Accuracy,10}";
            }
        }

        /// <summary>
        /// Metrics for image featurization values. The input image is passed through
        /// the network and features are extracted from second or last layer to
        /// train a custom full connected layer that serves as classifier.
        /// </summary>
        public sealed class BottleneckMetrics
        {
            /// <summary>
            /// Indicates the dataset on which metrics are being reported.
            /// <see cref="ImageClassificationMetrics.Dataset"/>
            /// </summary>
            public ImageClassificationMetrics.Dataset DatasetUsed { get; set; }

            /// <summary>
            /// Name of the input image.
            /// </summary>
            public string Name { get; set; }

            /// <summary>
            /// Index of the input image.
            /// </summary>
            public int Index { get; set; }

            /// <summary>
            /// String representation of the metrics.
            /// </summary>
            public override string ToString() => $"Phase: Bottleneck Computation, Dataset used: {DatasetUsed.ToString(),10}, Image Index: {Index,3}, Image Name: {Name}";
        }

        /// <summary>
        /// Metrics for image classification training.
        /// </summary>
        public sealed class ImageClassificationMetrics
        {
            /// <summary>
            /// Indicates the kind of the dataset of which metric is reported.
            /// </summary>
            public enum Dataset
            {
                Train,
                Validation
            };

            /// <summary>
            /// Contains train time metrics.
            /// </summary>
            public TrainMetrics Train { get; set; }

            /// <summary>
            /// Contains pre-train time metrics. These contains metrics on image
            /// featurization.
            /// </summary>
            public BottleneckMetrics Bottleneck { get; set; }

            /// <summary>
            /// String representation of the metrics.
            /// </summary>
            public override string ToString() => Train != null ? Train.ToString() : Bottleneck.ToString();
        }

        /// <summary>
        /// The options for the <see cref="ImageClassificationTransformer"/>.
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
            /// Learning rate to use during optimization.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate to use during optimization.", SortOrder = 12)]
            public float LearningRate = 0.01f;

            /// <summary>
            /// Specifies the model architecture to be used in the case of image classification training using transfer learning.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Model architecture to be used in transfer learning for image classification.", SortOrder = 15)]
            public Architecture Arch = Architecture.InceptionV3;

            /// <summary>
            /// Name of the tensor that will contain the output scores of the last layer when transfer learning is done.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Softmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string ScoreColumnName = "Scores";

            /// <summary>
            /// Name of the tensor that will contain the predicted label from output scores of the last layer when transfer learning is done.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Argmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string PredictedLabelColumnName = "PredictedLabel";

            /// <summary>
            /// Final model and checkpoint files/folder prefix for storing graph files.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Final model and checkpoint files/folder prefix for storing graph files.", SortOrder = 15)]
            public string FinalModelPrefix = "custom_retrained_model_based_on_";

            /// <summary>
            /// Callback to report statistics on accuracy/cross entropy during training phase.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Callback to report metrics during training and validation phase.", SortOrder = 15)]
            public ImageClassificationMetricsCallback MetricsCallback = null;

            /// <summary>
            /// Frequency of epochs at which statistics on training phase should be reported.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Frequency of epochs at which statistics on training/validation phase should be reported.", SortOrder = 15)]
            public int StatisticsFrequency = 1;

            /// <summary>
            /// Indicates the choice DNN training framework. Currently only TensorFlow is supported.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the choice DNN training framework. Currently only TensorFlow is supported.", SortOrder = 15)]
            public DnnFramework Framework = DnnFramework.Tensorflow;

            /// <summary>
            /// Indicates the path where the newly retrained model should be saved.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the path where the newly retrained model should be saved.", SortOrder = 15)]
            public string ModelSavePath = null;

            /// <summary>
            /// Indicates to evaluate the model on train set after every epoch.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates to evaluate the model on train set after every epoch.", SortOrder = 15)]
            public bool TestOnTrainSet;

            /// <summary>
            /// Indicates to not re-compute cached bottleneck trainset values if already available in the bin folder.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates to not re-compute trained cached bottleneck values if already available in the bin folder.", SortOrder = 15)]
            public bool ReuseTrainSetBottleneckCachedValues;

            /// <summary>
            /// Indicates to not re-compute cached bottleneck validationset values if already available in the bin folder.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates to not re-compute validataionset cached bottleneck validationset values if already available in the bin folder.", SortOrder = 15)]
            public bool ReuseValidationSetBottleneckCachedValues;

            /// <summary>
            /// Validation set.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Validation set.", SortOrder = 15)]
            public IDataView ValidationSet;

            /// <summary>
            /// Indicates the file path to store trainset bottleneck values for caching.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the file path to store trainset bottleneck values for caching.", SortOrder = 15)]
            public string TrainSetBottleneckCachedValuesFilePath;

            /// <summary>
            /// Indicates the file path to store validationset bottleneck values for caching.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the file path to store validationset bottleneck values for caching.", SortOrder = 15)]
            public string ValidationSetBottleneckCachedValuesFilePath;
        }

        private readonly IHost _host;
        private readonly Options _options;
        private readonly DnnModel _dnnModel;
        private readonly TF_DataType[] _tfInputTypes;
        private readonly DataViewType[] _outputTypes;
        private ImageClassificationTransformer _transformer;

        internal ImageClassificationEstimator(IHostEnvironment env, Options options, DnnModel dnnModel)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageClassificationEstimator));
            _options = options;
            _dnnModel = dnnModel;
            _tfInputTypes = new[] { TF_DataType.TF_STRING };
            _outputTypes = new[] { new VectorDataViewType(NumberDataViewType.Single), NumberDataViewType.UInt32.GetItemType() };
        }

        private static Options CreateArguments(DnnModel tensorFlowModel, string[] outputColumnNames, string[] inputColumnName, bool addBatchDimensionInput)
        {
            var options = new Options();
            options.ModelLocation = tensorFlowModel.ModelPath;
            options.InputColumns = inputColumnName;
            options.OutputColumns = outputColumnNames;
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
                var expectedType = DnnUtils.Tf2MlNetType(_tfInputTypes[i]);
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
        /// Trains and returns a <see cref="ImageClassificationTransformer"/>.
        /// </summary>
        public ImageClassificationTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            if (_transformer == null)
                _transformer = new ImageClassificationTransformer(_host, _options, _dnnModel, input);

            // Validate input schema.
            _transformer.GetOutputSchema(input.Schema);
            return _transformer;
        }
    }
}
