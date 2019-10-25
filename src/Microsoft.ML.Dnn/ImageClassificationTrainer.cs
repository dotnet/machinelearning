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
using Microsoft.ML.Dnn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Tensorflow;
using Tensorflow.Summaries;
using static Microsoft.ML.Data.TextLoader;
using static Microsoft.ML.Dnn.DnnUtils;
using static Tensorflow.Binding;
using Column = Microsoft.ML.Data.TextLoader.Column;

[assembly: LoadableClass(ImageClassificationTrainer.Summary, typeof(ImageClassificationTrainer),
    typeof(ImageClassificationTrainer.Options),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer) },
    ImageClassificationTrainer.UserName,
    ImageClassificationTrainer.LoadName,
    ImageClassificationTrainer.ShortName)]

[assembly: LoadableClass(typeof(ImageClassificationModelParameters), null, typeof(SignatureLoadModel),
    "Image classification predictor", ImageClassificationModelParameters.LoaderSignature)]

namespace Microsoft.ML.Dnn
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a Deep Neural Network(DNN) to classify images.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [ImageClassification](xref:Microsoft.ML.Dnn.DnnCatalog.ImageClassification(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String,System.String,System.String,Microsoft.ML.IDataView)).
    ///
    /// ### Input and Output Columns
    /// The input label column data must be[key] (xref:Microsoft.ML.Data.KeyDataViewType) type and the feature column must be a variable-sized vector of<xref:System.Byte>.
    ///
    /// This trainer outputs the following columns:
    ///
    /// | Output Column Name | Column Type | Description|
    /// | -- | -- | -- |
    /// | `Score` | Vector of<xref:System.Single> | The scores of all classes.Higher value means higher probability to fall into the associated class. If the i-th element has the largest value, the predicted label index would be i.Note that i is zero-based index. |
    /// | `PredictedLabel` | [key](xref:Microsoft.ML.Data.KeyDataViewType) type | The predicted label's index. If its value is i, the actual label would be the i-th category in the key-valued input label type. |
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | No |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | Micrsoft.ML.Dnn and SciSharp.TensorFlow.Redist / SciSharp.TensorFlow.Redist-Windows-GPU / SciSharp.TensorFlow.Redist-Linux-GPU |
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/tensorflow-usage.md)]
    ///
    /// ### Training Algorithm Details
    /// Trains a Deep Neural Network(DNN) by leveraging an existing pre-trained model such as Resnet50 for the purpose
    /// of classifying images.
    /// ]]>
    /// </format>
    /// </remarks>
    public sealed class ImageClassificationTrainer :
        TrainerEstimatorBase<MulticlassPredictionTransformer<ImageClassificationModelParameters>,
            ImageClassificationModelParameters>
    {
        internal const string LoadName = "ImageClassificationTrainer";
        internal const string UserName = "Image Classification Trainer";
        internal const string ShortName = "IMGCLSS";
        internal const string Summary = "Trains a DNN model to classify images.";

        /// <summary>
        /// Image classification model.
        /// </summary>
        public enum Architecture
        {
            ResnetV2101,
            InceptionV3,
            MobilenetV2,
            ResnetV250
        };

        /// <summary>
        /// Dictionary mapping model architecture to model location.
        /// </summary>
        internal static IReadOnlyDictionary<Architecture, string> ModelLocation = new Dictionary<Architecture, string>
        {
            { Architecture.ResnetV2101, @"resnet_v2_101_299.meta" },
            { Architecture.InceptionV3, @"InceptionV3.meta" },
            { Architecture.MobilenetV2, @"mobilenet_v2.meta" },
            { Architecture.ResnetV250, @"resnet_v2_50_299.meta" }
        };

        /// <summary>
        /// Dictionary mapping model architecture to image input size supported.
        /// </summary>
        internal static IReadOnlyDictionary<Architecture, Tuple<int, int>> ImagePreprocessingSize =
            new Dictionary<Architecture, Tuple<int, int>>
        {
            { Architecture.ResnetV2101, new Tuple<int, int>(299,299) },
            { Architecture.InceptionV3, new Tuple<int, int>(299,299) },
            { Architecture.MobilenetV2, new Tuple<int, int>(224,224) },
            { Architecture.ResnetV250, new Tuple<int, int>(299,299) }
        };

        /// <summary>
        /// Indicates the metric to be monitored to decide Early Stopping criteria.
        /// </summary>
        public enum EarlyStoppingMetric
        {
            Accuracy,
            Loss
        }

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
            /// Learning Rate used for this <see cref="Epoch"/>. Changes for learning rate scheduling.
            /// </summary>
            public float LearningRate { get; set; }

            /// <summary>
            /// String representation of the metrics.
            /// </summary>
            public override string ToString()
            {
                if (DatasetUsed == ImageClassificationMetrics.Dataset.Train)
                    return $"Phase: Training, Dataset used: {DatasetUsed.ToString(),10}, Batch Processed Count: {BatchProcessedCount,3}, Learning Rate: {LearningRate,10} " +
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
            /// Index of the input image.
            /// </summary>
            public int Index { get; set; }

            /// <summary>
            /// String representation of the metrics.
            /// </summary>
            public override string ToString() => $"Phase: Bottleneck Computation, Dataset used: {DatasetUsed.ToString(),10}, Image Index: {Index,3}";
        }

        /// <summary>
        /// Early Stopping feature stops training when monitored quantity stops improving'.
        /// Modeled after https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/
        /// tensorflow/python/keras/callbacks.py#L1143
        /// </summary>
        public sealed class EarlyStopping
        {
            /// <summary>
            /// Best value of metric seen so far.
            /// </summary>
            private float _bestMetricValue;

            /// <summary>
            /// Current counter for number of epochs where there has been no improvement.
            /// </summary>
            private int _wait;

            /// <summary>
            /// The metric to be monitored (eg Accuracy, Loss).
            /// </summary>
            private EarlyStoppingMetric _metric;

            /// <summary>
            /// Minimum change in the monitored quantity to be considered as an improvement.
            /// </summary>
            public float MinDelta { get; set; }

            /// <summary>
            /// Number of epochs to wait after no improvement is seen consecutively
            /// before stopping the training.
            /// </summary>
            public int Patience { get; set; }

            /// <summary>
            /// Whether the monitored quantity is to be increasing (eg. Accuracy, CheckIncreasing = true)
            /// or decreasing (eg. Loss, CheckIncreasing = false).
            /// </summary>
            public bool CheckIncreasing { get; set; }

            /// <param name="minDelta"></param>
            /// <param name="patience"></param>
            /// <param name="metric"></param>
            /// <param name="checkIncreasing"></param>
            public EarlyStopping(float minDelta = 0.01f, int patience = 20, EarlyStoppingMetric metric = EarlyStoppingMetric.Accuracy, bool checkIncreasing = true)
            {
                _bestMetricValue = 0.0f;
                _wait = 0;
                _metric = metric;
                MinDelta = Math.Abs(minDelta);
                Patience = patience;
                CheckIncreasing = checkIncreasing;

                //Set the CheckIncreasing according to the metric being monitored
                if (metric == EarlyStoppingMetric.Accuracy)
                    CheckIncreasing = true;
                else if (metric == EarlyStoppingMetric.Loss)
                    CheckIncreasing = false;
            }

            /// <summary>
            /// To be called at the end of every epoch to check if training should stop.
            /// For increasing metric(eg.: Accuracy), if metric stops increasing, stop training if
            /// value of metric doesn't increase within 'patience' number of epochs.
            /// For decreasing metric(eg.: Loss), stop training if value of metric doesn't decrease
            /// within 'patience' number of epochs.
            /// Any change  in the value of metric of less than 'minDelta' is not considered a change.
            /// </summary>
            public bool ShouldStop(TrainMetrics currentMetrics)
            {
                float currentMetricValue = _metric == EarlyStoppingMetric.Accuracy ? currentMetrics.Accuracy : currentMetrics.CrossEntropy;

                if (CheckIncreasing)
                {
                    if ((currentMetricValue - _bestMetricValue) < MinDelta)
                    {
                        _wait += 1;
                        if (_wait >= Patience)
                            return true;
                    }
                    else
                    {
                        _wait = 0;
                        _bestMetricValue = currentMetricValue;
                    }
                }
                else
                {
                    if ((_bestMetricValue - currentMetricValue) < MinDelta)
                    {
                        _wait += 1;
                        if (_wait >= Patience)
                            return true;
                    }
                    else
                    {
                        _wait = 0;
                        _bestMetricValue = currentMetricValue;
                    }
                }
                return false;
            }
        }

        /// <summary>
        /// Metrics for image classification bottlenect phase and training.
        /// Train metrics may be null when bottleneck phase is running, so have check!
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
            }

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
        /// Options class for <see cref="ImageClassificationTrainer"/>.
        /// </summary>
        public sealed class Options : TrainerInputBaseWithLabel
        {
            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of samples to use for mini-batch training.", SortOrder = 9)]
            public int BatchSize = 64;

            /// <summary>
            /// Number of training iterations.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of training iterations.", SortOrder = 10)]
            public int Epoch = 100;

            /// <summary>
            /// Learning rate to use during optimization.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate to use during optimization.", SortOrder = 12)]
            public float LearningRate = 0.01f;

            /// <summary>
            /// Early stopping technique parameters to be used to terminate training when training metric stops improving.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Early stopping technique parameters to be used to terminate training when training metric stops improving.", SortOrder = 15)]
            public EarlyStopping EarlyStoppingCriteria = new EarlyStopping();

            /// <summary>
            /// Specifies the model architecture to be used in the case of image classification training using transfer learning.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Model architecture to be used in transfer learning for image classification.", SortOrder = 15)]
            public Architecture Arch = Architecture.InceptionV3;

            /// <summary>
            /// Name of the tensor that will contain the output scores of the last layer when transfer learning is done.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Softmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string ScoreColumnName = "Score";

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
            public Action<ImageClassificationMetrics> MetricsCallback = null;

            /// <summary>
            /// Indicates the path where the newly retrained model should be saved.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the path where the newly retrained model should be saved.", SortOrder = 15)]
            public string ModelSavePath = null;

            /// <summary>
            /// Indicates to evaluate the model on train set after every epoch.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates to evaluate the model on train set after every epoch.", SortOrder = 15)]
            public bool TestOnTrainSet = true;

            /// <summary>
            /// Indicates to not re-compute cached bottleneck trainset values if already available in the bin folder.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates to not re-compute trained cached bottleneck values if already available in the bin folder.", SortOrder = 15)]
            public bool ReuseTrainSetBottleneckCachedValues = false;

            /// <summary>
            /// Indicates to not re-compute cached bottleneck validationset values if already available in the bin folder.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates to not re-compute validataionset cached bottleneck validationset values if already available in the bin folder.", SortOrder = 15)]
            public bool ReuseValidationSetBottleneckCachedValues = false;

            /// <summary>
            /// Validation set.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Validation set.", SortOrder = 15)]
            public IDataView ValidationSet;

            /// <summary>
            /// Indicates the file path to store trainset bottleneck values for caching.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the file path to store trainset bottleneck values for caching.", SortOrder = 15)]
            public string TrainSetBottleneckCachedValuesFilePath = "trainSetBottleneckFile.csv";

            /// <summary>
            /// Indicates the file path to store validationset bottleneck values for caching.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the file path to store validationset bottleneck values for caching.", SortOrder = 15)]
            public string ValidationSetBottleneckCachedValuesFilePath = "validationSetBottleneckFile.csv";

            /// <summary>
            /// A class that performs learning rate scheduling.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "A class that performs learning rate scheduling.", SortOrder = 15)]
            public LearningRateScheduler LearningRateScheduler;
        }

        /// <summary> Return the type of prediction task.</summary>
        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);

        /// <summary>
        /// Auxiliary information about the trainer in terms of its capabilities
        /// and requirements.
        /// </summary>
        public override TrainerInfo Info => _info;

        private readonly Options _options;
        private Session _session;
        private Operation _trainStep;
        private Tensor _bottleneckTensor;
        private Tensor _learningRateInput;
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
        private string _softmaxTensorName;
        private readonly string _checkpointPath;
        private readonly string _bottleneckOperationName;
        private readonly bool _useLRScheduling;
        private int _classCount;
        private Graph Graph => _session.graph;

        /// <summary>
        /// Initializes a new instance of <see cref="ImageClassificationTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="scoreColumn">The name of score column.</param>
        /// <param name="predictedLabelColumn">The name of the predicted label column.</param>
        /// <param name="validationSet">The validation set used while training to improve model quality.</param>
        internal ImageClassificationTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string scoreColumn = DefaultColumnNames.Score,
            string predictedLabelColumn = DefaultColumnNames.PredictedLabel,
            IDataView validationSet = null)
            : this(env, new Options()
            {
                FeatureColumnName = featureColumn,
                LabelColumnName = labelColumn,
                ScoreColumnName = scoreColumn,
                PredictedLabelColumnName = predictedLabelColumn,
                ValidationSet = validationSet
            })
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="ImageClassificationTrainer"/>
        /// </summary>
        internal ImageClassificationTrainer(IHostEnvironment env, Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadName),
                  new SchemaShape.Column(options.FeatureColumnName, SchemaShape.Column.VectorKind.VariableVector,
                      NumberDataViewType.Byte, false),
                  TrainerUtils.MakeU4ScalarColumn(options.LabelColumnName))
        {
            Host.CheckValue(options, nameof(options));
            Host.CheckNonEmpty(options.FeatureColumnName, nameof(options.FeatureColumnName));
            Host.CheckNonEmpty(options.LabelColumnName, nameof(options.LabelColumnName));
            Host.CheckNonEmpty(options.ScoreColumnName, nameof(options.ScoreColumnName));
            Host.CheckNonEmpty(options.PredictedLabelColumnName, nameof(options.PredictedLabelColumnName));

            _options = options;
            _session = DnnUtils.LoadDnnModel(env, _options.Arch, true).Session;
            _useLRScheduling = _options.LearningRateScheduler != null;
            _checkpointPath = _options.ModelSavePath ??
                Path.Combine(Directory.GetCurrentDirectory(), _options.FinalModelPrefix +
                    ModelLocation[_options.Arch]);

            // Configure bottleneck tensor based on the model.
            var arch = _options.Arch;
            if (arch == Architecture.ResnetV2101)
            {
                _bottleneckOperationName = "resnet_v2_101/SpatialSqueeze";
                _inputTensorName = "input";
            }
            else if (arch == Architecture.InceptionV3)
            {
                _bottleneckOperationName = "module_apply_default/hub_output/feature_vector/SpatialSqueeze";
                _inputTensorName = "Placeholder";
            }
            else if (arch == Architecture.MobilenetV2)
            {
                _bottleneckOperationName = "import/MobilenetV2/Logits/Squeeze";
                _inputTensorName = "import/input";
            }
            else if (arch == Architecture.ResnetV250)
            {
                _bottleneckOperationName = "resnet_v2_50/SpatialSqueeze";
                _inputTensorName = "input";
            }
        }

        private void InitializeTrainingGraph(IDataView input)
        {
            var labelColumn = input.Schema.GetColumnOrNull(_options.LabelColumnName).Value;
            var labelType = labelColumn.Type;
            var labelCount = labelType.GetKeyCount();
            if (labelCount <= 0)
            {
                throw Host.ExceptSchemaMismatch(nameof(input.Schema), "label", (string)labelColumn.Name, "Key",
                    (string)labelType.ToString());
            }

            _classCount = labelCount == 1 ? 2 : (int)labelCount;
            var imageSize = ImagePreprocessingSize[_options.Arch];
            (_jpegData, _resizedImage) = AddJpegDecoding(imageSize.Item1, imageSize.Item2, 3);
            _jpegDataTensorName = _jpegData.name;
            _resizedImageTensorName = _resizedImage.name;

            // Add transfer learning layer.
            AddTransferLearningLayer(_options.LabelColumnName, _options.ScoreColumnName, _options.LearningRate,
                _useLRScheduling, _classCount);

            // Initialize the variables.
            new Runner(_session).AddOperation(tf.global_variables_initializer()).Run();

            // Add evaluation layer.
            (_evaluationStep, _) = AddEvaluationStep(_softMaxTensor, _labelTensor);
            _softmaxTensorName = _softMaxTensor.name;
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(_options.LabelColumnName, out _);
            Contracts.Assert(success);
            var metadata = new List<SchemaShape.Column>();
            metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                TextDataViewType.Instance, false));

            return new[]
            {
                new SchemaShape.Column(_options.ScoreColumnName, SchemaShape.Column.VectorKind.Vector,
                    NumberDataViewType.Single, false),
                new SchemaShape.Column(_options.PredictedLabelColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.UInt32, true, new SchemaShape(metadata.ToArray()))
            };
        }

        private protected override MulticlassPredictionTransformer<ImageClassificationModelParameters> MakeTransformer(
            ImageClassificationModelParameters model, DataViewSchema trainSchema)
            => new MulticlassPredictionTransformer<ImageClassificationModelParameters>(Host, model, trainSchema,
                FeatureColumn.Name, LabelColumn.Name, _options.ScoreColumnName, _options.PredictedLabelColumnName);

        private protected override ImageClassificationModelParameters TrainModelCore(TrainContext trainContext)
        {
            InitializeTrainingGraph(trainContext.TrainingSet.Data);
            CheckTrainingParameters(_options);
            var validationSet = trainContext.ValidationSet?.Data ?? _options.ValidationSet;
            var imageProcessor = new ImageProcessor(_session, _jpegDataTensorName, _resizedImageTensorName);
            int trainingsetSize = -1;
            if (!_options.ReuseTrainSetBottleneckCachedValues ||
                !File.Exists(_options.TrainSetBottleneckCachedValuesFilePath))
            {
                trainingsetSize = CacheFeaturizedImagesToDisk(trainContext.TrainingSet.Data, _options.LabelColumnName,
                    _options.FeatureColumnName, imageProcessor,
                    _inputTensorName, _bottleneckTensor.name, _options.TrainSetBottleneckCachedValuesFilePath,
                    ImageClassificationMetrics.Dataset.Train, _options.MetricsCallback);

                // Write training set size to a file for use during training
                File.WriteAllText("TrainingSetSize.txt", trainingsetSize.ToString());
            }

            if (validationSet != null &&
                    (!_options.ReuseTrainSetBottleneckCachedValues ||
                    !File.Exists(_options.ValidationSetBottleneckCachedValuesFilePath)))
            {
                CacheFeaturizedImagesToDisk(validationSet, _options.LabelColumnName,
                    _options.FeatureColumnName, imageProcessor, _inputTensorName, _bottleneckTensor.name,
                    _options.ValidationSetBottleneckCachedValuesFilePath,
                    ImageClassificationMetrics.Dataset.Validation, _options.MetricsCallback);
            }

            TrainAndEvaluateClassificationLayer(_options.TrainSetBottleneckCachedValuesFilePath, _options,
                _options.ValidationSetBottleneckCachedValuesFilePath, trainingsetSize);

            // Leave the ownership of _session so that it is not disposed/closed when this object goes out of scope
            // since it will be used by ImageClassificationModelParameters class (new owner that will take care of
            // disposing).
            var session = _session;
            _session = null;
            return new ImageClassificationModelParameters(Host, session, _classCount, _jpegDataTensorName,
                _resizedImageTensorName, _inputTensorName, _softmaxTensorName);
        }

        private void CheckTrainingParameters(Options options)
        {
            Host.CheckNonWhiteSpace(options.LabelColumnName, nameof(options.LabelColumnName));

            if (_session.graph.OperationByName(_labelTensor.name.Split(':')[0]) == null)
            {
                throw Host.ExceptParam(nameof(_labelTensor.name), $"'{_labelTensor.name}' does not" +
                    $"exist in the model");
            }

            if (options.EarlyStoppingCriteria != null && options.ValidationSet == null &&
                options.TestOnTrainSet == false)
            {
                throw Host.ExceptParam(nameof(options.EarlyStoppingCriteria), $"Early stopping enabled but unable to" +
                    $"find a validation set and/or train set testing disabled. Please disable early stopping " +
                    $"or either provide a validation set or enable train set training.");
            }

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

        private static Tensor EncodeByteAsString(VBuffer<byte> buffer)
        {
            int length = buffer.Length;
            var size = c_api.TF_StringEncodedSize((UIntPtr)length);
            var handle = c_api.TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

            IntPtr tensor = c_api.TF_TensorData(handle);
            Marshal.WriteInt64(tensor, 0);

            var status = new Status();
            unsafe
            {
                fixed (byte* src = buffer.GetValues())
                    c_api.TF_StringEncode(src, (UIntPtr)length, (sbyte*)(tensor + sizeof(Int64)), size, status);
            }

            status.Check(true);
            status.Dispose();
            return new Tensor(handle);
        }

        internal sealed class ImageProcessor
        {
            private Runner _imagePreprocessingRunner;

            public ImageProcessor(Session session, string jpegDataTensorName, string resizeImageTensorName)
            {
                _imagePreprocessingRunner = new Runner(session);
                _imagePreprocessingRunner.AddInput(jpegDataTensorName);
                _imagePreprocessingRunner.AddOutputs(resizeImageTensorName);
            }

            public Tensor ProcessImage(in VBuffer<byte> imageBuffer)
            {
                var imageTensor = EncodeByteAsString(imageBuffer);
                var processedTensor = _imagePreprocessingRunner.AddInput(imageTensor, 0).Run()[0];
                imageTensor.Dispose();
                return processedTensor;
            }
        }

        private int CacheFeaturizedImagesToDisk(IDataView input, string labelColumnName, string imageColumnName,
            ImageProcessor imageProcessor, string inputTensorName, string outputTensorName, string cacheFilePath,
            ImageClassificationMetrics.Dataset dataset, Action<ImageClassificationMetrics> metricsCallback)
        {
            var labelColumn = input.Schema[labelColumnName];

            if (labelColumn.Type.RawType != typeof(UInt32))
                throw Host.ExceptSchemaMismatch(nameof(labelColumn), "Label",
                    labelColumnName, typeof(uint).ToString(),
                    labelColumn.Type.RawType.ToString());

            var imageColumn = input.Schema[imageColumnName];
            Runner runner = new Runner(_session);
            runner.AddOutputs(outputTensorName);
            int datasetsize = 0;
            using (TextWriter writer = File.CreateText(cacheFilePath))
            using (var cursor = input.GetRowCursor(
                input.Schema.Where(c => c.Index == labelColumn.Index || c.Index == imageColumn.Index)))
            {
                var labelGetter = cursor.GetGetter<uint>(labelColumn);
                var imageGetter = cursor.GetGetter<VBuffer<byte>>(imageColumn);
                UInt32 label = UInt32.MaxValue;
                VBuffer<byte> image = default;
                runner.AddInput(inputTensorName);
                ImageClassificationMetrics metrics = new ImageClassificationMetrics();
                metrics.Bottleneck = new BottleneckMetrics();
                metrics.Bottleneck.DatasetUsed = dataset;
                float[] imageArray = null;
                while (cursor.MoveNext())
                {
                    labelGetter(ref label);
                    imageGetter(ref image);
                    if (image.Length <= 0)
                        continue; //Empty Image

                    var imageTensor = imageProcessor.ProcessImage(image);
                    runner.AddInput(imageTensor, 0);
                    var featurizedImage = runner.Run()[0]; // Reuse memory
                    featurizedImage.ToArray<float>(ref imageArray);
                    Host.Assert((int)featurizedImage.size == imageArray.Length);
                    writer.WriteLine(label - 1 + "," + string.Join(",", imageArray));
                    featurizedImage.Dispose();
                    imageTensor.Dispose();
                    metrics.Bottleneck.Index++;
                    metricsCallback?.Invoke(metrics);
                }
                datasetsize = metrics.Bottleneck.Index;
            }
            return datasetsize;
        }

        private IDataView GetShuffledData(string path)
        {
            return new RowShufflingTransformer(
                Host,
                new RowShufflingTransformer.Options
                {
                    ForceShuffle = true,
                    ForceShuffleSource = true
                },
                new TextLoader(
                    Host,
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

        private int GetNumSamples(string path)
        {
            using var reader = File.OpenText(path);
            return int.Parse(reader.ReadLine());
        }

        private void TrainAndEvaluateClassificationLayer(string trainBottleneckFilePath, Options options,
            string validationSetBottleneckFilePath, int trainingsetSize)
        {
            int batchSize = options.BatchSize;
            int epochs = options.Epoch;
            float learningRate = options.LearningRate;
            bool evaluateOnly = !string.IsNullOrEmpty(validationSetBottleneckFilePath);
            Action<ImageClassificationMetrics> statisticsCallback = _options.MetricsCallback;
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
            trainWriter = tf.summary.FileWriter(Path.Combine(Directory.GetCurrentDirectory(), "train"),
                _session.graph);

            trainSaver = tf.train.Saver();
            trainSaver.save(_session, _checkpointPath);

            runner.AddInput(_bottleneckInput.name).AddInput(_labelTensor.name);
            if (options.LearningRateScheduler != null)
                runner.AddInput(_learningRateInput.name);
            testEvalRunner.AddInput(_bottleneckInput.name).AddInput(_labelTensor.name);
            Dictionary<long, int> classStatsTrain = new Dictionary<long, int>();
            Dictionary<long, int> classStatsValidate = new Dictionary<long, int>();
            for (int index = 0; index < _classCount; index += 1)
                classStatsTrain[index] = classStatsValidate[index] = 0;

            ImageClassificationMetrics metrics = new ImageClassificationMetrics();
            metrics.Train = new TrainMetrics();
            float accuracy = 0;
            float crossentropy = 0;
            TrainState trainstate = new TrainState
            {
                BatchSize = options.BatchSize,
                BatchesPerEpoch =
                (trainingsetSize < 0 ? GetNumSamples("TrainingSetSize.txt") : trainingsetSize) / options.BatchSize
            };

            for (int epoch = 0; epoch < epochs; epoch += 1)
            {
                batchIndex = 0;
                metrics.Train.Accuracy = 0;
                metrics.Train.CrossEntropy = 0;
                metrics.Train.BatchProcessedCount = 0;
                metrics.Train.LearningRate = learningRate;
                // Update train state.
                trainstate.CurrentEpoch = epoch;
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
                        trainstate.CurrentBatchIndex = batchIndex;
                        // Train.
                        if (batchIndex == batchSize)
                        {
                            runner.AddInput(new Tensor(featureBatchPtr, featureTensorShape, TF_DataType.TF_FLOAT,
                                                  featureBatchSizeInBytes), 0)
                                .AddInput(new Tensor(labelBatchPtr, labelTensorShape, TF_DataType.TF_INT64,
                                                  labelBatchSizeInBytes), 1);

                            if (options.LearningRateScheduler != null)
                            {
                                // Add learning rate as a placeholder only when learning rate scheduling is used.
                                learningRate = options.LearningRateScheduler.GetLearningRate(trainstate);
                                metrics.Train.LearningRate = learningRate;
                                runner.AddInput(new Tensor(learningRate, TF_DataType.TF_FLOAT), 2);
                            }
                            runner.Run();

                            metrics.Train.BatchProcessedCount += 1;
                            if (options.TestOnTrainSet && statisticsCallback != null)
                            {
                                var outputTensors = testEvalRunner
                                    .AddInput(new Tensor(featureBatchPtr, featureTensorShape, TF_DataType.TF_FLOAT,
                                                    featureBatchSizeInBytes), 0)
                                    .AddInput(new Tensor(labelBatchPtr, labelTensorShape, TF_DataType.TF_INT64,
                                                    labelBatchSizeInBytes), 1)
                                    .Run();

                                outputTensors[0].ToScalar<float>(ref accuracy);
                                outputTensors[1].ToScalar<float>(ref crossentropy);
                                metrics.Train.Accuracy += accuracy;
                                metrics.Train.CrossEntropy += crossentropy;

                                outputTensors[0].Dispose();
                                outputTensors[1].Dispose();
                            }

                            batchIndex = 0;
                        }
                    }

                    //Process last incomplete batch
                    if (batchIndex > 0)
                    {
                        featureTensorShape[0] = batchIndex;
                        featureBatchSizeInBytes = sizeof(float) * featureLength * batchIndex;
                        labelTensorShape[0] = batchIndex;
                        labelBatchSizeInBytes = sizeof(long) * batchIndex;
                        runner.AddInput(new Tensor(featureBatchPtr, featureTensorShape, TF_DataType.TF_FLOAT,
                                                featureBatchSizeInBytes), 0)
                                .AddInput(new Tensor(labelBatchPtr, labelTensorShape, TF_DataType.TF_INT64,
                                                labelBatchSizeInBytes), 1);
                        if (options.LearningRateScheduler != null)
                        {
                            // Add learning rate as a placeholder only when learning rate scheduling is used.
                            learningRate = options.LearningRateScheduler.GetLearningRate(trainstate);
                            metrics.Train.LearningRate = learningRate;
                            runner.AddInput(new Tensor(learningRate, TF_DataType.TF_FLOAT), 2);
                        }
                        runner.Run();

                        metrics.Train.BatchProcessedCount += 1;

                        if (options.TestOnTrainSet && statisticsCallback != null)
                        {
                            var outputTensors = testEvalRunner
                                .AddInput(new Tensor(featureBatchPtr, featureTensorShape, TF_DataType.TF_FLOAT,
                                                  featureBatchSizeInBytes), 0)
                                .AddInput(new Tensor(labelBatchPtr, labelTensorShape, TF_DataType.TF_INT64,
                                                  labelBatchSizeInBytes), 1)
                                .Run();

                            outputTensors[0].ToScalar<float>(ref accuracy);
                            outputTensors[1].ToScalar<float>(ref crossentropy);
                            metrics.Train.Accuracy += accuracy;
                            metrics.Train.CrossEntropy += crossentropy;

                            outputTensors[0].Dispose();
                            outputTensors[1].Dispose();
                        }

                        batchIndex = 0;
                        featureTensorShape[0] = batchSize;
                        featureBatchSizeInBytes = sizeof(float) * featureBatch.Length;
                        labelTensorShape[0] = batchSize;
                        labelBatchSizeInBytes = sizeof(long) * batchSize;
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
                {
                    //Early stopping check
                    if (options.EarlyStoppingCriteria != null)
                    {
                        if (options.EarlyStoppingCriteria.ShouldStop(metrics.Train))
                            break;
                    }
                    continue;
                }

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
                                .AddInput(new Tensor(featureBatchPtr, featureTensorShape, TF_DataType.TF_FLOAT,
                                                  featureBatchSizeInBytes), 0)
                                .AddInput(new Tensor(labelBatchPtr, labelTensorShape, TF_DataType.TF_INT64,
                                                  labelBatchSizeInBytes), 1)
                                .Run();

                            outputTensors[0].ToScalar<float>(ref accuracy);
                            metrics.Train.Accuracy += accuracy;
                            metrics.Train.BatchProcessedCount += 1;
                            batchIndex = 0;

                            outputTensors[0].Dispose();
                        }
                    }

                    //Process last incomplete batch
                    if (batchIndex > 0)
                    {
                        featureTensorShape[0] = batchIndex;
                        featureBatchSizeInBytes = sizeof(float) * featureLength * batchIndex;
                        labelTensorShape[0] = batchIndex;
                        labelBatchSizeInBytes = sizeof(long) * batchIndex;
                        var outputTensors = validationEvalRunner
                                .AddInput(new Tensor(featureBatchPtr, featureTensorShape, TF_DataType.TF_FLOAT,
                                                  featureBatchSizeInBytes), 0)
                                .AddInput(new Tensor(labelBatchPtr, labelTensorShape, TF_DataType.TF_INT64,
                                                  labelBatchSizeInBytes), 1)
                                .Run();

                        outputTensors[0].ToScalar<float>(ref accuracy);
                        metrics.Train.Accuracy += accuracy;
                        metrics.Train.BatchProcessedCount += 1;
                        batchIndex = 0;

                        featureTensorShape[0] = batchSize;
                        featureBatchSizeInBytes = sizeof(float) * featureBatch.Length;
                        labelTensorShape[0] = batchSize;
                        labelBatchSizeInBytes = sizeof(long) * batchSize;

                        outputTensors[0].Dispose();
                    }

                    if (statisticsCallback != null)
                    {
                        metrics.Train.Epoch = epoch;
                        metrics.Train.Accuracy /= metrics.Train.BatchProcessedCount;
                        metrics.Train.DatasetUsed = ImageClassificationMetrics.Dataset.Validation;
                        statisticsCallback(metrics);
                    }
                }

                //Early stopping check
                if (options.EarlyStoppingCriteria != null)
                {
                    if (options.EarlyStoppingCriteria.ShouldStop(metrics.Train))
                        break;
                }
            }

            trainSaver.save(_session, _checkpointPath);
            UpdateTransferLearningModelOnDisk(_classCount);
        }

        private (Session, Tensor, Tensor, Tensor) BuildEvaluationSession(int classCount)
        {
            var evalGraph = DnnUtils.LoadMetaGraph(ModelLocation[_options.Arch]);
            var evalSess = tf.Session(graph: evalGraph);
            Tensor evaluationStep = null;
            Tensor prediction = null;
            Tensor bottleneckTensor = evalGraph.OperationByName(_bottleneckOperationName);
            evalGraph.as_default();
            var (_, _, groundTruthInput, finalTensor) = AddFinalRetrainOps(classCount, _options.LabelColumnName,
                    _options.ScoreColumnName, bottleneckTensor, false, (_options.LearningRateScheduler == null ? false : true), _options.LearningRate);
            tf.train.Saver().restore(evalSess, _checkpointPath);
            (evaluationStep, prediction) = AddEvaluationStep(finalTensor, groundTruthInput);
            var imageSize = ImagePreprocessingSize[_options.Arch];
            (_jpegData, _resizedImage) = AddJpegDecoding(imageSize.Item1, imageSize.Item2, 3);
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

        private void UpdateTransferLearningModelOnDisk(int classCount)
        {
            var (sess, _, _, _) = BuildEvaluationSession(classCount);
            var graph = sess.graph;
            var outputGraphDef = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), new string[] { _softMaxTensor.name.Split(':')[0],
                    _prediction.name.Split(':')[0], _jpegData.name.Split(':')[0], _resizedImage.name.Split(':')[0] });

            string frozenModelPath = _checkpointPath + ".pb";
            File.WriteAllBytes(_checkpointPath + ".pb", outputGraphDef.ToByteArray());
            _session.graph.Dispose();
            _session.Dispose();
            _session = LoadTFSessionByModelFilePath(Host, frozenModelPath, false);
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
            string scoreColumnName, Tensor bottleneckTensor, bool isTraining, bool useLearningRateScheduler,
            float learningRate)
        {
            var bottleneckTensorDims = bottleneckTensor.TensorShape.dims;
            var (batch_size, bottleneck_tensor_size) = (bottleneckTensorDims[0], bottleneckTensorDims[1]);
            tf_with(tf.name_scope("input"), scope =>
            {
                if (isTraining)
                {
                    _bottleneckInput = tf.placeholder_with_default(
                        bottleneckTensor,
                        shape: bottleneckTensorDims,
                        name: "BottleneckInputPlaceholder");
                    if (useLearningRateScheduler)
                        _learningRateInput = tf.placeholder(tf.float32, null, name: "learningRateInputPlaceholder");

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
                    var initialValue = tf.truncated_normal(new int[] { bottleneck_tensor_size, classCount },
                        stddev: 0.001f);

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
                var optimizer = useLearningRateScheduler ? tf.train.GradientDescentOptimizer(_learningRateInput) :
                                    tf.train.GradientDescentOptimizer(learningRate);

                _trainStep = optimizer.minimize(crossEntropyMean);
            });

            return (_trainStep, crossEntropyMean, _labelTensor, _softMaxTensor);
        }

        private void AddTransferLearningLayer(string labelColumn,
            string scoreColumnName, float learningRate, bool useLearningRateScheduling, int classCount)
        {
            _bottleneckTensor = Graph.OperationByName(_bottleneckOperationName);
            (_trainStep, _crossEntropy, _labelTensor, _softMaxTensor) =
                    AddFinalRetrainOps(classCount, labelColumn, scoreColumnName, _bottleneckTensor, true,
                        useLearningRateScheduling, learningRate);

        }

        ~ImageClassificationTrainer()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            // Ensure that the Session is not null and it's handle is not Zero, as it may have already been
            // disposed/finalized. Technically we shouldn't be calling this if disposing == false,
            // since we're running in finalizer and the GC doesn't guarantee ordering of finalization of managed
            // objects, but we have to make sure that the Session is closed before deleting our temporary directory.
            if (_session != null && _session != IntPtr.Zero)
            {
                _session.close();
            }
        }

        /// <summary>
        /// Trains a <see cref="ImageClassificationTrainer"/> using both training and validation data,
        /// returns a <see cref="ImageClassificationModelParameters"/>.
        /// </summary>
        /// <param name="trainData">The training data set.</param>
        /// <param name="validationData">The validation data set.</param>
        public MulticlassPredictionTransformer<ImageClassificationModelParameters> Fit(
            IDataView trainData, IDataView validationData) => TrainTransformer(trainData, validationData);
    }

    /// <summary>
    /// Image Classification predictor. This class encapsulates the trained Deep Neural Network(DNN) model
    /// and is used to score images.
    /// </summary>
    public sealed class ImageClassificationModelParameters : ModelParametersBase<VBuffer<float>>, IValueMapper
    {
        internal const string LoaderSignature = "ImageClassificationPred";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMAGPRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImageClassificationModelParameters).Assembly.FullName);
        }

        private readonly VectorDataViewType _inputType;
        private readonly VectorDataViewType _outputType;
        private readonly int _classCount;
        private readonly string _imagePreprocessorTensorInput;
        private readonly string _imagePreprocessorTensorOutput;
        private readonly string _graphInputTensor;
        private readonly string _graphOutputTensor;
        private readonly Session _session;

        internal ImageClassificationModelParameters(IHostEnvironment env, Session session, int classCount,
            string imagePreprocessorTensorInput, string imagePreprocessorTensorOutput, string graphInputTensor,
            string graphOutputTensor) : base(env, LoaderSignature)
        {
            Host.AssertValue(session);
            Host.Assert(classCount > 1);
            Host.AssertNonEmpty(imagePreprocessorTensorInput);
            Host.AssertNonEmpty(imagePreprocessorTensorOutput);
            Host.AssertNonEmpty(graphInputTensor);
            Host.AssertNonEmpty(graphOutputTensor);

            _inputType = new VectorDataViewType(NumberDataViewType.Byte);
            _outputType = new VectorDataViewType(NumberDataViewType.Single, classCount);
            _classCount = classCount;
            _session = session;
            _imagePreprocessorTensorInput = imagePreprocessorTensorInput;
            _imagePreprocessorTensorOutput = imagePreprocessorTensorOutput;
            _graphInputTensor = graphInputTensor;
            _graphOutputTensor = graphOutputTensor;
        }

        /// <summary> Return the type of prediction task.</summary>
        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        DataViewType IValueMapper.InputType => _inputType;

        DataViewType IValueMapper.OutputType => _outputType;

        private ImageClassificationModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: _classCount
            // string: _imagePreprocessorTensorInput
            // string: _imagePreprocessorTensorOutput
            // string: _graphInputTensor
            // string: _graphOutputTensor
            // Graph.

            _classCount = ctx.Reader.ReadInt32();
            _imagePreprocessorTensorInput = ctx.Reader.ReadString();
            _imagePreprocessorTensorOutput = ctx.Reader.ReadString();
            _graphInputTensor = ctx.Reader.ReadString();
            _graphOutputTensor = ctx.Reader.ReadString();
            byte[] modelBytes = null;
            if (!ctx.TryLoadBinaryStream("TFModel", r => modelBytes = r.ReadByteArray()))
                throw env.ExceptDecode();

            _session = LoadTFSession(env, modelBytes);
            _inputType = new VectorDataViewType(NumberDataViewType.Byte);
            _outputType = new VectorDataViewType(NumberDataViewType.Single, _classCount);
        }

        private static ImageClassificationModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new ImageClassificationModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: _classCount
            // string: _imagePreprocessorTensorInput
            // string: _imagePreprocessorTensorOutput
            // string: _graphInputTensor
            // string: _graphOutputTensor
            // Graph.

            ctx.Writer.Write(_classCount);
            ctx.Writer.Write(_imagePreprocessorTensorInput);
            ctx.Writer.Write(_imagePreprocessorTensorOutput);
            ctx.Writer.Write(_graphInputTensor);
            ctx.Writer.Write(_graphOutputTensor);

            Status status = new Status();
            var buffer = _session.graph.ToGraphDef(status);
            ctx.SaveBinaryStream("TFModel", w =>
            {
                w.WriteByteArray(buffer.MemoryBlock.ToArray());
            });
            status.Check(true);
        }

        private class Classifier
        {
            private Runner _runner;
            private ImageClassificationTrainer.ImageProcessor _imageProcessor;

            public Classifier(ImageClassificationModelParameters model)
            {
                _runner = new Runner(model._session);
                _runner.AddInput(model._graphInputTensor);
                _runner.AddOutputs(model._graphOutputTensor);
                _imageProcessor = new ImageClassificationTrainer.ImageProcessor(model._session,
                    model._imagePreprocessorTensorInput, model._imagePreprocessorTensorOutput);
            }

            public void Score(in VBuffer<byte> image, Span<float> classProbabilities)
            {
                var processedTensor = _imageProcessor.ProcessImage(image);
                var outputTensor = _runner.AddInput(processedTensor, 0).Run();
                outputTensor[0].CopyTo(classProbabilities);
                outputTensor[0].Dispose();
                processedTensor.Dispose();
            }
        }

        ValueMapper<TSrc, TDst> IValueMapper.GetMapper<TSrc, TDst>()
        {
            Host.Check(typeof(TSrc) == typeof(VBuffer<byte>));
            Host.Check(typeof(TDst) == typeof(VBuffer<float>));
            _session.graph.as_default();
            Classifier classifier = new Classifier(this);
            ValueMapper<VBuffer<byte>, VBuffer<float>> del = (in VBuffer<byte> src, ref VBuffer<float> dst) =>
            {
                var editor = VBufferEditor.Create(ref dst, _classCount);
                classifier.Score(src, editor.Values);
                dst = editor.Commit();
            };

            return (ValueMapper<TSrc, TDst>)(Delegate)del;
        }

        ~ImageClassificationModelParameters()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            // Ensure that the Session is not null and it's handle is not Zero, as it may have already been
            // disposed/finalized. Technically we shouldn't be calling this if disposing == false,
            // since we're running in finalizer and the GC doesn't guarantee ordering of finalization of managed
            // objects, but we have to make sure that the Session is closed before deleting our temporary directory.
            if (_session != null && _session != IntPtr.Zero)
            {
                _session.close();
            }
        }
    }
}
