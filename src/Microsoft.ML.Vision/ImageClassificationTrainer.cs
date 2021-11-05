// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using Google.Protobuf;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TensorFlow;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using Tensorflow;
using Tensorflow.Summaries;
using static Microsoft.ML.Data.TextLoader;
using static Microsoft.ML.TensorFlow.TensorFlowUtils;
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

namespace Microsoft.ML.Vision
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a Deep Neural Network(DNN) to classify images.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [ImageClassification](xref:Microsoft.ML.VisionCatalog.ImageClassification(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String,System.String,System.String,Microsoft.ML.IDataView)).
    ///
    /// ### Input and Output Columns
    /// The input label column data must be [key](xref:Microsoft.ML.Data.KeyDataViewType) type and the feature column must be a variable-sized vector of <xref:System.Byte>.
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
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.Vision and SciSharp.TensorFlow.Redist / SciSharp.TensorFlow.Redist-Windows-GPU / SciSharp.TensorFlow.Redist-Linux-GPU |
    /// | Exportable to ONNX | No |
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/tensorflow-usage.md)]
    ///
    /// ### Training Algorithm Details
    /// Trains a Deep Neural Network(DNN) by leveraging an existing pre-trained model such as Resnet50 for the purpose
    /// of classifying images. The technique was inspired from [TensorFlow's retrain image classification tutorial](https://www.tensorflow.org/hub/tutorials/image_retraining)
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
        internal static IReadOnlyDictionary<Architecture, string> ModelFileName = new Dictionary<Architecture, string>
        {
            { Architecture.ResnetV2101, @"resnet_v2_101_299.meta" },
            { Architecture.InceptionV3, @"inception_v3.meta" },
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
                    return $"Phase: Training, Dataset used: {DatasetUsed.ToString(),10}, Batch Processed Count: {BatchProcessedCount,3}, " +
                        $"Epoch: {Epoch,3}, Accuracy: {Accuracy,10}, Cross-Entropy: {CrossEntropy,10}, Learning Rate: {LearningRate,10}";
                else
                    return $"Phase: Training, Dataset used: {DatasetUsed.ToString(),10}, Batch Processed Count: {BatchProcessedCount,3}, " +
                        $"Epoch: {Epoch,3}, Accuracy: {Accuracy,10}, Cross-Entropy: {CrossEntropy,10}";
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
            private readonly EarlyStoppingMetric _metric;

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
                {
                    CheckIncreasing = false;
                    _bestMetricValue = Single.MaxValue;
                }
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
        /// Metrics for image classification bottleneck phase and training.
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
            /// Number of samples to use for mini-batch training. The default value for BatchSize is 10.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of samples to use for mini-batch training.", SortOrder = 9)]
            public int BatchSize = 10;

            /// <summary>
            /// Number of training iterations. The default value for Epoch is 200.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of training iterations.", SortOrder = 10)]
            public int Epoch = 200;

            /// <summary>
            /// Learning rate to use during optimization. The default value for Learning Rate is 0.01.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate to use during optimization.", SortOrder = 12)]
            public float LearningRate = 0.01f;

            /// <summary>
            /// Early stopping technique parameters to be used to terminate training when training metric stops improving. By default EarlyStopping is turned on and the monitoring metric is Accuracy.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Early stopping technique parameters to be used to terminate training when training metric stops improving.", SortOrder = 15)]
            public EarlyStopping EarlyStoppingCriteria = new EarlyStopping();

            /// <summary>
            /// Specifies the model architecture to be used in the case of image classification training using transfer learning. The default Architecture is Resnet_v2_50.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Model architecture to be used in transfer learning for image classification.", SortOrder = 15)]
            public Architecture Arch = Architecture.ResnetV250;

            /// <summary>
            /// Name of the tensor that will contain the output scores of the last layer when transfer learning is done. The default tensor name is "Score".
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Softmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string ScoreColumnName = "Score";

            /// <summary>
            /// Name of the tensor that will contain the predicted label from output scores of the last layer when transfer learning is done. The default tensor name is "PredictedLabel".
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Argmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string PredictedLabelColumnName = "PredictedLabel";

            /// <summary>
            /// Final model and checkpoint files/folder prefix for storing graph files. The default prefix is "custom_retrained_model_based_on_".
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Final model and checkpoint files/folder prefix for storing graph files.", SortOrder = 15)]
            public string FinalModelPrefix = "custom_retrained_model_based_on_";

            /// <summary>
            /// Callback to report statistics on accuracy/cross entropy during training phase. Metrics Callback is set to null by default.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Callback to report metrics during training and validation phase.", SortOrder = 15)]
            public Action<ImageClassificationMetrics> MetricsCallback = null;

            /// <summary>
            /// Indicates the path where the image bottleneck cache files and trained model are saved, default is a new temporary directory.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the path where the models get downloaded to and cache files saved, default is a new temporary directory.", SortOrder = 15)]
            public string WorkspacePath = null;

            /// <summary>
            /// Indicates to evaluate the model on train set after every epoch. Test on trainset is set to true by default.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates to evaluate the model on train set after every epoch.", SortOrder = 15)]
            public bool TestOnTrainSet = true;

            /// <summary>
            /// Indicates to not re-compute cached bottleneck trainset values if already available in the bin folder. This parameter is set to false by default.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates to not re-compute trained cached bottleneck values if already available in the bin folder.", SortOrder = 15)]
            public bool ReuseTrainSetBottleneckCachedValues = false;

            /// <summary>
            /// Indicates to not re-compute cached bottleneck validationset values if already available in the bin folder. This parameter is set to false by default.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates to not re-compute validataionset cached bottleneck validationset values if already available in the bin folder.", SortOrder = 15)]
            public bool ReuseValidationSetBottleneckCachedValues = false;

            /// <summary>
            /// Validation set.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Validation set.", SortOrder = 15)]
            public IDataView ValidationSet;

            /// <summary>
            /// When validation set is not passed then a fraction of train set is used as validation. To disable this
            /// behavior set <see cref="ValidationSetFraction"/> to null. Accepts value between 0 and 1.0, default
            /// value is 0.1 or 10% of the trainset.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Validation fraction.", SortOrder = 15)]
            public float? ValidationSetFraction = 0.1f;

            /// <summary>
            /// Indicates the file name within the workspace to store trainset bottleneck values for caching, default file name is "trainSetBottleneckFile.csv".
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the file name to store trainset bottleneck values for caching.", SortOrder = 15)]
            public string TrainSetBottleneckCachedValuesFileName = "trainSetBottleneckFile.csv";

            /// <summary>
            /// Indicates the file name within the workspace to store validationset  bottleneck values for caching, default file name is "validationSetBottleneckFile.csv".
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the file name to store validationset bottleneck values for caching.", SortOrder = 15)]
            public string ValidationSetBottleneckCachedValuesFileName = "validationSetBottleneckFile.csv";

            /// <summary>
            /// A class that performs learning rate scheduling. The default learning rate scheduler is exponential learning rate decay.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "A class that performs learning rate scheduling.", SortOrder = 15)]
            public LearningRateScheduler LearningRateScheduler = new ExponentialLRDecay();
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
        private readonly string _inputTensorName;
        private string _softmaxTensorName;
        private readonly string _checkpointPath;
        private readonly string _bottleneckOperationName;
        private readonly bool _useLRScheduling;
        private readonly bool _cleanupWorkspace;
        private int _classCount;
        private Graph Graph => _session.graph;
        private readonly string _resourcePath;
        private readonly string _sizeFile;

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
            tf.compat.v1.disable_eager_execution();
            _resourcePath = Path.Combine(((IHostEnvironmentInternal)env).TempFilePath, "MLNET");

            if (string.IsNullOrEmpty(options.WorkspacePath))
            {
                options.WorkspacePath = GetTemporaryDirectory(env);
                _cleanupWorkspace = true;
            }

            if (!Directory.Exists(_resourcePath))
            {
                Directory.CreateDirectory(_resourcePath);
            }

            if (string.IsNullOrEmpty(options.TrainSetBottleneckCachedValuesFileName))
            {
                //If the user decided to set to null reset back to default value
                options.TrainSetBottleneckCachedValuesFileName = _options.TrainSetBottleneckCachedValuesFileName;
            }

            if (string.IsNullOrEmpty(options.ValidationSetBottleneckCachedValuesFileName))
            {
                //If the user decided to set to null reset back to default value
                options.ValidationSetBottleneckCachedValuesFileName = _options.ValidationSetBottleneckCachedValuesFileName;
            }

            if (options.MetricsCallback == null)
            {
                var logger = Host.Start(nameof(ImageClassificationTrainer));
                options.MetricsCallback = (ImageClassificationMetrics metric) => { logger.Trace(metric.ToString()); };
            }

            _options = options;
            _useLRScheduling = _options.LearningRateScheduler != null;
            _checkpointPath = Path.Combine(_options.WorkspacePath, _options.FinalModelPrefix +
                    ModelFileName[_options.Arch]);
            _sizeFile = Path.Combine(_options.WorkspacePath, "TrainingSetSize.txt");

            // Configure bottleneck tensor based on the model.
            var arch = _options.Arch;
            if (arch == Architecture.ResnetV2101)
            {
                _bottleneckOperationName = "resnet_v2_101/SpatialSqueeze";
                _inputTensorName = "input";
            }
            else if (arch == Architecture.InceptionV3)
            {
                _bottleneckOperationName = "InceptionV3/Logits/SpatialSqueeze";
                _inputTensorName = "input";
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

            var msg = $"Only one class found in the {_options.LabelColumnName} column. To build a multiclass classification model, the number of classes needs to be 2 or greater";
            Contracts.CheckParam(labelCount > 1, nameof(labelCount), msg);

            _classCount = (int)labelCount;
            var imageSize = ImagePreprocessingSize[_options.Arch];
            _session = LoadTensorFlowSessionFromMetaGraph(Host, _options.Arch).Session;
            _session.graph.as_default();
            (_jpegData, _resizedImage) = AddJpegDecoding(imageSize.Item1, imageSize.Item2, 3);
            _jpegDataTensorName = _jpegData.name;
            _resizedImageTensorName = _resizedImage.name;

            // Add transfer learning layer.
            AddTransferLearningLayer(_options.LabelColumnName, _options.ScoreColumnName, _options.LearningRate,
                _useLRScheduling, _classCount);

            // Initialize the variables.
            new Runner(_session, operations: new IntPtr[] { tf.global_variables_initializer() }).Run();

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
            // Workspace directory is cleaned after training run. However, the pipeline can be re-used by calling
            // fit() again after transform(), in which case we must ensure workspace directory exists. This scenario
            // is typical in the case of cross-validation.
            if (!Directory.Exists(_options.WorkspacePath))
            {
                Directory.CreateDirectory(_options.WorkspacePath);
            }

            InitializeTrainingGraph(trainContext.TrainingSet.Data);
            CheckTrainingParameters(_options);
            var validationSet = trainContext.ValidationSet?.Data ?? _options.ValidationSet;
            var imageProcessor = new ImageProcessor(_session, _jpegDataTensorName, _resizedImageTensorName);
            string trainSetBottleneckCachedValuesFilePath = Path.Combine(_options.WorkspacePath,
                _options.TrainSetBottleneckCachedValuesFileName);

            string validationSetBottleneckCachedValuesFilePath = Path.Combine(_options.WorkspacePath,
                _options.ValidationSetBottleneckCachedValuesFileName);

            bool needValidationSet = _options.EarlyStoppingCriteria != null || _options.MetricsCallback != null;
            bool validationSetPresent = _options.ReuseValidationSetBottleneckCachedValues &&
                File.Exists(validationSetBottleneckCachedValuesFilePath + "_features.bin") &&
                    File.Exists(validationSetBottleneckCachedValuesFilePath + "_labels.bin");

            bool generateValidationSet = needValidationSet && !validationSetPresent;

            if (generateValidationSet && _options.ValidationSet != null)
            {
                CacheFeaturizedImagesToDisk(validationSet, _options.LabelColumnName,
                    _options.FeatureColumnName, imageProcessor, _inputTensorName, _bottleneckTensor.name,
                    validationSetBottleneckCachedValuesFilePath,
                    ImageClassificationMetrics.Dataset.Validation, _options.MetricsCallback);

                generateValidationSet = false;
                validationSetPresent = true;
            }

            if (!_options.ReuseTrainSetBottleneckCachedValues ||
                !(File.Exists(trainSetBottleneckCachedValuesFilePath + "_features.bin") &&
                    File.Exists(trainSetBottleneckCachedValuesFilePath + "_labels.bin")))
            {
                CacheFeaturizedImagesToDisk(trainContext.TrainingSet.Data, _options.LabelColumnName,
                    _options.FeatureColumnName, imageProcessor,
                    _inputTensorName, _bottleneckTensor.name, trainSetBottleneckCachedValuesFilePath,
                    ImageClassificationMetrics.Dataset.Train, _options.MetricsCallback,
                    generateValidationSet ? _options.ValidationSetFraction : null);

                validationSetPresent = validationSetPresent ||
                    (generateValidationSet && _options.ValidationSetFraction.HasValue);

                generateValidationSet = needValidationSet && !validationSetPresent;
            }

            if (generateValidationSet && _options.ReuseTrainSetBottleneckCachedValues &&
                !_options.ReuseValidationSetBottleneckCachedValues)
            {
                // Not sure if it makes sense to support this scenario.
            }

            Contracts.Assert(!generateValidationSet, "Validation set needed but cannot generate.");

            TrainAndEvaluateClassificationLayer(trainSetBottleneckCachedValuesFilePath,
                validationSetPresent && (_options.EarlyStoppingCriteria != null || _options.MetricsCallback != null) ?
                    validationSetBottleneckCachedValuesFilePath : null);

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
            var resizedImage = tf.image.resize_bilinear(decodedImage4d, resizeShapeAsInt, false, name: "ResizeTensor");
            return (jpegData, resizedImage);
        }

        private static Tensor EncodeByteAsString(VBuffer<byte> buffer)
        {
            int length = buffer.Length;
            var size = c_api.TF_StringEncodedSize((ulong)length);
            var handle = c_api.TF_AllocateTensor(TF_DataType.TF_STRING, Array.Empty<long>(), 0, ((ulong)size + 8));

            IntPtr tensor = c_api.TF_TensorData(handle);
            Marshal.WriteInt64(tensor, 0);

            var status = new Status();
            unsafe
            {
                fixed (byte* src = buffer.GetValues())
                    c_api.TF_StringEncode(src, (ulong)length, (byte*)(tensor + sizeof(Int64)), size, status.Handle);
            }

            status.Check(true);
            status.Dispose();
            return new Tensor(handle);
        }

        internal sealed class ImageProcessor
        {
            private readonly Runner _imagePreprocessingRunner;

            public ImageProcessor(Session session, string jpegDataTensorName, string resizeImageTensorName)
            {
                _imagePreprocessingRunner = new Runner(session, new[] { jpegDataTensorName },
                    new[] { resizeImageTensorName });
            }

            public Tensor ProcessImage(in VBuffer<byte> imageBuffer)
            {
                using (var imageTensor = EncodeByteAsString(imageBuffer))
                {
                    try
                    {
                        return _imagePreprocessingRunner.AddInput(imageTensor, 0).Run()[0];
                    }
                    catch (TensorflowException e)
                    {
                        //catch the exception for images of unknown format
                        if (e.HResult == -2146233088)
                            return null;
                        else
                            throw;
                    }
                }
            }
        }

        private void CacheFeaturizedImagesToDisk(IDataView input, string labelColumnName, string imageColumnName,
            ImageProcessor imageProcessor, string inputTensorName, string outputTensorName, string cacheFilePath,
            ImageClassificationMetrics.Dataset dataset, Action<ImageClassificationMetrics> metricsCallback,
            float? validationFraction = null)
        {
            var labelColumn = input.Schema[labelColumnName];

            if (labelColumn.Type.RawType != typeof(uint))
            {
                throw Host.ExceptSchemaMismatch(nameof(labelColumn), "Label",
                    labelColumnName, typeof(uint).ToString(),
                    labelColumn.Type.RawType.ToString());
            }

            var imageColumn = input.Schema[imageColumnName];
            Runner runner = new Runner(_session, new[] { inputTensorName }, new[] { outputTensorName });
            List<(long, float[])> featurizedImages = new List<(long, float[])>();
            using (var cursor = input.GetRowCursor(
                input.Schema.Where(c => c.Index == labelColumn.Index || c.Index == imageColumn.Index)))
            {
                var labelGetter = cursor.GetGetter<uint>(labelColumn);
                var imageGetter = cursor.GetGetter<VBuffer<byte>>(imageColumn);
                uint label = uint.MaxValue;
                VBuffer<byte> image = default;
                ImageClassificationMetrics metrics = new ImageClassificationMetrics();
                metrics.Bottleneck = new BottleneckMetrics();
                metrics.Bottleneck.DatasetUsed = dataset;
                while (cursor.MoveNext())
                {
                    CheckAlive();
                    labelGetter(ref label);
                    imageGetter(ref image);
                    if (image.Length <= 0)
                        continue; //Empty Image

                    var imageTensor = imageProcessor.ProcessImage(image);
                    if (imageTensor != null)
                    {
                        runner.AddInput(imageTensor, 0);
                        var featurizedImage = runner.Run()[0];
                        featurizedImages.Add((label - 1, featurizedImage.ToArray<float>()));
                        featurizedImage.Dispose();
                        imageTensor.Dispose();
                        metrics.Bottleneck.Index++;
                        metricsCallback?.Invoke(metrics);
                    }
                }

                featurizedImages = featurizedImages.OrderBy(x => Host.Rand.Next(0, metrics.Bottleneck.Index)).ToList();
                int featureLength = featurizedImages.Count > 0 ? featurizedImages[0].Item2.Length : 0;
                int validationSetCount = 0;
                if (validationFraction.HasValue)
                {
                    Contracts.Assert(validationFraction >= 0 && validationFraction <= 1);

                    validationSetCount = (int)(metrics.Bottleneck.Index * validationFraction);
                    CreateFeaturizedCacheFile(
                        Path.Combine(_options.WorkspacePath, _options.ValidationSetBottleneckCachedValuesFileName),
                        validationSetCount, featureLength, featurizedImages.Take(validationSetCount));
                }

                CreateFeaturizedCacheFile(cacheFilePath, metrics.Bottleneck.Index - validationSetCount, featureLength,
                    featurizedImages.Skip(validationSetCount));
            }
        }

        private void CreateFeaturizedCacheFile(string cacheFilePath, int examples, int featureLength,
            IEnumerable<(long, float[])> featurizedImages)
        {
            Contracts.Assert(examples == featurizedImages.Count());
            Contracts.Assert(featurizedImages.All(x => x.Item2.Length == featureLength));

            using Stream featuresWriter = File.Open(cacheFilePath + "_features.bin", FileMode.Create);
            using Stream labelWriter = File.Open(cacheFilePath + "_labels.bin", FileMode.Create);
            using TextWriter writer = File.CreateText(cacheFilePath);

            featuresWriter.Write(BitConverter.GetBytes(examples), 0, sizeof(int));
            featuresWriter.Write(BitConverter.GetBytes(featureLength), 0, sizeof(int));
            long[] labels = new long[1];
            var labelsSpan = MemoryMarshal.Cast<long, byte>(labels);

            foreach (var row in featurizedImages)
            {
                CheckAlive();
                writer.WriteLine(row.Item1 + "," + string.Join(",", row.Item2));
                labels[0] = row.Item1;
                for (int index = 0; index < sizeof(long); index++)
                {
                    labelWriter.WriteByte(labelsSpan[index]);
                }

                var featureSpan = MemoryMarshal.Cast<float, byte>(row.Item2);
                for (int index = 0; index < featureLength * sizeof(float); index++)
                {
                    featuresWriter.WriteByte(featureSpan[index]);
                }
            }
        }

        private void TrainAndEvaluateClassificationLayer(string trainBottleneckFilePath,
            string validationSetBottleneckFilePath)
        {
            Contracts.Assert(validationSetBottleneckFilePath == null ||
                (File.Exists(validationSetBottleneckFilePath + "_labels.bin") &&
                    File.Exists(validationSetBottleneckFilePath + "_features.bin")));

            Contracts.Assert(trainBottleneckFilePath != null &&
                File.Exists(trainBottleneckFilePath + "_labels.bin") &&
                    File.Exists(trainBottleneckFilePath + "_features.bin"));

            bool validationNeeded = validationSetBottleneckFilePath != null;

            Contracts.Assert(_options.EarlyStoppingCriteria == null || validationNeeded);

            using (Stream trainSetLabelReader = File.Open(trainBottleneckFilePath + "_labels.bin", FileMode.Open))
            using (Stream trainSetFeatureReader = File.Open(trainBottleneckFilePath + "_features.bin", FileMode.Open))
            {
                Stream validationSetLabelReader = validationNeeded ?
                    File.Open(validationSetBottleneckFilePath + "_labels.bin", FileMode.Open) : null;

                Stream validationSetFeatureReader = validationNeeded ?
                    File.Open(validationSetBottleneckFilePath + "_features.bin", FileMode.Open) : null;

                int batchSize = _options.BatchSize;
                int epochs = _options.Epoch;
                float learningRate = _options.LearningRate;
                Action<ImageClassificationMetrics> statisticsCallback = _options.MetricsCallback;
                Runner runner = null;
                Runner validationEvalRunner = null;
                List<string> runnerInputTensorNames = new List<string>();
                List<string> runnerOutputTensorNames = new List<string>();
                runnerInputTensorNames.Add(_bottleneckInput.name);
                runnerInputTensorNames.Add(_labelTensor.name);
                if (_options.LearningRateScheduler != null)
                    runnerInputTensorNames.Add(_learningRateInput.name);

                if (statisticsCallback != null && _options.TestOnTrainSet)
                {
                    runnerOutputTensorNames.Add(_evaluationStep.name);
                    runnerOutputTensorNames.Add(_crossEntropy.name);
                }

                if (validationNeeded)
                {
                    validationEvalRunner = new Runner(_session, new[] { _bottleneckInput.name, _labelTensor.name },
                        new[] { _evaluationStep.name, _crossEntropy.name });
                }

                runner = new Runner(_session, runnerInputTensorNames.ToArray(),
                    runnerOutputTensorNames.Count() > 0 ? runnerOutputTensorNames.ToArray() : null,
                    new[] { _trainStep.name });

                Saver trainSaver = null;
                FileWriter trainWriter = null;
                Tensor merged = tf.summary.merge_all();
                trainWriter = tf.summary.FileWriter(Path.Combine(_options.WorkspacePath, "train"),
                    _session.graph);

                trainSaver = tf.train.Saver();
                trainSaver.save(_session, _checkpointPath);
                ImageClassificationMetrics metrics = new ImageClassificationMetrics();
                metrics.Train = new TrainMetrics();
                float accuracy = 0;
                float crossentropy = 0;
                var labelTensorShape = _labelTensor.TensorShape.dims.Select(x => (long)x).ToArray();
                var featureTensorShape = _bottleneckInput.TensorShape.dims.Select(x => (long)x).ToArray();
                byte[] buffer = new byte[sizeof(int)];
                trainSetFeatureReader.Read(buffer, 0, 4);
                int trainingExamples = BitConverter.ToInt32(buffer, 0);
                trainSetFeatureReader.Read(buffer, 0, 4);
                int featureFileRecordSize = sizeof(float) * BitConverter.ToInt32(buffer, 0);
                const int featureFileStartOffset = sizeof(int) * 2;
                var labelBufferSizeInBytes = sizeof(long) * batchSize;
                var featureBufferSizeInBytes = featureFileRecordSize * batchSize;
                byte[] featuresBuffer = new byte[featureBufferSizeInBytes];
                byte[] labelBuffer = new byte[labelBufferSizeInBytes];
                var featureBufferHandle = GCHandle.Alloc(featuresBuffer, GCHandleType.Pinned);
                IntPtr featureBufferPtr = featureBufferHandle.AddrOfPinnedObject();
                var labelBufferHandle = GCHandle.Alloc(labelBuffer, GCHandleType.Pinned);
                IntPtr labelBufferPtr = labelBufferHandle.AddrOfPinnedObject();
                DnnTrainState trainState = new DnnTrainState
                {
                    BatchSize = _options.BatchSize,
                    BatchesPerEpoch = trainingExamples / _options.BatchSize
                };

                for (int epoch = 0; epoch < epochs; epoch += 1)
                {
                    CheckAlive();
                    // Train.
                    TrainAndEvaluateClassificationLayerCore(epoch, learningRate, featureFileStartOffset,
                        metrics, labelTensorShape, featureTensorShape, batchSize,
                        trainSetLabelReader, trainSetFeatureReader, labelBuffer, featuresBuffer,
                        labelBufferSizeInBytes, featureBufferSizeInBytes, featureFileRecordSize,
                        _options.LearningRateScheduler, trainState, runner, featureBufferPtr, labelBufferPtr,
                        (outputTensors, metrics) =>
                            {
                                if (_options.TestOnTrainSet && statisticsCallback != null)
                                {
                                    outputTensors[0].ToScalar(ref accuracy);
                                    outputTensors[1].ToScalar(ref crossentropy);
                                    metrics.Train.Accuracy += accuracy;
                                    metrics.Train.CrossEntropy += crossentropy;
                                    outputTensors[0].Dispose();
                                    outputTensors[1].Dispose();
                                }
                            });

                    if (_options.TestOnTrainSet && statisticsCallback != null)
                    {
                        metrics.Train.Epoch = epoch;
                        metrics.Train.Accuracy /= metrics.Train.BatchProcessedCount;
                        metrics.Train.CrossEntropy /= metrics.Train.BatchProcessedCount;
                        metrics.Train.DatasetUsed = ImageClassificationMetrics.Dataset.Train;
                        statisticsCallback(metrics);
                    }

                    if (!validationNeeded)
                        continue;

                    // Evaluate.
                    TrainAndEvaluateClassificationLayerCore(epoch, learningRate, featureFileStartOffset,
                        metrics, labelTensorShape, featureTensorShape, batchSize,
                        validationSetLabelReader, validationSetFeatureReader, labelBuffer, featuresBuffer,
                        labelBufferSizeInBytes, featureBufferSizeInBytes, featureFileRecordSize, null,
                        trainState, validationEvalRunner, featureBufferPtr, labelBufferPtr,
                        (outputTensors, metrics) =>
                            {
                                outputTensors[0].ToScalar(ref accuracy);
                                outputTensors[1].ToScalar(ref crossentropy);
                                metrics.Train.Accuracy += accuracy;
                                metrics.Train.CrossEntropy += crossentropy;
                                outputTensors[0].Dispose();
                                outputTensors[1].Dispose();
                            });

                    if (statisticsCallback != null)
                    {
                        metrics.Train.Epoch = epoch;
                        metrics.Train.Accuracy /= metrics.Train.BatchProcessedCount;
                        metrics.Train.CrossEntropy /= metrics.Train.BatchProcessedCount;
                        metrics.Train.DatasetUsed = ImageClassificationMetrics.Dataset.Validation;
                        statisticsCallback(metrics);
                    }

                    //Early stopping check
                    if (_options.EarlyStoppingCriteria != null)
                    {
                        if (_options.EarlyStoppingCriteria.ShouldStop(metrics.Train))
                            break;
                    }
                }

                trainSaver.save(_session, _checkpointPath);
                validationSetLabelReader?.Dispose();
                validationSetFeatureReader?.Dispose();
                featureBufferHandle.Free();
                labelBufferHandle.Free();
            }

            UpdateTransferLearningModelOnDisk(_classCount);
            TryCleanupTemporaryWorkspace();
        }

        private void TrainAndEvaluateClassificationLayerCore(int epoch, float learningRate,
            int featureFileStartOffset, ImageClassificationMetrics metrics,
            long[] labelTensorShape, long[] featureTensorShape, int batchSize, Stream trainSetLabelReader,
            Stream trainSetFeatureReader, byte[] labelBufferBytes, byte[] featuresBufferBytes,
            int labelBufferSizeInBytes, int featureBufferSizeInBytes, int featureFileRecordSize,
            LearningRateScheduler learningRateScheduler, DnnTrainState trainState, Runner runner,
            IntPtr featureBufferPtr, IntPtr labelBufferPtr, Action<Tensor[], ImageClassificationMetrics> metricsAggregator)
        {
            int labelFileBytesRead;
            int featuresFileBytesRead;
            labelTensorShape[0] = featureTensorShape[0] = batchSize;
            metrics.Train.Accuracy = 0;
            metrics.Train.CrossEntropy = 0;
            metrics.Train.BatchProcessedCount = 0;
            metrics.Train.LearningRate = learningRate;
            trainState.CurrentBatchIndex = 0;
            trainState.CurrentEpoch = epoch;
            trainSetLabelReader.Seek(0, SeekOrigin.Begin);
            trainSetFeatureReader.Seek(featureFileStartOffset, SeekOrigin.Begin);
            labelTensorShape[0] = featureTensorShape[0] = batchSize;

            while ((labelFileBytesRead = trainSetLabelReader.TryReadBlock(labelBufferBytes, 0, labelBufferSizeInBytes)) > 0 &&
                (featuresFileBytesRead = trainSetFeatureReader.TryReadBlock(featuresBufferBytes, 0, featureBufferSizeInBytes)) > 0)
            {
                Contracts.Assert(labelFileBytesRead <= labelBufferSizeInBytes);
                Contracts.Assert(featuresFileBytesRead <= featureBufferSizeInBytes);
                Contracts.Assert(labelFileBytesRead % sizeof(long) == 0);
                Contracts.Assert(featuresFileBytesRead % featureFileRecordSize == 0);
                Contracts.Assert(labelFileBytesRead / sizeof(long) == featuresFileBytesRead / featureFileRecordSize);

                if (labelFileBytesRead < labelBufferSizeInBytes)
                {
                    featureTensorShape[0] = featuresFileBytesRead / featureFileRecordSize;
                    labelTensorShape[0] = labelFileBytesRead / sizeof(long);
                }

                Contracts.Assert(featureTensorShape[0] <= featuresBufferBytes.Length / featureFileRecordSize);
                Contracts.Assert(labelTensorShape[0] <= labelBufferBytes.Length / sizeof(long));

                if (learningRateScheduler != null)
                {
                    // Add learning rate as a placeholder only when learning rate scheduling is used.
                    metrics.Train.LearningRate = learningRateScheduler.GetLearningRate(trainState);
                    runner.AddInput(new Tensor(metrics.Train.LearningRate, TF_DataType.TF_FLOAT), 2);
                }

                var outputTensors = runner.AddInput(new Tensor(featureBufferPtr, featureTensorShape, TF_DataType.TF_FLOAT, featuresFileBytesRead), 0)
                                    .AddInput(new Tensor(labelBufferPtr, labelTensorShape, TF_DataType.TF_INT64, labelFileBytesRead), 1)
                                    .Run();

                metrics.Train.BatchProcessedCount += 1;
                metricsAggregator(outputTensors, metrics);
                trainState.CurrentBatchIndex += 1;
            }
        }

        private void CheckAlive()
        {
            try
            {
                Host.CheckAlive();
            }
            catch (OperationCanceledException)
            {
                TryCleanupTemporaryWorkspace();
                throw;
            }
        }

        private void TryCleanupTemporaryWorkspace()
        {
            if (_cleanupWorkspace && Directory.Exists(_options.WorkspacePath))
            {
                try
                {
                    Directory.Delete(_options.WorkspacePath, true);
                }
                catch (Exception)
                {
                    //We do not want to stop pipeline due to failed cleanup.
                }
            }
        }

        private (Session, Tensor, Tensor, Tensor) BuildEvaluationSession(int classCount)
        {
            var evalGraph = LoadMetaGraph(Path.Combine(_resourcePath, ModelFileName[_options.Arch]));
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

            sess.graph.Dispose();
            sess.Dispose();
        }

        private void VariableSummaries(ResourceVariable var)
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
                ResourceVariable layerWeights = null;
                tf_with(tf.name_scope("weights"), delegate
                {
                    var initialValue = tf.truncated_normal(new int[] { bottleneck_tensor_size, classCount },
                        stddev: 0.001f);

                    layerWeights = tf.Variable(initialValue, name: "final_weights");
                    VariableSummaries(layerWeights);
                });

                ResourceVariable layerBiases = null;
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

        private TensorFlowSessionWrapper LoadTensorFlowSessionFromMetaGraph(IHostEnvironment env, Architecture arch)
        {
            var modelFileName = ModelFileName[arch];
            var modelFilePath = Path.Combine(_resourcePath, modelFileName);
            int timeout = 10 * 60 * 1000;
            DownloadIfNeeded(env, @"meta\" + modelFileName, _resourcePath, modelFileName, timeout);
            return new TensorFlowSessionWrapper(GetSession(env, modelFilePath, true), modelFilePath);
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
    public sealed class ImageClassificationModelParameters : ModelParametersBase<VBuffer<float>>, IValueMapper, IDisposable
    {
        private bool _isDisposed;

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

        internal static ImageClassificationModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
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
            using (var status = new Status())
            using (var buffer = _session.graph.ToGraphDef(status))
            {
                ctx.SaveBinaryStream("TFModel", w =>
                {
                    w.WriteByteArray(buffer.DangerousMemoryBlock.ToArray());
                });
                status.Check(true);
            }
        }

        private class Classifier
        {
            private readonly Runner _runner;
            private readonly ImageClassificationTrainer.ImageProcessor _imageProcessor;

            public Classifier(ImageClassificationModelParameters model)
            {
                _runner = new Runner(model._session, new[] { model._graphInputTensor }, new[] { model._graphOutputTensor });
                _imageProcessor = new ImageClassificationTrainer.ImageProcessor(model._session,
                    model._imagePreprocessorTensorInput, model._imagePreprocessorTensorOutput);
            }

            public void Score(in VBuffer<byte> image, Span<float> classProbabilities)
            {
                var processedTensor = _imageProcessor.ProcessImage(image);
                if (processedTensor != null)
                {
                    var outputTensor = _runner.AddInput(processedTensor, 0).Run();
                    outputTensor[0].CopyTo(classProbabilities);
                    outputTensor[0].Dispose();
                    processedTensor.Dispose();
                }
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

        public void Dispose()
        {
            if (_isDisposed)
                return;

            if (_session?.graph != IntPtr.Zero)
            {
                _session.graph.Dispose();
            }

            if (_session != null && _session != IntPtr.Zero)
            {
                _session.close();
            }

            _isDisposed = true;
        }
    }
}
