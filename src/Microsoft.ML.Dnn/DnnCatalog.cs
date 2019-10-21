// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.IO.Compression;
using System.Net;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Dnn;
using static Microsoft.ML.Transforms.ImageClassificationEstimator;

namespace Microsoft.ML
{
    /// <include file='doc.xml' path='doc/members/member[@name="DnnRetrainTransformer"]/*' />
    public static class DnnCatalog
    {

        /// <summary>
        /// Retrain the dnn model on new data.
        /// usage of this API requires additional NuGet dependencies on TensorFlow redist, see linked document for more information.
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!include[io](~/../docs/samples/docs/api-reference/tensorflow-usage.md)]
        /// ]]>
        /// </format>
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="inputColumnNames"> The names of the model inputs.</param>
        /// <param name="outputColumnNames">The names of the requested model outputs.</param>
        /// <param name="labelColumnName">Name of the label column.</param>
        /// <param name="dnnLabel">Name of the node in DNN graph that is used as label during training in Dnn.
        /// The value of <paramref name="labelColumnName"/> from <see cref="IDataView"/> is fed to this node.</param>
        /// <param name="optimizationOperation">The name of the optimization operation in the Dnn graph.</param>
        /// <param name="modelPath">Path to model file to retrain.</param>
        /// <param name="epoch">Number of training iterations.</param>
        /// <param name="batchSize">Number of samples to use for mini-batch training.</param>
        /// <param name="lossOperation">The name of the operation in the Dnn graph to compute training loss (Optional).</param>
        /// <param name="metricOperation">The name of the operation in the Dnn graph to compute performance metric during training (Optional).</param>
        /// <param name="learningRateOperation">The name of the operation in the Dnn graph which sets optimizer learning rate (Optional).</param>
        /// <param name="learningRate">Learning rate to use during optimization (Optional).</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        /// <remarks>
        /// The support for retraining is under preview.
        /// </remarks>
        internal static DnnRetrainEstimator RetrainDnnModel(
            this ModelOperationsCatalog catalog,
            string[] outputColumnNames,
            string[] inputColumnNames,
            string labelColumnName,
            string dnnLabel,
            string optimizationOperation,
            string modelPath,
            int epoch = 10,
            int batchSize = 20,
            string lossOperation = null,
            string metricOperation = null,
            string learningRateOperation = null,
            float learningRate = 0.01f,
            bool addBatchDimensionInput = false)
        {
            var options = new DnnRetrainEstimator.Options()
            {
                ModelLocation = modelPath,
                InputColumns = inputColumnNames,
                OutputColumns = outputColumnNames,
                LabelColumn = labelColumnName,
                TensorFlowLabel = dnnLabel,
                OptimizationOperation = optimizationOperation,
                LossOperation = lossOperation,
                MetricOperation = metricOperation,
                Epoch = epoch,
                LearningRateOperation = learningRateOperation,
                LearningRate = learningRate,
                BatchSize = batchSize,
                AddBatchDimensionInputs = addBatchDimensionInput
            };

            var env = CatalogUtils.GetEnvironment(catalog);
            return new DnnRetrainEstimator(env, options, DnnUtils.LoadDnnModel(env, modelPath, true));
        }

        /// <summary>
        /// The options for the <see cref="ImageClassificationTransformer"/>.
        /// </summary>
        public sealed class Options
        {
            /// <summary>
            /// The names of the model inputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string FeaturesColumnName;

            /// <summary>
            /// The name of the label column in <see cref="IDataView"/> that will be mapped to label node in TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Training labels.", ShortName = "label", SortOrder = 4)]
            public string LabelColumnName;

            /// <summary>
            /// The names of the requested model outputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the outputs", ShortName = "outputs", SortOrder = 2)]
            public string ScoreColumnName = "Score";

            /// <summary>
            /// Name of the tensor that will contain the predicted label from output scores of the last layer when transfer learning is done.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Argmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string PredictedLabelColumnName = "PredictedLabel";

            /// <summary>
            /// Validation set.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Validation set.", SortOrder = 15)]
            public IDataView ValidationSet = null;

            /// <summary>
            /// Specifies the model architecture to be used in the case of image classification training using transfer learning.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Model architecture to be used in transfer learning for image classification.", SortOrder = 15)]
            public Architecture Arch = Architecture.InceptionV3;

            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of samples to use for mini-batch training.", SortOrder = 9)]
            public int BatchSize = 10;

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
            /// Early Stopping technique to stop training when accuracy stops improving.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Early Stopping technique to stop training when accuracy stops improving.", SortOrder = 15)]
            public bool DisableEarlyStopping = false;

            /// <summary>
            /// Early Stopping technique to stop training when accuracy stops improving.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Early Stopping technique to stop training when accuracy stops improving.", SortOrder = 15)]
            public EarlyStopping EarlyStoppingCriteria = null;

            /// <summary>
            /// Callback to report statistics on accuracy/cross entropy during training phase.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Callback to report metrics during training and validation phase.", SortOrder = 15)]
            public ImageClassificationMetricsCallback MetricsCallback = null;

            /// <summary>
            /// Final model and checkpoint files/folder prefix for storing graph files.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Final model and checkpoint files/folder prefix for storing graph files.", SortOrder = 15)]
            public string FinalModelPrefix = "custom_retrained_model_based_on_";

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
            /// Indicates the file path to store trainset bottleneck values for caching.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the file path to store trainset bottleneck values for caching.", SortOrder = 15)]
            public string TrainSetBottleneckCachedValuesFilePath = "trainSetBottleneckFile.csv";

            /// <summary>
            /// Indicates the file path to store validationset bottleneck values for caching.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates the file path to store validationset bottleneck values for caching.", SortOrder = 15)]
            public string ValidationSetBottleneckCachedValuesFilePath = "validationSetBottleneckFile.csv";
        }

        /// <summary>
        /// Performs image classification using transfer learning.
        /// usage of this API requires additional NuGet dependencies on TensorFlow redist, see linked document for more information.
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!include[io](~/../docs/samples/docs/api-reference/tensorflow-usage.md)]
        /// ]]>
        /// </format>
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="featuresColumnName">The name of the input features column.</param>
        /// <param name="labelColumnName">The name of the labels column.</param>
        /// <param name="scoreColumnName">The name of the output score column.</param>
        /// <param name="predictedLabelColumnName">The name of the output predicted label columns.</param>
        /// <param name="validationSet">Validation set.</param>
        /// <remarks>
        /// The support for image classification is under preview.
        /// </remarks>
        public static ImageClassificationEstimator ImageClassification(
            this ModelOperationsCatalog catalog,
            string featuresColumnName,
            string labelColumnName,
            string scoreColumnName = "Score",
            string predictedLabelColumnName = "PredictedLabel",
            IDataView validationSet = null
            )
        {
            var options = new ImageClassificationEstimator.Options()
            {
                InputColumns = new[] { featuresColumnName },
                OutputColumns = new[] { scoreColumnName, predictedLabelColumnName },
                LabelColumn = labelColumnName,
                TensorFlowLabel = labelColumnName,
                Epoch = 100,
                EarlyStoppingCriteria = new EarlyStopping(),
                ScoreColumnName = scoreColumnName,
                PredictedLabelColumnName = predictedLabelColumnName,
                FinalModelPrefix = "custom_retrained_model_based_on_",
                Arch = Architecture.InceptionV3,
                MetricsCallback = null,
                ValidationSet = validationSet,
                TestOnTrainSet = true,
                TrainSetBottleneckCachedValuesFilePath = "trainSetBottleneckFile.csv",
                ValidationSetBottleneckCachedValuesFilePath = "validationSetBottleneckFile.csv",
                ReuseTrainSetBottleneckCachedValues = false,
                ReuseValidationSetBottleneckCachedValues = false
            };

            var env = CatalogUtils.GetEnvironment(catalog);
            return new ImageClassificationEstimator(env, options, DnnUtils.LoadDnnModel(env, options.Arch, true));
        }

        public static ImageClassificationEstimator ImageClassification(
            this ModelOperationsCatalog catalog, Options options)
        {
            var estimatorOptions = new ImageClassificationEstimator.Options()
            {
                InputColumns = new[] { options.FeaturesColumnName },
                OutputColumns = new[] { options.ScoreColumnName, options.PredictedLabelColumnName },
                LabelColumn = options.LabelColumnName,
                TensorFlowLabel = options.LabelColumnName,
                Epoch = options.Epoch,
                LearningRate = options.LearningRate,
                BatchSize = options.BatchSize,
                EarlyStoppingCriteria = options.DisableEarlyStopping ? null : options.EarlyStoppingCriteria == null ? new EarlyStopping() : options.EarlyStoppingCriteria,
                ScoreColumnName = options.ScoreColumnName,
                PredictedLabelColumnName = options.PredictedLabelColumnName,
                FinalModelPrefix = options.FinalModelPrefix,
                Arch = options.Arch,
                MetricsCallback = options.MetricsCallback,
                ValidationSet = options.ValidationSet,
                TestOnTrainSet = options.TestOnTrainSet,
                TrainSetBottleneckCachedValuesFilePath = options.TrainSetBottleneckCachedValuesFilePath,
                ValidationSetBottleneckCachedValuesFilePath = options.ValidationSetBottleneckCachedValuesFilePath,
                ReuseTrainSetBottleneckCachedValues = options.ReuseTrainSetBottleneckCachedValues,
                ReuseValidationSetBottleneckCachedValues = options.ReuseValidationSetBottleneckCachedValues
            };

            var env = CatalogUtils.GetEnvironment(catalog);
            return new ImageClassificationEstimator(env, estimatorOptions, DnnUtils.LoadDnnModel(env, options.Arch, true));
        }
    }
}
