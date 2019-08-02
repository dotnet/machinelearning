// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Dnn;
using static Microsoft.ML.Transforms.DnnEstimator;

namespace Microsoft.ML
{
    /// <include file='doc.xml' path='doc/members/member[@name="DnnTransformer"]/*' />
    public static class DnnCatalog
    {

        /// <summary>
        /// Retrain the dnn model on new data.
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="inputColumnNames"> The names of the model inputs.</param>
        /// <param name="outputColumnNames">The names of the requested model outputs.</param>
        /// <param name="labelColumnName">Name of the label column.</param>
        /// <param name="tensorFlowLabel">Name of the node in TensorFlow graph that is used as label during training in TensorFlow.
        /// The value of <paramref name="labelColumnName"/> from <see cref="IDataView"/> is fed to this node.</param>
        /// <param name="optimizationOperation">The name of the optimization operation in the TensorFlow graph.</param>
        /// <param name="modelPath">Path to model file to retrain.</param>
        /// <param name="epoch">Number of training iterations.</param>
        /// <param name="batchSize">Number of samples to use for mini-batch training.</param>
        /// <param name="lossOperation">The name of the operation in the TensorFlow graph to compute training loss (Optional).</param>
        /// <param name="metricOperation">The name of the operation in the TensorFlow graph to compute performance metric during training (Optional).</param>
        /// <param name="learningRateOperation">The name of the operation in the TensorFlow graph which sets optimizer learning rate (Optional).</param>
        /// <param name="learningRate">Learning rate to use during optimization (Optional).</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        /// <param name="dnnFramework"></param>
        /// <remarks>
        /// The support for retraining is under preview.
        /// </remarks>
        public static DnnEstimator RetrainDnnModel(
            this ModelOperationsCatalog catalog,
            string[] outputColumnNames,
            string[] inputColumnNames,
            string labelColumnName,
            string tensorFlowLabel,
            string optimizationOperation,
            string modelPath,
            int epoch = 10,
            int batchSize = 20,
            string lossOperation = null,
            string metricOperation = null,
            string learningRateOperation = null,
            float learningRate = 0.01f,
            bool addBatchDimensionInput = false,
            DnnFramework dnnFramework = DnnFramework.Tensorflow)
        {
            var options = new Options()
            {
                ModelLocation = modelPath,
                InputColumns = inputColumnNames,
                OutputColumns = outputColumnNames,
                LabelColumn = labelColumnName,
                TensorFlowLabel = tensorFlowLabel,
                OptimizationOperation = optimizationOperation,
                LossOperation = lossOperation,
                MetricOperation = metricOperation,
                Epoch = epoch,
                LearningRateOperation = learningRateOperation,
                LearningRate = learningRate,
                BatchSize = batchSize,
                AddBatchDimensionInputs = addBatchDimensionInput,
                ReTrain = true
            };

            var env = CatalogUtils.GetEnvironment(catalog);
            return new DnnEstimator(env, options, DnnUtils.LoadDnnModel(env, modelPath, true));
        }

        /// <summary>
        /// Performs image classification using transfer learning.
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="featuresColumnName"></param>
        /// <param name="labelColumnName"></param>
        /// <param name="outputGraphPath"></param>
        /// <param name="scoreColumnName"></param>
        /// <param name="predictedLabelColumnName"></param>
        /// <param name="checkpointName"></param>
        /// <param name="arch"></param>
        /// <param name="dnnFramework"></param>
        /// <param name="epoch"></param>
        /// <param name="batchSize"></param>
        /// <param name="learningRate"></param>
        /// <param name="addBatchDimensionInput"></param>
        /// <remarks>
        /// The support for image classification is under preview.
        /// </remarks>
        public static DnnEstimator ImageClassification(
            this ModelOperationsCatalog catalog,
            string featuresColumnName,
            string labelColumnName,
            string outputGraphPath = null,
            string scoreColumnName = "Scores",
            string predictedLabelColumnName = "PredictedLabel",
            string checkpointName = "_retrain_checkpoint",
            Architecture arch = Architecture.ResnetV2101,
            DnnFramework dnnFramework = DnnFramework.Tensorflow,
            int epoch = 10,
            int batchSize = 20,
            float learningRate = 0.01f,
            bool addBatchDimensionInput = false)
        {
            var options = new Options()
            {
                ModelLocation = arch == Architecture.ResnetV2101 ? @"DnnImageModels\Resnet101V2Tensorflow\resnet_v2_101_299.meta" : "",
                InputColumns = new[] { featuresColumnName },
                OutputColumns = new[] { scoreColumnName, predictedLabelColumnName },
                LabelColumn = labelColumnName,
                TensorFlowLabel = labelColumnName,
                Epoch = epoch,
                LearningRate = learningRate,
                BatchSize = batchSize,
                AddBatchDimensionInputs = addBatchDimensionInput,
                TransferLearning = true,
                ScoreColumnName = scoreColumnName,
                PredictedLabelColumnName = predictedLabelColumnName,
                CheckpointName = checkpointName
            };

            var env = CatalogUtils.GetEnvironment(catalog);
            return new DnnEstimator(env, options, DnnUtils.LoadDnnModel(env, options.ModelLocation, true));
        }
    }
}
