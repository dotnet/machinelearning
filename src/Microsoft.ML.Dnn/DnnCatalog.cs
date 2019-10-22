﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
                FeaturesColumnName = featuresColumnName,
                LabelColumnName = labelColumnName,
                ScoreColumnName = scoreColumnName,
                PredictedLabelColumnName = predictedLabelColumnName,
                ValidationSet = validationSet,
            };

            return ImageClassification(catalog, options);
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
        /// <param name="options">An <see cref="Options"/> object specifying advanced options for <see cref="ImageClassificationEstimator"/>.</param>
        public static ImageClassificationEstimator ImageClassification(
            this ModelOperationsCatalog catalog, Options options)
        {
            options.EarlyStoppingCriteria = options.DisableEarlyStopping ? null : options.EarlyStoppingCriteria ?? new EarlyStopping();

            var env = CatalogUtils.GetEnvironment(catalog);
            return new ImageClassificationEstimator(env, options, DnnUtils.LoadDnnModel(env, options.Arch, true));
        }
    }
}
