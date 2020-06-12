// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using static Microsoft.ML.TensorFlow.TensorFlowUtils;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for <see cref="T:Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers" /> to create instances of ImageClassification trainer components.
    /// </summary>
    /// <remarks>
    /// This requires additional nuget dependencies to link against Tensorflow native dlls. See <see cref="T:Microsoft.ML.Vision.ImageClassificationTrainer"/> for more information.
    /// </remarks>
    public static class VisionCatalog
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
            return new DnnRetrainEstimator(env, options, LoadDnnModel(env, modelPath, true));
        }

        /// <summary>
        /// Create <see cref="Microsoft.ML.Vision.ImageClassificationTrainer"/> using advanced options, which trains a Deep Neural Network(DNN) to classify images.
        /// </summary>
        /// <param name="catalog">Catalog</param>
        /// <param name="options">An <see cref="ImageClassificationTrainer.Options"/> object specifying advanced
        /// options for <see cref="ImageClassificationTrainer"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ImageClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/ImageClassification/ResnetV2101TransferLearningTrainTestSplit.cs)]
        /// ]]></format>
        /// </example>

        public static ImageClassificationTrainer ImageClassification(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            ImageClassificationTrainer.Options options) =>
                new ImageClassificationTrainer(CatalogUtils.GetEnvironment(catalog), options);

        /// <summary>
        /// Create <see cref="Microsoft.ML.Vision.ImageClassificationTrainer"/>, which trains a Deep Neural Network(DNN) to classify images.
        /// </summary>
        /// <param name="catalog">Catalog</param>
        /// <param name="labelColumnName">The name of the labels column. The default for this parameter is "label".</param>
        /// <param name="featureColumnName">The name of the input features column. The default for this parameter is "Features".</param>
        /// <param name="scoreColumnName">The name of the output score column. The default for this parameter is "Score"</param>
        /// <param name="predictedLabelColumnName">The name of the output predicted label columns. The default for this parameter is "PredictedLabel"</param>
        /// <param name="validationSet">The validation set used while training to improve model quality. The default for this parameter is null.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[ImageClassification](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/MulticlassClassification/ImageClassification/ImageClassificationDefault.cs)]
        ///  ]]></format>
        /// </example>

        public static ImageClassificationTrainer ImageClassification(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string scoreColumnName = DefaultColumnNames.Score,
            string predictedLabelColumnName = DefaultColumnNames.PredictedLabel,
            IDataView validationSet = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            return new ImageClassificationTrainer(CatalogUtils.GetEnvironment(catalog), labelColumnName,
                featureColumnName, scoreColumnName, predictedLabelColumnName, validationSet);
        }
    }
}
