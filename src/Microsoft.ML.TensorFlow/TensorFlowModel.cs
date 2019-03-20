// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TensorFlow;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This class holds the information related to TensorFlow model and session.
    /// It provides some convenient methods to query model schema as well as
    /// creation of <see cref="TensorFlowEstimator"/> object.
    /// </summary>
    public sealed class TensorFlowModel
    {
        internal TFSession Session { get; }
        internal string ModelPath { get; }

        private readonly IHostEnvironment _env;

        /// <summary>
        /// Instantiates <see cref="TensorFlowModel"/>.
        /// </summary>
        /// <param name="env">An <see cref="IHostEnvironment"/> object.</param>
        /// <param name="session">TensorFlow session object.</param>
        /// <param name="modelLocation">Location of the model from where <paramref name="session"/> was loaded.</param>
        internal TensorFlowModel(IHostEnvironment env, TFSession session, string modelLocation)
        {
            Session = session;
            ModelPath = modelLocation;
            _env = env;
        }

        /// <summary>
        /// Get <see cref="DataViewSchema"/> for complete model. Every node in the TensorFlow model will be included in the <see cref="DataViewSchema"/> object.
        /// </summary>
        public DataViewSchema GetModelSchema()
        {
            return TensorFlowUtils.GetModelSchema(_env, Session.Graph);
        }

        /// <summary>
        /// Get <see cref="DataViewSchema"/> for only those nodes which are marked "Placeholder" in the TensorFlow model.
        /// This method is convenient for exploring the model input(s) in case TensorFlow graph is very large.
        /// </summary>
        public DataViewSchema GetInputSchema()
        {
            return TensorFlowUtils.GetModelSchema(_env, Session.Graph, "Placeholder");
        }

        /// <summary>
        /// Scores a dataset using a pre-traiend <a href="https://www.tensorflow.org/">TensorFlow</a> model.
        /// </summary>
        /// <param name="inputColumnName"> The name of the model input.</param>
        /// <param name="outputColumnName">The name of the requested model output.</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlowTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public TensorFlowEstimator ScoreTensorFlowModel(string outputColumnName, string inputColumnName, bool addBatchDimensionInput = false)
            => new TensorFlowEstimator(_env, new[] { outputColumnName }, new[] { inputColumnName }, this, addBatchDimensionInput);

        /// <summary>
        /// Scores a dataset using a pre-traiend TensorFlow model.
        /// </summary>
        /// <param name="inputColumnNames"> The names of the model inputs.</param>
        /// <param name="outputColumnNames">The names of the requested model outputs.</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlow/ImageClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public TensorFlowEstimator ScoreTensorFlowModel(string[] outputColumnNames, string[] inputColumnNames, bool addBatchDimensionInput = false)
            => new TensorFlowEstimator(_env, outputColumnNames, inputColumnNames, this, addBatchDimensionInput);

        /// <summary>
        /// Retrain the TensorFlow model on new data.
        /// The model is not loaded again instead the information contained in <see cref="TensorFlowModel"/> class is reused
        /// (c.f. <see cref="TensorFlowModel.ModelPath"/> and <see cref="TensorFlowModel.Session"/>).
        /// </summary>
        /// <param name="inputColumnNames"> The names of the model inputs.</param>
        /// <param name="outputColumnNames">The names of the requested model outputs.</param>
        /// <param name="labelColumnName">Name of the label column.</param>
        /// <param name="tensorFlowLabel">Name of the node in TensorFlow graph that is used as label during training in TensorFlow.
        /// The value of <paramref name="labelColumnName"/> from <see cref="IDataView"/> is fed to this node.</param>
        /// <param name="optimizationOperation">The name of the optimization operation in the TensorFlow graph.</param>
        /// <param name="epoch">Number of training iterations.</param>
        /// <param name="batchSize">Number of samples to use for mini-batch training.</param>
        /// <param name="lossOperation">The name of the operation in the TensorFlow graph to compute training loss (Optional).</param>
        /// <param name="metricOperation">The name of the operation in the TensorFlow graph to compute performance metric during training (Optional).</param>
        /// <param name="learningRateOperation">The name of the operation in the TensorFlow graph which sets optimizer learning rate (Optional).</param>
        /// <param name="learningRate">Learning rate to use during optimization (Optional).</param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        /// <remarks>
        /// The support for retraining is experimental.
        /// </remarks>
        public TensorFlowEstimator RetrainTensorFlowModel(
            string[] outputColumnNames,
            string[] inputColumnNames,
            string labelColumnName,
            string tensorFlowLabel,
            string optimizationOperation,
            int epoch = 10,
            int batchSize = 20,
            string lossOperation = null,
            string metricOperation = null,
            string learningRateOperation = null,
            float learningRate = 0.01f,
            bool addBatchDimensionInput = false)
        {
            var options = new TensorFlowEstimator.Options()
            {
                ModelLocation = ModelPath,
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
                ReTrain = true,
                AddBatchDimensionInputs = addBatchDimensionInput
            };
            return new TensorFlowEstimator(_env, options, this);
        }
    }
}
