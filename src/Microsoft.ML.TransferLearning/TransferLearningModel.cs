// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TensorFlow;
using Tensorflow;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This class holds the information related to TensorFlow model and session.
    /// It provides some convenient methods to query model schema as well as
    /// creation of <see cref="TensorFlowEstimator"/> object.
    /// </summary>
    public sealed class TransferLearningModel
    {
        internal Session Session { get; }
        internal string ModelPath { get; }

        private readonly IHostEnvironment _env;

        internal TransferLearningModel(IHostEnvironment env, Session session)
        {
            Session = session;
            ModelPath = "resnet_v2_101_299_frozen.pb";
            _env = env;
        }

        /// <summary>
        /// Get <see cref="DataViewSchema"/> for complete model. Every node in the TensorFlow model will be included in the <see cref="DataViewSchema"/> object.
        /// </summary>
        public DataViewSchema GetModelSchema()
        {
            return TransferLearning.GetModelSchema(_env, Session.Graph);
        }

        /// <summary>
        /// Get <see cref="DataViewSchema"/> for only those nodes which are marked "Placeholder" in the TensorFlow model.
        /// This method is convenient for exploring the model input(s) in case TensorFlow graph is very large.
        /// </summary>
        public DataViewSchema GetInputSchema()
        {
            return TransferLearning.GetModelSchema(_env, Session.Graph, "Placeholder");
        }

        /// <summary>
        /// Scores a dataset using a pre-trained <a href="https://www.tensorflow.org/">TensorFlow</a> model.
        /// </summary>
        /// <param name="inputColumnName"> The name of the model input. The data type is a vector of <see cref="System.Single"/>.</param>
        /// <param name="outputColumnName">The name of the requested model output. The data type is a vector of <see cref="System.Single"/></param>
        /// <param name="addBatchDimensionInput">Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
        /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlow/ImageClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public TransferLearningEstimator ScoreTranferLearningModel(string outputColumnName, string inputColumnName, bool addBatchDimensionInput = false)
            => new TransferLearningEstimator(_env, new[] { outputColumnName }, new[] { inputColumnName }, this, addBatchDimensionInput);

        /// <summary>
        /// Scores a dataset using a pre-trained TensorFlow model.
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
       
        internal TransferLearningEstimator TrainTransferLearningModel(
            string[] outputColumnNames,
            string[] inputColumnNames,
            string labelColumnName,
            string tensorFlowLabel,
            int epoch = 10,
            int batchSize = 20,
            float learningRate = 0.01f,
            bool addBatchDimensionInput = false)
        {
            var options = new TransferLearning.Options()
            {
                ModelLocation = ModelPath,
                InputColumns = inputColumnNames,
                OutputColumns = outputColumnNames,
                LabelColumn = labelColumnName,
                TensorFlowLabel = tensorFlowLabel,
                LearningRate = learningRate,
                BatchSize = batchSize,
                AddBatchDimensionInputs = addBatchDimensionInput
            };
            return new TransferLearningEstimator(_env, options, this);
        }
    }
}
