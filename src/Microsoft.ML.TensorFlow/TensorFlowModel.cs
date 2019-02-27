// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
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
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlowTransform.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public TensorFlowEstimator ScoreTensorFlowModel(string outputColumnName, string inputColumnName)
            => new TensorFlowEstimator(_env, new[] { outputColumnName }, new[] { inputColumnName }, ModelPath);

        /// <summary>
        /// Scores a dataset using a pre-traiend TensorFlow model.
        /// </summary>
        /// <param name="inputColumnNames"> The names of the model inputs.</param>
        /// <param name="outputColumnNames">The names of the requested model outputs.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ScoreTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlow/ImageClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public TensorFlowEstimator ScoreTensorFlowModel(string[] outputColumnNames, string[] inputColumnNames)
            => new TensorFlowEstimator(_env, outputColumnNames, inputColumnNames, ModelPath);

        /// <summary>
        /// Create the <see cref="TensorFlowEstimator"/> for scoring or retraining using the tensorflow model.
        /// The model is not loaded again instead the information contained in <see cref="TensorFlowModel"/> class is reused
        /// (c.f. <see cref="TensorFlowModel.ModelPath"/> and <see cref="TensorFlowModel.Session"/>).
        /// </summary>
        /// <param name="options">The <see cref="TensorFlowEstimator.Options"/> specifying the inputs and the settings of the <see cref="TensorFlowEstimator"/>.</param>
        public TensorFlowEstimator CreateTensorFlowEstimator(TensorFlowEstimator.Options options)
        {
            options.ModelLocation = ModelPath;
            return new TensorFlowEstimator(_env, options, this);
        }
    }
}
