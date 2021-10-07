// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.TensorFlow;
using Tensorflow;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This class holds the information related to TensorFlow model and session.
    /// It provides some convenient methods to query model schema as well as
    /// creation of <see cref="TensorFlowEstimator"/> object.
    /// </summary>
    public sealed class TensorFlowModel : IDisposable
    {
        internal Session Session { get; }
        internal string ModelPath { get; }
        internal bool TreatOutputAsBatched { get; }

        private readonly IHostEnvironment _env;

        /// <summary>
        /// Instantiates <see cref="TensorFlowModel"/>.
        /// </summary>
        /// <param name="env">An <see cref="IHostEnvironment"/> object.</param>
        /// <param name="session">TensorFlow session object.</param>
        /// <param name="modelLocation">Location of the model from where <paramref name="session"/> was loaded.</param>
        /// <param name="treatOutputAsBatched">If the first dimension of the output is unknown, should it be treated as batched or not.</param>
        internal TensorFlowModel(IHostEnvironment env, Session session, string modelLocation, bool treatOutputAsBatched = true)
        {
            Session = session;
            ModelPath = modelLocation;
            TreatOutputAsBatched = treatOutputAsBatched;
            _env = env;
            _disposed = false;
        }

        /// <summary>
        /// Get <see cref="DataViewSchema"/> for complete model. Every node in the TensorFlow model will be included in the <see cref="DataViewSchema"/> object.
        /// </summary>
        public DataViewSchema GetModelSchema()
        {
            return TensorFlowUtils.GetModelSchema(_env, Session.graph, TreatOutputAsBatched);
        }

        /// <summary>
        /// Get <see cref="DataViewSchema"/> for only those nodes which are marked "Placeholder" in the TensorFlow model.
        /// This method is convenient for exploring the model input(s) in case TensorFlow graph is very large.
        /// </summary>
        public DataViewSchema GetInputSchema()
        {
            return TensorFlowUtils.GetModelSchema(_env, Session.graph, TreatOutputAsBatched, "Placeholder");
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
        public TensorFlowEstimator ScoreTensorFlowModel(string outputColumnName, string inputColumnName, bool addBatchDimensionInput = false)
            => new TensorFlowEstimator(_env, new[] { outputColumnName }, new[] { inputColumnName }, this, addBatchDimensionInput);

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
        public TensorFlowEstimator ScoreTensorFlowModel(string[] outputColumnNames, string[] inputColumnNames, bool addBatchDimensionInput = false)
            => new TensorFlowEstimator(_env, outputColumnNames, inputColumnNames, this, addBatchDimensionInput);

        #region IDisposable Support
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
                return;

            Session.Dispose();

            _disposed = true;
        }
        #endregion
    }
}
