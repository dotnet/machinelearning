// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TransferLearning;
using Tensorflow;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This class holds the information related to TensorFlow model and session.
    /// It provides some convenient methods to query model schema as well as
    /// creation of <see cref="TransferLearningEstimator"/> object.
    /// </summary>
    public sealed class TransferLearningModel
    {

        private readonly IHostEnvironment _env;

        internal TransferLearningModel(IHostEnvironment env)
        {
            _env = env;
        }

        /// <summary>
        /// Get <see cref="DataViewSchema"/> for complete model. Every node in the TensorFlow model will be included in the <see cref="DataViewSchema"/> object.
        /// </summary>
        public DataViewSchema GetModelSchema()
        {
            throw new System.NotImplementedException();
            //return TransferLearning.GetModelSchema(_env, Session.graph);
        }

        /// <summary>
        /// Get <see cref="DataViewSchema"/> for only those nodes which are marked "Placeholder" in the TensorFlow model.
        /// This method is convenient for exploring the model input(s) in case TensorFlow graph is very large.
        /// </summary>
        public DataViewSchema GetInputSchema()
        {
            throw new System.NotImplementedException();
            //return TransferLearning.GetModelSchema(_env, Session.graph, "Placeholder");
        }
    }
}
