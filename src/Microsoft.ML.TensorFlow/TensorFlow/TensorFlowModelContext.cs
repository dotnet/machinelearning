// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Transforms.TensorFlow
{
    /// <summary>
    /// This class holds the information related to TensorFLow model and session.
    /// It provides a convenient way to query model schema
    ///     - Get complete schema by calling <see cref="GetModelSchema()"/>
    ///     - Get schema related to model input by calling <see cref="GetInputSchema()"/>
    /// </summary>
    public class TensorFlowModelContext
    {
        internal TFSession Session { get; private set; }
        public string ModelPath { get; private set; }

        private readonly IHostEnvironment _host;

        /// <summary>
        /// Instantiates <see cref="TensorFlowModelContext"/>
        /// </summary>
        /// <param name="env">An <see cref="IHostEnvironment"/> object.</param>
        /// <param name="session">TensorFlow session object.</param>
        /// <param name="modelLocation">Location of the model from where <paramref name="session"/> was loaded.</param>
        internal TensorFlowModelContext(IHostEnvironment env, TFSession session, string modelLocation)
        {
            Session = session;
            ModelPath = modelLocation;
            _host = env;
        }

        /// <summary>
        /// Get <see cref="ISchema"/> for complete model. Every node in the TensorFlow model will be included in the <see cref="ISchema"/> object.
        /// </summary>
        public ISchema GetModelSchema()
        {
            return TensorFlowUtils.GetModelSchema(_host, Session.Graph);
        }

        /// <summary>
        /// Get <see cref="ISchema"/> for only those nodes which are marked "PlaceHolder" in the TensorFlow model.
        /// This method is convenient for exploring the model input(s) in case TensorFlow graph is very large.
        /// </summary>
        public ISchema GetInputSchema()
        {
            return TensorFlowUtils.GetModelSchema(_host, Session.Graph, "Placeholder");
        }
    }
}
