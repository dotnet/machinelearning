// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Tensorflow;

namespace Microsoft.ML.Dnn
{
    /// <summary>
    /// This class holds the information related to TensorFlow model and session.
    /// It provides some convenient methods to query model schema as well as
    /// creation of <see cref="DnnRetrainEstimator"/> object.
    /// </summary>
    internal sealed class DnnModel
    {
        internal Session Session { get; }
        internal string ModelPath { get; }

        /// <summary>
        /// Instantiates <see cref="DnnModel"/>.
        /// </summary>
        /// <param name="session">TensorFlow session object.</param>
        /// <param name="modelLocation">Location of the model from where <paramref name="session"/> was loaded.</param>
        internal DnnModel(Session session, string modelLocation)
        {
            Session = session;
            ModelPath = modelLocation;
        }
    }
}
