// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Tensorflow;
using static Microsoft.ML.Transforms.DnnRetrainEstimator;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This class holds the information related to TensorFlow model and session.
    /// It provides some convenient methods to query model schema as well as
    /// creation of <see cref="DnnRetrainEstimator"/> object.
    /// </summary>
    public sealed class DnnModel
    {
        internal Session Session { get; }
        internal string ModelPath { get; }

        private readonly IHostEnvironment _env;

        /// <summary>
        /// Instantiates <see cref="DnnModel"/>.
        /// </summary>
        /// <param name="env">An <see cref="IHostEnvironment"/> object.</param>
        /// <param name="session">TensorFlow session object.</param>
        /// <param name="modelLocation">Location of the model from where <paramref name="session"/> was loaded.</param>
        internal DnnModel(IHostEnvironment env, Session session, string modelLocation)
        {
            Session = session;
            ModelPath = modelLocation;
            _env = env;
        }
    }
}
