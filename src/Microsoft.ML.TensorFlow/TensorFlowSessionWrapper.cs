// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Tensorflow;

namespace Microsoft.ML.TensorFlow
{
    /// <summary>
    /// This class holds the information related to TensorFlow model and session.
    /// </summary>
    internal sealed class TensorFlowSessionWrapper
    {
        internal Session Session { get; }
        internal string ModelPath { get; }

        /// <summary>
        /// Instantiates <see cref="TensorFlowSessionWrapper"/>.
        /// </summary>
        /// <param name="session">TensorFlow session object.</param>
        /// <param name="modelLocation">Location of the model from where <paramref name="session"/> was loaded.</param>
        internal TensorFlowSessionWrapper(Session session, string modelLocation)
        {
            Session = session;
            ModelPath = modelLocation;
        }
    }
}
