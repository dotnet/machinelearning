// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TensorFlow;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// This class holds the information related to TensorFlow model and session.
    /// It provides a convenient way to query model schema as follows.
    /// </summary>
    public sealed class TensorFlowModelInfo
    {
        internal TFSession Session { get; }
        public string ModelPath { get; }

        /// <summary>
        /// Instantiates <see cref="TensorFlowModelInfo"/>.
        /// </summary>
        /// <param name="session">TensorFlow session object.</param>
        /// <param name="modelLocation">Location of the model from where <paramref name="session"/> was loaded.</param>
        internal TensorFlowModelInfo(TFSession session, string modelLocation)
        {
            Session = session;
            ModelPath = modelLocation;
        }
    }
}
