// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Extensions.Primitives;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Defines a class that provides the mechanisms to load an ML.NET model
    /// and to propagate notifications that the source of the model has changed.
    /// </summary>
    public abstract class ModelLoader
    {
        /// <summary>
        /// Gets an object that can propagate notifications that
        /// the model has changed.
        /// </summary>
        public abstract IChangeToken GetReloadToken();

        /// <summary>
        /// Gets the ML.NET model.
        /// </summary>
        public abstract ITransformer GetModel();
    }
}
