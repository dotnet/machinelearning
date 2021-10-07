// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// Specifies the options to use when creating a <see cref="PredictionEnginePool{TData, TPrediction}"/>.
    /// </summary>
    public class PredictionEnginePoolOptions<TData, TPrediction>
        where TData : class
        where TPrediction : class, new()
    {
        /// <summary>
        /// Gets the <see cref="ModelLoader"/> object used to load the model
        /// from the source location.
        /// </summary>
        public ModelLoader ModelLoader { get; set; }
    }
}
