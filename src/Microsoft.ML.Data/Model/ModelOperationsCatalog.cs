// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System.IO;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// An object serving as a 'catalog' of available model operations.
    /// </summary>
    public sealed class ModelOperationsCatalog
    {
        internal IHostEnvironment Environment { get; }

        internal ModelOperationsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            Environment = env;
        }

        /// <summary>
        /// Save the model to the stream.
        /// </summary>
        /// <param name="model">The trained model to be saved.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public void Save(ITransformer model, Stream stream) => model.SaveTo(Environment, stream);

        /// <summary>
        /// Load the model from the stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream to load from.</param>
        /// <returns>The loaded model.</returns>
        public ITransformer Load(Stream stream) => TransformerChain.LoadFrom(Environment, stream);
    }
}
