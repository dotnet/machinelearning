// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// An object serving as a 'catalog' of available model operations.
    /// </summary>
    public sealed class ModelOperationsCatalog : IInternalCatalog
    {
        IHostEnvironment IInternalCatalog.Environment => _env;
        private readonly IHostEnvironment _env;

        public ExplainabilityTransforms Explainability { get; }

        internal ModelOperationsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            _env = env;

            Explainability = new ExplainabilityTransforms(this);
        }

        /// <summary>
        /// Save the model to the stream.
        /// </summary>
        /// <param name="model">The trained model to be saved.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public void Save(ITransformer model, Stream stream) => model.SaveTo(_env, stream);

        /// <summary>
        /// Load the model from the stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream to load from.</param>
        /// <returns>The loaded model.</returns>
        public ITransformer Load(Stream stream) => TransformerChain.LoadFrom(_env, stream);

        /// <summary>
        /// The catalog of model explainability operations.
        /// </summary>
        public sealed class ExplainabilityTransforms : IInternalCatalog
        {
            IHostEnvironment IInternalCatalog.Environment => _env;
            private readonly IHostEnvironment _env;

            internal ExplainabilityTransforms(ModelOperationsCatalog owner)
            {
                _env = owner._env;
            }
        }

        /// <summary>
        /// Create a prediction engine for one-time prediction.
        /// </summary>
        /// <typeparam name="TSrc">The class that defines the input data.</typeparam>
        /// <typeparam name="TDst">The class that defines the output data.</typeparam>
        /// <param name="transformer">The transformer to use for prediction.</param>
        /// <param name="inputSchemaDefinition">Additional settings of the input schema.</param>
        /// <param name="outputSchemaDefinition">Additional settings of the output schema.</param>
        public PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(ITransformer transformer,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            return new PredictionEngine<TSrc, TDst>(_env, transformer, false, inputSchemaDefinition, outputSchemaDefinition);
        }
    }
}
