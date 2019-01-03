// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML
{
    /// <summary>
    /// An object serving as a 'catalog' of available model operations.
    /// </summary>
    public sealed class ModelOperationsCatalog
    {
        internal IHostEnvironment Environment { get; }

        public ExplainabilityTransforms Explainability { get; }

        internal ModelOperationsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            Environment = env;

            Explainability = new ExplainabilityTransforms(this);
        }

        public abstract class SubCatalogBase
        {
            internal IHostEnvironment Environment { get; }

            protected SubCatalogBase(ModelOperationsCatalog owner)
            {
                Environment = owner.Environment;
            }

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

        /// <summary>
        /// The catalog of model explainability operations.
        /// </summary>
        public sealed class ExplainabilityTransforms : SubCatalogBase
        {
            internal ExplainabilityTransforms(ModelOperationsCatalog owner) : base(owner)
            {
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
            return new PredictionEngine<TSrc, TDst>(Environment, transformer, false, inputSchemaDefinition, outputSchemaDefinition);
        }
    }
}
