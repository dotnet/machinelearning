// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// Extension methods to create a prediction engine.
    /// </summary>
    internal static class PredictionEngineExtensions
    {
        /// <summary>
        /// Create a prediction engine for one-time prediction.
        /// </summary>
        /// <typeparam name="TSrc">The class that defines the input data.</typeparam>
        /// <typeparam name="TDst">The class that defines the output data.</typeparam>
        /// <param name="transformer">The transformer to use for prediction.</param>
        /// <param name="env">The environment to use.</param>
        /// <param name="ignoreMissingColumns">Whether to throw an exception if a column exists in
        /// <paramref name="outputSchemaDefinition"/> but the corresponding member doesn't exist in
        /// <typeparamref name="TDst"/>.</param>
        /// <param name="inputSchemaDefinition">Additional settings of the input schema.</param>
        /// <param name="outputSchemaDefinition">Additional settings of the output schema.</param>
        public static PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(this ITransformer transformer,
            IHostEnvironment env, bool ignoreMissingColumns = true, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
            => new PredictionEngine<TSrc, TDst>(env, transformer, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
    }
}
