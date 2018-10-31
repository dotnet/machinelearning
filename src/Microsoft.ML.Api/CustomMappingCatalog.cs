// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML
{
    /// <summary>
    /// Extension methods for custom mapping transformers.
    /// </summary>
    public static class CustomMappingCatalog
    {
        /// <summary>
        /// Create a custom mapping of input columns to output columns.
        /// </summary>
        /// <typeparam name="TSrc">The class defining which columns to take from the incoming data.</typeparam>
        /// <typeparam name="TDst">The class defining which new columns are added to the data.</typeparam>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="mapAction">The mapping action. This must be thread-safe and free from side effects.</param>
        /// <param name="contractName">The contract name, used by ML.NET for loading the model. If <c>null</c> is specified, such a trained model would not be save-able.</param>
        /// <param name="inputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TSrc"/> and input data.</param>
        /// <param name="outputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TDst"/> and output data.</param>
        public static CustomMappingEstimator<TSrc, TDst> CustomMapping<TSrc, TDst>(this TransformsCatalog catalog, Action<TSrc, TDst> mapAction, string contractName,
                SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class, new()
            where TDst : class, new()
            => new CustomMappingEstimator<TSrc, TDst>(catalog.GetEnvironment(), mapAction, contractName, inputSchemaDefinition, outputSchemaDefinition);

        /// <summary>
        /// Create a custom mapping of input columns to output columns. Most likely, you should call this method when you are loading the model:
        /// use <see cref="CustomMapping{TSrc, TDst}(TransformsCatalog, Action{TSrc, TDst}, string, SchemaDefinition, SchemaDefinition)"/> when you are
        /// training the model.
        /// </summary>
        /// <typeparam name="TSrc">The class defining which columns to take from the incoming data.</typeparam>
        /// <typeparam name="TDst">The class defining which new columns are added to the data.</typeparam>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="mapAction">The mapping action. This must be thread-safe and free from side effects.</param>
        /// <param name="contractName">The contract name, used by ML.NET for loading the model. If <c>null</c> is specified, such a trained model would not be save-able.</param>
        /// <param name="inputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TSrc"/> and input data.</param>
        /// <param name="outputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TDst"/> and output data.</param>
        public static CustomMappingTransformer<TSrc, TDst> CustomMappingTransformer<TSrc, TDst>(this TransformsCatalog catalog, Action<TSrc, TDst> mapAction, string contractName,
                SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class, new()
            where TDst : class, new()
            => new CustomMappingTransformer<TSrc, TDst>(catalog.GetEnvironment(), mapAction, contractName, inputSchemaDefinition, outputSchemaDefinition);
    }
}
