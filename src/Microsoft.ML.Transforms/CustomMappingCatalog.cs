// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// Class containing an extension method for <see cref="TransformsCatalog"/> to create instances of
    /// user-defined one-to-one row mapping transformer components.
    /// </summary>
    public static class CustomMappingCatalog
    {
        /// <summary>
        /// Create a <see cref="CustomMappingEstimator{TSrc, TDst}"/>, which applies a custom mapping of input columns to output columns.
        /// </summary>
        /// <typeparam name="TSrc">The class defining which columns to take from the incoming data.</typeparam>
        /// <typeparam name="TDst">The class defining which new columns are added to the data.</typeparam>
        /// <param name="catalog">The transform catalog</param>
        /// <param name="mapAction">The mapping action. This must be thread-safe and free from side effects.
        /// If the resulting transformer needs to be save-able, the class defining <paramref name="mapAction"/> should implement
        /// <see cref="CustomMappingFactory{TSrc, TDst}"/> and needs to be decorated with
        /// <see cref="CustomMappingFactoryAttributeAttribute"/> with the provided <paramref name="contractName"/>.
        /// The assembly containing the class should be registered in the environment where it is loaded back
        /// using <see cref="ComponentCatalog.RegisterAssembly(System.Reflection.Assembly, bool)"/>.</param>
        /// <param name="contractName">The contract name, used by ML.NET for loading the model.
        /// If <see langword="null"/> is specified, resulting transformer would not be save-able.</param>
        /// <param name="inputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TSrc"/> and input data.
        /// Useful when dealing with annotations.</param>
        /// <param name="outputSchemaDefinition">Additional parameters for schema mapping between <typeparamref name="TDst"/> and output data.
        /// Useful when dealing with annotations.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[CustomMapping](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/CustomMapping.cs)]
        ///  [!code-csharp[CustomMapping](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/CustomMappingSaveAndLoad.cs)]
        ///  [!code-csharp[CustomMapping](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/CustomMappingWithInMemoryCustomType.cs)]
        /// ]]></format>
        /// </example>
        public static CustomMappingEstimator<TSrc, TDst> CustomMapping<TSrc, TDst>(this TransformsCatalog catalog, Action<TSrc, TDst> mapAction, string contractName,
                SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class, new()
            where TDst : class, new()
            => new CustomMappingEstimator<TSrc, TDst>(catalog.GetEnvironment(), mapAction, contractName, inputSchemaDefinition, outputSchemaDefinition);
    }
}
