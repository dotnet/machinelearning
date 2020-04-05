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
        /// In versions v1.5-preview2 and earlier, the assembly containing the class should be registered in the environment where it is loaded back
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

        /// <summary>
        /// Create a <see cref="StatefulCustomMappingEstimator{TSrc, TState, TDst}"/>, which applies a custom mapping of input columns to output columns,
        /// while allowing a per-cursor state.
        /// </summary>
        /// <typeparam name="TSrc">The class defining which columns to take from the incoming data.</typeparam>
        /// <typeparam name="TState">The type that describes per-cursor state.</typeparam>
        /// <typeparam name="TDst">The class defining which new columns are added to the data.</typeparam>
        /// <param name="catalog">The transform catalog.</param>
        /// <param name="mapAction">The mapping action. In addition to the input and output objects, the action is given a state object that it can look at and/or modify.
        /// If the resulting transformer needs to be save-able, the class defining <paramref name="mapAction"/> should implement
        /// <see cref="StatefulCustomMappingFactory{TSrc, TDst, TState}"/> and needs to be decorated with
        /// <see cref="CustomMappingFactoryAttributeAttribute"/> with the provided <paramref name="contractName"/>.
        /// The assembly containing the class should be registered in the environment where it is loaded back
        /// using <see cref="ComponentCatalog.RegisterAssembly(System.Reflection.Assembly, bool)"/>.</param>
        /// <param name="stateInitAction">The action to initialize the state object, that is called once before the cursor is initialized.</param>
        /// <param name="contractName">The contract name, used by ML.NET for loading the model.
        /// If <see langword="null"/> is specified, resulting transformer would not be save-able.</param>
        public static StatefulCustomMappingEstimator<TSrc, TDst, TState> StatefulCustomMapping<TSrc, TDst, TState>(this TransformsCatalog catalog, Action<TSrc, TDst, TState> mapAction,
            Action<TState> stateInitAction, string contractName)
            where TSrc : class, new()
            where TDst : class, new()
            where TState : class, new()
            => new StatefulCustomMappingEstimator<TSrc, TDst, TState>(catalog.GetEnvironment(), mapAction, contractName, stateInitAction);

        /// <summary>
        /// Drop rows where a specified predicate returns true.
        /// </summary>
        /// <typeparam name="TSrc">The class defining which columns to take from the incoming data.</typeparam>
        /// <param name="catalog">The data operations catalog.</param>
        /// <param name="input">The input data.</param>
        /// <param name="filterPredicate">A predicate, that takes an input of type <typeparamref name="TSrc"/> and returns true if the row should be filtered (dropped) and false otherwise.</param>
        public static IDataView FilterByCustomPredicate<TSrc>(this DataOperationsCatalog catalog, IDataView input, Func<TSrc, bool> filterPredicate)
            where TSrc : class, new()
            => new CustomMappingFilter<TSrc>(catalog.GetEnvironment(), input, filterPredicate);

        /// <summary>
        /// Drop rows where a specified predicate returns true. This filter allows to maintain a per-cursor state.
        /// </summary>
        /// <typeparam name="TSrc">The class defining which columns to take from the incoming data.</typeparam>
        /// <typeparam name="TState">The type that describes per-cursor state.</typeparam>
        /// <param name="catalog">The data operations catalog.</param>
        /// <param name="input">The input data.</param>
        /// <param name="filterPredicate">A predicate, that takes an input of type <typeparamref name="TSrc"/> and a state object of type
        /// <typeparamref name="TState"/>, and returns true if the row should be filtered (dropped) and false otherwise.</param>
        /// <param name="stateInitAction">The action to initialize the state object, that is called once before the cursor is initialized.</param>
        public static IDataView FilterByStatefulCustomPredicate<TSrc, TState>(this DataOperationsCatalog catalog, IDataView input, Func<TSrc, TState, bool> filterPredicate,
            Action<TState> stateInitAction)
            where TSrc : class, new()
            where TState : class, new()
            => new StatefulCustomMappingFilter<TSrc, TState>(catalog.GetEnvironment(), input, filterPredicate, stateInitAction);
    }
}
