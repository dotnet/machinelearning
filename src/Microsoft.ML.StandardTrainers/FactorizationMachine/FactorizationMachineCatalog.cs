// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML
{
    /// <summary>
    /// Extension method to create <see cref="FieldAwareFactorizationMachineTrainer"/>
    /// </summary>
    public static class FactorizationMachineExtensions
    {
        /// <summary>
        /// Predict a target using a field-aware factorization machine algorithm.
        /// </summary>
        /// <remarks>
        /// Note that because there is only one feature column, the underlying model is equivalent to standard factorization machine.
        /// </remarks>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FieldAwareFactorizationMachineWithoutArguments](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/FieldAwareFactorizationMachineWithoutArguments.cs)]
        /// ]]></format>
        /// </example>
        public static FieldAwareFactorizationMachineTrainer FieldAwareFactorizationMachine(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string featureColumnName = DefaultColumnNames.Features,
            string labelColumnName = DefaultColumnNames.Label,
            string exampleWeightColumnName = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FieldAwareFactorizationMachineTrainer(env, new string[] { featureColumnName }, labelColumnName, exampleWeightColumnName);
        }

        /// <summary>
        /// Predict a target using a field-aware factorization machine algorithm.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="featureColumnNames">The name(s) of the feature columns.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FieldAwareFactorizationMachine](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/FieldAwareFactorizationMachine.cs)]
        /// ]]></format>
        /// </example>
        public static FieldAwareFactorizationMachineTrainer FieldAwareFactorizationMachine(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string[] featureColumnNames,
            string labelColumnName = DefaultColumnNames.Label,
            string exampleWeightColumnName = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FieldAwareFactorizationMachineTrainer(env, featureColumnNames, labelColumnName, exampleWeightColumnName);
        }

        /// <summary>
        /// Predict a target using a field-aware factorization machine algorithm.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FieldAwareFactorizationMachine](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/FieldAwareFactorizationMachineWithOptions.cs)]
        /// ]]></format>
        /// </example>
        public static FieldAwareFactorizationMachineTrainer FieldAwareFactorizationMachine(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            FieldAwareFactorizationMachineTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FieldAwareFactorizationMachineTrainer(env, options);
        }
    }
}
