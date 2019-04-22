// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for the <see cref="BinaryClassificationCatalog"/> to create instances
    /// of field aware factorization trainer components.
    /// </summary>
    public static class FactorizationMachineExtensions
    {
        /// <summary>
        /// Create <see cref="FieldAwareFactorizationMachineTrainer"/>, which predicts a target using a field-aware factorization machine trained over boolean label data.
        /// </summary>
        /// <remarks>
        /// Note that because there is only one feature column, the underlying model is equivalent to standard factorization machine.
        /// </remarks>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column. The column data must be <see cref="System.Boolean"/>.</param>
        /// <param name="featureColumnName">The name of the feature column. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[FieldAwareFactorizationMachineWithoutArguments](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/BinaryClassification/FactorizationMachine.cs)]
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
        /// Create <see cref="FieldAwareFactorizationMachineTrainer"/>, which predicts a target using a field-aware factorization machine trained over boolean label data.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="labelColumnName">The name of the label column. The column data must be <see cref="System.Boolean"/>.</param>
        /// <param name="featureColumnNames">The names of the feature columns. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
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
        /// Create <see cref="FieldAwareFactorizationMachineTrainer"/> using advanced options, which predicts a target using a field-aware factorization machine trained over boolean label data.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="options">Trainer options.</param>
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
