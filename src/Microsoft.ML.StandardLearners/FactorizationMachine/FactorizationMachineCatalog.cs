﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.FactorizationMachine;

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
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="featureColumns">The features, or independent variables.</param>
        /// <param name="labelColumn">The label, or dependent variable.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct method signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        public static FieldAwareFactorizationMachineTrainer FieldAwareFactorizationMachine(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string[] featureColumns,
            string labelColumn = DefaultColumnNames.Label,
            string weights = null,
            Action<FieldAwareFactorizationMachineTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new FieldAwareFactorizationMachineTrainer(env, featureColumns, labelColumn, weights, advancedSettings: advancedSettings);
        }
    }
}
