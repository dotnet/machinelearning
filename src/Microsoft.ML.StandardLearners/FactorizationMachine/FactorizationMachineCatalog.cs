// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FactorizationMachine;
using Microsoft.ML.Trainers;
using System;

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
        /// <param name="ctx">The binary classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="advancedSettings">A delegate to set more settings.</param>
        /// <returns></returns>
        public static FieldAwareFactorizationMachineTrainer FieldAwareFactorizationMachine(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
                string label, string[] features,
                string weights = null,
                Action<FieldAwareFactorizationMachineTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new FieldAwareFactorizationMachineTrainer(env, label, features, weights, advancedSettings: advancedSettings);
        }
    }
}
