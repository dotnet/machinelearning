// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Trainers.SymSgd;
using System;

namespace Microsoft.ML
{
    /// <summary>
    /// The trainer context extensions for the <see cref="OlsLinearRegressionTrainer"/> and <see cref="SymSgdClassificationTrainer"/>.
    /// </summary>
    public static class HalLearnersCatalog
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OlsLinearRegressionTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The features column.</param>
        /// <param name="weights">The weights column.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static OlsLinearRegressionTrainer OrdinaryLeastSquares(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            Action<OlsLinearRegressionTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new OlsLinearRegressionTrainer(env, labelColumn, featureColumn, weights, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="SymSgdClassificationTrainer"/>.
        /// </summary>
        /// <param name="ctx">The <see cref="RegressionContext"/>.</param>
        /// <param name="labelColumn">The labelColumn column.</param>
        /// <param name="featureColumn">The features column.</param>
        /// <param name="advancedSettings">Algorithm advanced settings.</param>
        public static SymSgdClassificationTrainer SymbolicStochasticGradientDescent(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            Action<SymSgdClassificationTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new SymSgdClassificationTrainer(env, labelColumn, featureColumn, advancedSettings);
        }
    }
}
