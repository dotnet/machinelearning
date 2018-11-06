// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using System;

namespace Microsoft.ML
{
    using Arguments = LogisticRegression.Arguments;

    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static class LbfgsBinaryClassificationExtensions
    {
        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/> trainer.
        /// </summary>
        /// <param name="ctx">The binary classificaiton context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularization term.</param>
        /// <param name="l2Weight">Weight of L2 regularization term.</param>
        /// <param name="memorySize">Memory size for <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/>. Lower=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static LogisticRegression LogisticRegression(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string label,
            string features,
            string weights = null,
            float l1Weight = Arguments.Defaults.L1Weight,
            float l2Weight = Arguments.Defaults.L2Weight,
            float optimizationTolerance = Arguments.Defaults.OptTol,
            int memorySize = Arguments.Defaults.MemorySize,
            bool enforceNoNegativity = Arguments.Defaults.EnforceNonNegativity,
            Action<Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new LogisticRegression(env, label, features, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity, advancedSettings);
        }
    }

    /// <summary>
    /// Regression trainer estimators.
    /// </summary>
    public static class LbfgsRegressionExtensions
    {

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/> trainer.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l1Weight">Weight of L1 regularization term.</param>
        /// <param name="l2Weight">Weight of L2 regularization term.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="memorySize">Memory size for <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/>. Lower=faster, less accurate.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static PoissonRegression PoissonRegression(this RegressionContext.RegressionTrainers ctx,
            string label,
            string features,
            string weights = null,
            float l1Weight = Arguments.Defaults.L1Weight,
            float l2Weight = Arguments.Defaults.L2Weight,
            float optimizationTolerance = Arguments.Defaults.OptTol,
            int memorySize = Arguments.Defaults.MemorySize,
            bool enforceNoNegativity = Arguments.Defaults.EnforceNonNegativity,
            Action<PoissonRegression.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new PoissonRegression(env, label, features, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity, advancedSettings);
        }
    }

    /// <summary>
    /// Multiclass Classification trainer estimators.
    /// </summary>
    public static class LbfgsMulticlassExtensions
    {

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the <see cref="Microsoft.ML.Runtime.Learners.MulticlassLogisticRegression"/> trainer.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularization term.</param>
        /// <param name="l2Weight">Weight of L2 regularization term.</param>
        /// <param name="memorySize">Memory size for <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/>. Lower=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static MulticlassLogisticRegression LogisticRegression(this MulticlassClassificationContext.MulticlassClassificationTrainers ctx,
            string label,
            string features,
            string weights = null,
            float l1Weight = Arguments.Defaults.L1Weight,
            float l2Weight = Arguments.Defaults.L2Weight,
            float optimizationTolerance = Arguments.Defaults.OptTol,
            int memorySize = Arguments.Defaults.MemorySize,
            bool enforceNoNegativity = Arguments.Defaults.EnforceNonNegativity,
            Action<MulticlassLogisticRegression.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new MulticlassLogisticRegression(env, label, features, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity, advancedSettings);
        }

    }
}
