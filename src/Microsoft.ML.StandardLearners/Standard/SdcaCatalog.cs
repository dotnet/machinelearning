// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using System;

namespace Microsoft.ML
{
    /// <summary>
    /// Extension methods for instantiating SDCA trainer estimators.
    /// </summary>
    public static class SdcaRegressionExtensions
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the SDCA trainer.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="loss">The custom loss, if unspecified will be <see cref="SquaredLossSDCARegressionLossFunction"/>.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct method signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        public static SdcaRegressionTrainer StochasticDualCoordinateAscent(this RegressionContext.RegressionTrainers ctx,
        string label = DefaultColumnNames.Label, string features = DefaultColumnNames.Features, string weights = null,
            ISupportSdcaRegressionLoss loss = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null,
            Action<SdcaRegressionTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new SdcaRegressionTrainer(env, features, label, weights, loss, l2Const, l1Threshold, maxIterations, advancedSettings);
        }
    }

    public static class SdcaBinaryClassificationExtensions
    {
        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="ctx">The binary classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="loss">The custom loss. Defaults to log-loss if not specified.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct method signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/SDCA.cs?range=50-51)]
        /// ]]></format>
        /// </example>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        ///  [!code-csharp[SDCA](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/SDCA.cs?range=6-9,14-77)]
        /// ]]></format>
        /// </example>
        public static LinearClassificationTrainer StochasticDualCoordinateAscent(
                this BinaryClassificationContext.BinaryClassificationTrainers ctx,
                string label = DefaultColumnNames.Label, string features = DefaultColumnNames.Features,
                string weights = null,
                ISupportSdcaClassificationLoss loss = null,
                float? l2Const = null,
                float? l1Threshold = null,
                int? maxIterations = null,
                Action<LinearClassificationTrainer.Arguments> advancedSettings = null
            )
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new LinearClassificationTrainer(env, features, label, weights, loss, l2Const, l1Threshold, maxIterations, advancedSettings);
        }
    }

    public static class SdcaMulticlassExtensions
    {

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="loss">The optional custom loss.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct method signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        public static SdcaMultiClassTrainer StochasticDualCoordinateAscent(this MulticlassClassificationContext.MulticlassClassificationTrainers ctx,
                    string label = DefaultColumnNames.Label,
                    string features = DefaultColumnNames.Features,
                    string weights = null,
                    ISupportSdcaClassificationLoss loss = null,
                    float? l2Const = null,
                    float? l1Threshold = null,
                    int? maxIterations = null,
                    Action<SdcaMultiClassTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new SdcaMultiClassTrainer(env, features, label, weights, loss, l2Const, l1Threshold, maxIterations, advancedSettings);
        }
    }
}
