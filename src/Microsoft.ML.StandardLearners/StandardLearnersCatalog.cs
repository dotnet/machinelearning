// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Online;
using System;

namespace Microsoft.ML
{
    using SgdArguments = StochasticGradientDescentClassificationTrainer.Arguments;
    using LRArguments = LogisticRegression.Arguments;

    /// <summary>
    /// TrainerEstimator extension methods.
    /// </summary>
    public static class StandardLearnersCatalog
    {
        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="StochasticGradientDescentClassificationTrainer"/> trainer.
        /// </summary>
        /// <param name="ctx">The binary classificaiton context trainer object.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="weights">The name for the example weight column.</param>
        /// <param name="maxIterations">The maximum number of iterations; set to 1 to simulate online learning.</param>
        /// <param name="initLearningRate">The initial learning rate used by SGD.</param>
        /// <param name="l2Weight">The L2 regularization constant.</param>
        /// <param name="loss">The loss function to use.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static StochasticGradientDescentClassificationTrainer StochasticGradientDescent(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            int maxIterations = SgdArguments.Defaults.MaxIterations,
            double initLearningRate = SgdArguments.Defaults.InitLearningRate,
            float l2Weight = SgdArguments.Defaults.L2Weight,
            ISupportClassificationLossFactory loss = null,
            Action<SgdArguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new StochasticGradientDescentClassificationTrainer(env, labelColumn, featureColumn, weights, maxIterations, initLearningRate, l2Weight, loss, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the SDCA trainer.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="labelColumn">The label column, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l2Const">The L2 regularization hyperparameter.</param>
        /// <param name="l1Threshold">The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.</param>
        /// <param name="maxIterations">The maximum number of passes to perform over the data.</param>
        /// <param name="loss">The custom loss, if unspecified will be <see cref="SquaredLoss"/>.</param>
        /// <param name="advancedSettings">A delegate to set more settings.
        /// The settings here will override the ones provided in the direct method signature,
        /// if both are present and have different values.
        /// The columns names, however need to be provided directly, not through the <paramref name="advancedSettings"/>.</param>
        public static SdcaRegressionTrainer StochasticDualCoordinateAscent(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            ISupportSdcaRegressionLoss loss = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null,
            Action<SdcaRegressionTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new SdcaRegressionTrainer(env, labelColumn, featureColumn, weights, loss, l2Const, l1Threshold, maxIterations, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="ctx">The binary classification context trainer object.</param>
        /// <param name="labelColumn">The labelColumn, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
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
        public static SdcaBinaryTrainer StochasticDualCoordinateAscent(
                this BinaryClassificationContext.BinaryClassificationTrainers ctx,
                string labelColumn = DefaultColumnNames.Label,
                string featureColumn = DefaultColumnNames.Features,
                string weights = null,
                ISupportSdcaClassificationLoss loss = null,
                float? l2Const = null,
                float? l1Threshold = null,
                int? maxIterations = null,
                Action<SdcaBinaryTrainer.Arguments> advancedSettings = null
            )
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new SdcaBinaryTrainer(env, labelColumn, featureColumn, weights, loss, l2Const, l1Threshold, maxIterations, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the SDCA trainer.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="labelColumn">The labelColumn, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
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
                    string labelColumn = DefaultColumnNames.Label,
                    string featureColumn = DefaultColumnNames.Features,
                    string weights = null,
                    ISupportSdcaClassificationLoss loss = null,
                    float? l2Const = null,
                    float? l1Threshold = null,
                    int? maxIterations = null,
                    Action<SdcaMultiClassTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new SdcaMultiClassTrainer(env, labelColumn, featureColumn, weights, loss, l2Const, l1Threshold, maxIterations, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a linear binary classification model trained with the AveragedPerceptron trainer, and a custom loss.
        /// </summary>
        /// <param name="ctx">The binary classification context trainer object.</param>
        /// <param name="labelColumn">The name of the label column, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="lossFunction">The custom loss.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="learningRate">The learning Rate.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 regularization weight.</param>
        /// <param name="numIterations">Number of training iterations through the data.</param>
        /// <param name="advancedSettings">A delegate to supply more advanced arguments to the algorithm.</param>
        public static AveragedPerceptronTrainer AveragedPerceptron(
            this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            IClassificationLoss lossFunction = null,
            float learningRate = AveragedLinearArguments.AveragedDefaultArgs.LearningRate,
            bool decreaseLearningRate = AveragedLinearArguments.AveragedDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = AveragedLinearArguments.AveragedDefaultArgs.L2RegularizerWeight,
            int numIterations = AveragedLinearArguments.AveragedDefaultArgs.NumIterations,
            Action<AveragedPerceptronTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new AveragedPerceptronTrainer(env, labelColumn, featureColumn, weights, lossFunction ?? new LogLoss(), learningRate, decreaseLearningRate, l2RegularizerWeight, numIterations, advancedSettings);
        }

        private sealed class TrivialClassificationLossFactory : ISupportClassificationLossFactory
        {
            private readonly IClassificationLoss _loss;

            public TrivialClassificationLossFactory(IClassificationLoss loss)
            {
                _loss = loss;
            }

            public IClassificationLoss CreateComponent(IHostEnvironment env)
            {
                return _loss;
            }
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="OnlineGradientDescentTrainer"/> trainer.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="labelColumn">The name of the label, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="lossFunction">The custom loss. Defaults to <see cref="SquaredLoss"/> if not provided.</param>
        /// <param name="learningRate">The learning Rate.</param>
        /// <param name="decreaseLearningRate">Decrease learning rate as iterations progress.</param>
        /// <param name="l2RegularizerWeight">L2 regularization weight.</param>
        /// <param name="numIterations">Number of training iterations through the data.</param>
        /// <param name="advancedSettings">A delegate to supply more advanced arguments to the algorithm.</param>
        public static OnlineGradientDescentTrainer OnlineGradientDescent(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            IRegressionLoss lossFunction = null,
            float learningRate = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.LearningRate,
            bool decreaseLearningRate = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = AveragedLinearArguments.AveragedDefaultArgs.L2RegularizerWeight,
            int numIterations = OnlineLinearArguments.OnlineDefaultArgs.NumIterations,
            Action<AveragedLinearArguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new OnlineGradientDescentTrainer(env, labelColumn, featureColumn, learningRate, decreaseLearningRate, l2RegularizerWeight, numIterations, weights, lossFunction, advancedSettings);
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/> trainer.
        /// </summary>
        /// <param name="ctx">The binary classificaiton context trainer object.</param>
        /// <param name="labelColumn">The label column name, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularization term.</param>
        /// <param name="l2Weight">Weight of L2 regularization term.</param>
        /// <param name="memorySize">Memory size for <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/>. Lower=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static LogisticRegression LogisticRegression(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = LRArguments.Defaults.L1Weight,
            float l2Weight = LRArguments.Defaults.L2Weight,
            float optimizationTolerance = LRArguments.Defaults.OptTol,
            int memorySize = LRArguments.Defaults.MemorySize,
            bool enforceNoNegativity = LRArguments.Defaults.EnforceNonNegativity,
            Action<LRArguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new LogisticRegression(env, labelColumn, featureColumn, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/> trainer.
        /// </summary>
        /// <param name="ctx">The regression context trainer object.</param>
        /// <param name="labelColumn">The labelColumn, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="l1Weight">Weight of L1 regularization term.</param>
        /// <param name="l2Weight">Weight of L2 regularization term.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="memorySize">Memory size for <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/>. Lower=faster, less accurate.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static PoissonRegression PoissonRegression(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = LRArguments.Defaults.L1Weight,
            float l2Weight = LRArguments.Defaults.L2Weight,
            float optimizationTolerance = LRArguments.Defaults.OptTol,
            int memorySize = LRArguments.Defaults.MemorySize,
            bool enforceNoNegativity = LRArguments.Defaults.EnforceNonNegativity,
            Action<PoissonRegression.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new PoissonRegression(env, labelColumn, featureColumn, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity, advancedSettings);
        }

        /// <summary>
        /// Predict a target using a linear multiclass classification model trained with the <see cref="Microsoft.ML.Runtime.Learners.MulticlassLogisticRegression"/> trainer.
        /// </summary>
        /// <param name="ctx">The multiclass classification context trainer object.</param>
        /// <param name="labelColumn">The labelColumn, or dependent variable.</param>
        /// <param name="featureColumn">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="enforceNoNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Weight">Weight of L1 regularization term.</param>
        /// <param name="l2Weight">Weight of L2 regularization term.</param>
        /// <param name="memorySize">Memory size for <see cref="Microsoft.ML.Runtime.Learners.LogisticRegression"/>. Lower=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static MulticlassLogisticRegression LogisticRegression(this MulticlassClassificationContext.MulticlassClassificationTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            float l1Weight = LRArguments.Defaults.L1Weight,
            float l2Weight = LRArguments.Defaults.L2Weight,
            float optimizationTolerance = LRArguments.Defaults.OptTol,
            int memorySize = LRArguments.Defaults.MemorySize,
            bool enforceNoNegativity = LRArguments.Defaults.EnforceNonNegativity,
            Action<MulticlassLogisticRegression.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new MulticlassLogisticRegression(env, labelColumn, featureColumn, weights, l1Weight, l2Weight, optimizationTolerance, memorySize, enforceNoNegativity, advancedSettings);
        }
    }
}
