// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.StaticPipe
{
    using Options = LogisticRegressionBinaryTrainer.Options;

    /// <summary>
    /// Binary Classification trainer estimators.
    /// </summary>
    public static class LbfgsBinaryClassificationStaticExtensions
    {
        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Microsoft.ML.Trainers.LogisticRegressionBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="enforceNonNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Regularization">Weight of L1 regularization term.</param>
        /// <param name="l2Regularization">Weight of L2 regularization term.</param>
        /// <param name="historySize">Memory size for <see cref="Microsoft.ML.Trainers.LogisticRegressionBinaryTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) LogisticRegressionBinaryClassifier(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights = null,
            float l1Regularization = Options.Defaults.L1Regularization,
            float l2Regularization = Options.Defaults.L2Regularization,
            float optimizationTolerance = Options.Defaults.OptimizationTolerance,
            int historySize = Options.Defaults.HistorySize,
            bool enforceNonNegativity = Options.Defaults.EnforceNonNegativity,
            Action<CalibratedModelParametersBase<LinearBinaryModelParameters,PlattCalibrator>> onFit = null)
        {
            LbfgsStaticUtils.ValidateParams(label, features, weights, l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity, onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new LogisticRegressionBinaryTrainer(env, labelName, featuresName, weightsName,
                        l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        ///  Predict a target using a linear binary classification model trained with the <see cref="Microsoft.ML.Trainers.LogisticRegressionBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The binary classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <returns>The predicted output.</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) LogisticRegressionBinaryClassifier(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            Scalar<bool> label,
            Vector<float> features,
            Scalar<float> weights,
            Options options,
            Action<CalibratedModelParametersBase<LinearBinaryModelParameters,PlattCalibrator>> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValue(options, nameof(options));
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.LabelColumnName = labelName;
                    options.FeatureColumnName = featuresName;
                    options.ExampleWeightColumnName = weightsName;

                    var trainer = new LogisticRegressionBinaryTrainer(env, options);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;

                }, label, features, weights);

            return rec.Output;
        }
    }

    /// <summary>
    /// Regression trainer estimators.
    /// </summary>
    public static class LbfgsRegressionExtensions
    {
        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Trainers.LogisticRegressionBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="enforceNonNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Regularization">Weight of L1 regularization term.</param>
        /// <param name="l2Regularization">Weight of L2 regularization term.</param>
        /// <param name="historySize">Memory size for <see cref="Microsoft.ML.Trainers.LogisticRegressionBinaryTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static Scalar<float> PoissonRegression(this RegressionCatalog.RegressionTrainers catalog,
            Scalar<float> label,
            Vector<float> features,
            Scalar<float> weights = null,
            float l1Regularization = Options.Defaults.L1Regularization,
            float l2Regularization = Options.Defaults.L2Regularization,
            float optimizationTolerance = Options.Defaults.OptimizationTolerance,
            int historySize = Options.Defaults.HistorySize,
            bool enforceNonNegativity = Options.Defaults.EnforceNonNegativity,
            Action<PoissonRegressionModelParameters> onFit = null)
        {
            LbfgsStaticUtils.ValidateParams(label, features, weights, l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity, onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new PoissonRegressionTrainer(env, labelName, featuresName, weightsName,
                        l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));

                    return trainer;
                }, label, features, weights);

            return rec.Score;
        }

        /// <summary>
        /// Predict a target using a linear regression model trained with the <see cref="Microsoft.ML.Trainers.LogisticRegressionBinaryTrainer"/> trainer.
        /// </summary>
        /// <param name="catalog">The regression catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained.  Note that this action cannot change the result in any way; it is only a way for the caller to
        /// be informed about what was learnt.</param>
        /// <returns>The predicted output.</returns>
        public static Scalar<float> PoissonRegression(this RegressionCatalog.RegressionTrainers catalog,
            Scalar<float> label,
            Vector<float> features,
            Scalar<float> weights,
            PoissonRegressionTrainer.Options options,
            Action<PoissonRegressionModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValue(options, nameof(options));
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.LabelColumnName = labelName;
                    options.FeatureColumnName = featuresName;
                    options.ExampleWeightColumnName = weightsName;

                    var trainer = new PoissonRegressionTrainer(env, options);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));

                    return trainer;
                }, label, features, weights);

            return rec.Score;
        }
    }

    /// <summary>
    /// Multiclass Classification trainer estimators.
    /// </summary>
    public static class LbfgsMulticlassExtensions
    {
        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with the L-BFGS method implemented in <see cref="LbfgsMaximumEntropyTrainer"/>.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="enforceNonNegativity">Enforce non-negative weights.</param>
        /// <param name="l1Regularization">Weight of L1 regularization term.</param>
        /// <param name="l2Regularization">Weight of L2 regularization term.</param>
        /// <param name="historySize">Memory size for <see cref="Microsoft.ML.Trainers.LogisticRegressionBinaryTrainer"/>. Low=faster, less accurate.</param>
        /// <param name="optimizationTolerance">Threshold for optimizer convergence.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel)
            LbfgsMaximumEntropy<TVal>(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Key<uint, TVal> label,
            Vector<float> features,
            Scalar<float> weights = null,
            float l1Regularization = Options.Defaults.L1Regularization,
            float l2Regularization = Options.Defaults.L2Regularization,
            float optimizationTolerance = Options.Defaults.OptimizationTolerance,
            int historySize = Options.Defaults.HistorySize,
            bool enforceNonNegativity = Options.Defaults.EnforceNonNegativity,
            Action<MaximumEntropyModelParameters> onFit = null)
        {
            LbfgsStaticUtils.ValidateParams(label, features, weights, l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity, onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassificationReconciler<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new LbfgsMaximumEntropyTrainer(env, labelName, featuresName, weightsName,
                         l1Regularization, l2Regularization, optimizationTolerance, historySize, enforceNonNegativity);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }

        /// <summary>
        /// Predict a target using a maximum entropy classification model trained with the L-BFGS method implemented in <see cref="LbfgsMaximumEntropyTrainer"/>.
        /// </summary>
        /// <param name="catalog">The multiclass classification catalog trainer object.</param>
        /// <param name="label">The label, or dependent variable.</param>
        /// <param name="features">The features, or independent variables.</param>
        /// <param name="weights">The optional example weights.</param>
        /// <param name="options">Advanced arguments to the algorithm.</param>
        /// <param name="onFit">A delegate that is called every time the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
        /// <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
        /// the linear model that was trained. Note that this action cannot change the
        /// result in any way; it is only a way for the caller to be informed about what was learnt.</param>
        /// <returns>The set of output columns including in order the predicted per-class likelihoods (between 0 and 1, and summing up to 1), and the predicted label.</returns>
        public static (Vector<float> score, Key<uint, TVal> predictedLabel)
            LbfgsMaximumEntropy<TVal>(this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            Key<uint, TVal> label,
            Vector<float> features,
            Scalar<float> weights,
            LbfgsMaximumEntropyTrainer.Options options,
            Action<MaximumEntropyModelParameters> onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValue(options, nameof(options));
            Contracts.CheckValueOrNull(onFit);

            var rec = new TrainerEstimatorReconciler.MulticlassClassificationReconciler<TVal>(
                (env, labelName, featuresName, weightsName) =>
                {
                    options.LabelColumnName = labelName;
                    options.FeatureColumnName = featuresName;
                    options.ExampleWeightColumnName = weightsName;

                    var trainer = new LbfgsMaximumEntropyTrainer(env, options);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    return trainer;
                }, label, features, weights);

            return rec.Output;
        }
    }

    internal static class LbfgsStaticUtils
    {
        internal static void ValidateParams(PipelineColumn label,
            Vector<float> features,
            Scalar<float> weights = null,
            float l1Regularization = Options.Defaults.L1Regularization,
            float l2Regularization = Options.Defaults.L2Regularization,
            float optimizationTolerance = Options.Defaults.OptimizationTolerance,
            int historySize = Options.Defaults.HistorySize,
            bool enforceNonNegativity = Options.Defaults.EnforceNonNegativity,
            Delegate onFit = null)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckParam(l2Regularization >= 0, nameof(l2Regularization), "Must be non-negative");
            Contracts.CheckParam(l1Regularization >= 0, nameof(l1Regularization), "Must be non-negative");
            Contracts.CheckParam(optimizationTolerance > 0, nameof(optimizationTolerance), "Must be positive");
            Contracts.CheckParam(historySize > 0, nameof(historySize), "Must be positive");
            Contracts.CheckValueOrNull(onFit);
        }
    }
}
