// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// A training catalog is an object instantiable by a user to do various tasks relating to a particular
    /// "area" of machine learning. A subclass would represent a particular task in machine learning. The idea
    /// is that a user can instantiate that particular area, and get trainers and evaluators.
    /// </summary>
    public abstract class TrainCatalogBase : IInternalCatalog
    {
        IHostEnvironment IInternalCatalog.Environment => Environment;

        [BestFriend]
        private protected IHostEnvironment Environment { get; }

        /// <summary>
        /// Results for specific cross-validation fold.
        /// </summary>
        [BestFriend]
        private protected struct CrossValidationResult
        {
            /// <summary>
            /// Model trained during cross validation fold.
            /// </summary>
            public readonly ITransformer Model;
            /// <summary>
            /// Scored test set with <see cref="Model"/> for this fold.
            /// </summary>
            public readonly IDataView Scores;
            /// <summary>
            /// Fold number.
            /// </summary>
            public readonly int Fold;

            public CrossValidationResult(ITransformer model, IDataView scores, int fold)
            {
                Model = model;
                Scores = scores;
                Fold = fold;
            }
        }
        /// <summary>
        /// Results of running cross-validation.
        /// </summary>
        /// <typeparam name="T">Type of metric class.</typeparam>
        public sealed class CrossValidationResult<T> where T : class
        {
            /// <summary>
            /// Metrics for this cross-validation fold.
            /// </summary>
            public readonly T Metrics;
            /// <summary>
            /// Model trained during cross-validation fold.
            /// </summary>
            public readonly ITransformer Model;
            /// <summary>
            /// The scored hold-out set for this fold.
            /// </summary>
            public readonly IDataView ScoredHoldOutSet;
            /// <summary>
            /// Fold number.
            /// </summary>
            public readonly int Fold;

            internal CrossValidationResult(ITransformer model, T metrics, IDataView scores, int fold)
            {
                Model = model;
                Metrics = metrics;
                ScoredHoldOutSet = scores;
                Fold = fold;
            }
        }

        /// <summary>
        /// Train the <paramref name="estimator"/> on <paramref name="numFolds"/> folds of the data sequentially.
        /// Return each model and each scored test dataset.
        /// </summary>
        [BestFriend]
        private protected CrossValidationResult[] CrossValidateTrain(IDataView data, IEstimator<ITransformer> estimator,
            int numFolds, string samplingKeyColumn, int? seed = null)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckValue(estimator, nameof(estimator));
            Environment.CheckParam(numFolds > 1, nameof(numFolds), "Must be more than 1");
            Environment.CheckValueOrNull(samplingKeyColumn);

            DataOperationsCatalog.EnsureGroupPreservationColumn(Environment, ref data, ref samplingKeyColumn, seed);

            Func<int, CrossValidationResult> foldFunction =
                fold =>
                {
                    var trainFilter = new RangeFilter(Environment, new RangeFilter.Options
                    {
                        Column = samplingKeyColumn,
                        Min = (double)fold / numFolds,
                        Max = (double)(fold + 1) / numFolds,
                        Complement = true
                    }, data);
                    var testFilter = new RangeFilter(Environment, new RangeFilter.Options
                    {
                        Column = samplingKeyColumn,
                        Min = (double)fold / numFolds,
                        Max = (double)(fold + 1) / numFolds,
                        Complement = false
                    }, data);

                    var model = estimator.Fit(trainFilter);
                    var scoredTest = model.Transform(testFilter);
                    return new CrossValidationResult(model, scoredTest, fold);
                };

            // Sequential per-fold training.
            // REVIEW: we could have a parallel implementation here. We would need to
            // spawn off a separate host per fold in that case.
            var result = new CrossValidationResult[numFolds];
            for (int fold = 0; fold < numFolds; fold++)
                result[fold] = foldFunction(fold);

            return result;
        }

        [BestFriend]
        private protected TrainCatalogBase(IHostEnvironment env, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(registrationName, nameof(registrationName));
            Environment = env;
        }

        /// <summary>
        /// Subclasses of <see cref="TrainContext"/> will provide little "extension method" hookable objects
        /// (for example, something like <see cref="BinaryClassificationCatalog.Trainers"/>). User code will only
        /// interact with these objects by invoking the extension methods. The actual component code can work
        /// through <see cref="CatalogUtils"/> to get more "hidden" information from this object,
        /// for example, the environment.
        /// </summary>
        public abstract class CatalogInstantiatorBase : IInternalCatalog
        {
            IHostEnvironment IInternalCatalog.Environment => Owner.GetEnvironment();

            [BestFriend]
            internal TrainCatalogBase Owner { get; }

            [BestFriend]
            private protected CatalogInstantiatorBase(TrainCatalogBase catalog)
            {
                Owner = catalog;
            }
        }
    }

    /// <summary>
    /// The central catalog for binary classification tasks and trainers.
    /// </summary>
    public sealed class BinaryClassificationCatalog : TrainCatalogBase
    {
        /// <summary>
        /// The list of trainers for performing binary classification.
        /// </summary>
        public BinaryClassificationTrainers Trainers { get; }

        internal BinaryClassificationCatalog(IHostEnvironment env)
            : base(env, nameof(BinaryClassificationCatalog))
        {
            Calibrators = new CalibratorsCatalog(this);
            Trainers = new BinaryClassificationTrainers(this);
        }

        public sealed class BinaryClassificationTrainers : CatalogInstantiatorBase
        {
            internal BinaryClassificationTrainers(BinaryClassificationCatalog catalog)
                : base(catalog)
            {
            }
        }

        /// <summary>
        /// Evaluates scored binary classification data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="labelColumnName">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="scoreColumnName">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="probabilityColumnName">The name of the probability column in <paramref name="data"/>, the calibrated version of <paramref name="scoreColumnName"/>.</param>
        /// <param name="predictedLabelColumnName">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public CalibratedBinaryClassificationMetrics Evaluate(IDataView data, string labelColumnName = DefaultColumnNames.Label, string scoreColumnName = DefaultColumnNames.Score,
            string probabilityColumnName = DefaultColumnNames.Probability, string predictedLabelColumnName = DefaultColumnNames.PredictedLabel)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            Environment.CheckNonEmpty(scoreColumnName, nameof(scoreColumnName));
            Environment.CheckNonEmpty(probabilityColumnName, nameof(probabilityColumnName));
            Environment.CheckNonEmpty(predictedLabelColumnName, nameof(predictedLabelColumnName));

            var eval = new BinaryClassifierEvaluator(Environment, new BinaryClassifierEvaluator.Arguments() { });
            return eval.Evaluate(data, labelColumnName, scoreColumnName, probabilityColumnName, predictedLabelColumnName);
        }

        /// <summary>
        /// Evaluates scored binary classification data, without probability-based metrics.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="labelColumnName">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="scoreColumnName">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="predictedLabelColumnName">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these uncalibrated outputs.</returns>
        public BinaryClassificationMetrics EvaluateNonCalibrated(IDataView data, string labelColumnName = DefaultColumnNames.Label, string scoreColumnName = DefaultColumnNames.Score,
            string predictedLabelColumnName = DefaultColumnNames.PredictedLabel)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            Environment.CheckNonEmpty(predictedLabelColumnName, nameof(predictedLabelColumnName));

            var eval = new BinaryClassifierEvaluator(Environment, new BinaryClassifierEvaluator.Arguments() { });
            return eval.Evaluate(data, labelColumnName, scoreColumnName, predictedLabelColumnName);
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numberOfFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="samplingKeyColumnName"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumnName"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numberOfFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumnName">The label column (for evaluation).</param>
        /// <param name="samplingKeyColumnName">Name of a column to use for grouping rows. If two examples share the same value of the <paramref name="samplingKeyColumnName"/>,
        /// they are guaranteed to appear in the same subset (train or test). This can be used to ensure no label leakage from the train to the test set.
        /// If <see langword="null"/> no row grouping will be performed.</param>
        /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public CrossValidationResult<BinaryClassificationMetrics>[] CrossValidateNonCalibrated(
            IDataView data, IEstimator<ITransformer> estimator, int numberOfFolds = 5, string labelColumnName = DefaultColumnNames.Label,
            string samplingKeyColumnName = null, int? seed = null)
        {
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            var result = CrossValidateTrain(data, estimator, numberOfFolds, samplingKeyColumnName, seed);
            return result.Select(x => new CrossValidationResult<BinaryClassificationMetrics>(x.Model,
                EvaluateNonCalibrated(x.Scores, labelColumnName), x.Scores, x.Fold)).ToArray();
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numberOfFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="samplingKeyColumnName"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumnName"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numberOfFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumnName">The label column (for evaluation).</param>
        /// <param name="samplingKeyColumnName">Name of a column to use for grouping rows. If two examples share the same value of the <paramref name="samplingKeyColumnName"/>,
        /// they are guaranteed to appear in the same subset (train or test). This can be used to ensure no label leakage from the train to the test set.
        /// If <see langword="null"/> no row grouping will be performed.</param>
        /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public CrossValidationResult<CalibratedBinaryClassificationMetrics>[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numberOfFolds = 5, string labelColumnName = DefaultColumnNames.Label,
            string samplingKeyColumnName = null, int? seed = null)
        {
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            var result = CrossValidateTrain(data, estimator, numberOfFolds, samplingKeyColumnName, seed);
            return result.Select(x => new CrossValidationResult<CalibratedBinaryClassificationMetrics>(x.Model,
                Evaluate(x.Scores, labelColumnName), x.Scores, x.Fold)).ToArray();
        }

        /// <summary>
        /// The list of trainers for performing binary classification.
        /// </summary>
        public CalibratorsCatalog Calibrators { get; }

        /// <summary>
        /// Catalog which contains different methods to produce calibrators.
        /// </summary>
        public sealed class CalibratorsCatalog : CatalogInstantiatorBase
        {
            internal CalibratorsCatalog(BinaryClassificationCatalog catalog)
                : base(catalog)
            {
            }
            /// <summary>
            /// Adds probability column by training naive binning-based calibrator.
            /// </summary>
            /// <param name="labelColumnName">The name of the label column.</param>
            /// <param name="scoreColumnName">The name of the score column.</param>
            /// <example>
            /// <format type="text/markdown">
            /// <![CDATA[
            /// [!code-csharp[NaiveCalibrator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/BinaryClassification/Calibrators/Naive.cs)]
            /// ]]>
            /// </format>
            /// </example>
            public NaiveCalibratorEstimator Naive(
                  string labelColumnName = DefaultColumnNames.Label,
                  string scoreColumnName = DefaultColumnNames.Score)
            {
                return new NaiveCalibratorEstimator(Owner.GetEnvironment(), labelColumnName, scoreColumnName);
            }
            /// <summary>
            /// Adds probability column by training <a href="https://en.wikipedia.org/wiki/Platt_scaling">platt calibrator</a>.
            /// </summary>
            /// <param name="labelColumnName">The name of the label column.</param>
            /// <param name="scoreColumnName">The name of the score column.</param>
            /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
            /// <example>
            /// <format type="text/markdown">
            /// <![CDATA[
            /// [!code-csharp[PlattCalibrator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/BinaryClassification/Calibrators/Platt.cs)]
            /// ]]>
            /// </format>
            /// </example>
            public PlattCalibratorEstimator Platt(
                  string labelColumnName = DefaultColumnNames.Label,
                  string scoreColumnName = DefaultColumnNames.Score,
                  string exampleWeightColumnName = null)
            {
                return new PlattCalibratorEstimator(Owner.GetEnvironment(), labelColumnName, scoreColumnName, exampleWeightColumnName);
            }

            /// <summary>
            /// Adds probability column by specifying <a href="https://en.wikipedia.org/wiki/Platt_scaling">platt calibrator</a>.
            /// </summary>
            /// <param name="slope">The slope in the function of the exponent of the sigmoid.</param>
            /// <param name="offset">The offset in the function of the exponent of the sigmoid.</param>
            /// <param name="scoreColumnName">The name of the score column.</param>
            /// <example>
            /// <format type="text/markdown">
            /// <![CDATA[
            /// [!code-csharp[FixedPlattCalibrator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/BinaryClassification/Calibrators/FixedPlatt.cs)]
            /// ]]>
            /// </format>
            /// </example>
            public FixedPlattCalibratorEstimator Platt(
                double slope,
                double offset,
                string scoreColumnName = DefaultColumnNames.Score)
            {
                return new FixedPlattCalibratorEstimator(Owner.GetEnvironment(), slope, offset, scoreColumnName);
            }

            /// <summary>
            /// Adds probability column by training pair adjacent violators calibrator.
            /// </summary>
            /// <remarks>
            ///  The calibrator finds a stepwise constant function (using the Pool Adjacent Violators Algorithm aka PAV) that minimizes the squared error.
            ///  Also know as <a href="https://en.wikipedia.org/wiki/Isotonic_regression">Isotonic regression</a>
            /// </remarks>
            /// <param name="labelColumnName">The name of the label column.</param>
            /// <param name="scoreColumnName">The name of the score column.</param>
            /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
            /// <example>
            /// <format type="text/markdown">
            /// <![CDATA[
            /// [!code-csharp[PairAdjacentViolators](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/BinaryClassification/Calibrators/Isotonic.cs)]
            /// ]]>
            /// </format>
            /// </example>
            public IsotonicCalibratorEstimator Isotonic(
                string labelColumnName = DefaultColumnNames.Label,
                string scoreColumnName = DefaultColumnNames.Score,
                string exampleWeightColumnName = null)
            {
                return new IsotonicCalibratorEstimator(Owner.GetEnvironment(), labelColumnName, scoreColumnName, exampleWeightColumnName);
            }
        }
    }

    /// <summary>
    /// The central catalog for clustering tasks and trainers.
    /// </summary>
    public sealed class ClusteringCatalog : TrainCatalogBase
    {
        /// <summary>
        /// The list of trainers for performing clustering.
        /// </summary>
        public ClusteringTrainers Trainers { get; }

        /// <summary>
        /// The clustering context.
        /// </summary>
        internal ClusteringCatalog(IHostEnvironment env)
            : base(env, nameof(ClusteringCatalog))
        {
            Trainers = new ClusteringTrainers(this);
        }

        public sealed class ClusteringTrainers : CatalogInstantiatorBase
        {
            internal ClusteringTrainers(ClusteringCatalog catalog)
                : base(catalog)
            {
            }
        }

        /// <summary>
        /// Evaluates scored clustering data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="scoreColumnName">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="labelColumnName">The name of the optional label column in <paramref name="data"/>.
        /// If present, the <see cref="ClusteringMetrics.NormalizedMutualInformation"/> metric will be computed.</param>
        /// <param name="featureColumnName">The name of the optional features column in <paramref name="data"/>.
        /// If present, the <see cref="ClusteringMetrics.DaviesBouldinIndex"/> metric will be computed.</param>
        /// <returns>The evaluation result.</returns>
        public ClusteringMetrics Evaluate(IDataView data,
            string labelColumnName = null,
            string scoreColumnName = DefaultColumnNames.Score,
            string featureColumnName = null)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(scoreColumnName, nameof(scoreColumnName));

            if (featureColumnName != null)
                Environment.CheckNonEmpty(featureColumnName, nameof(featureColumnName), "The features column name should be non-empty if you want to calculate the Dbi metric.");

            if (labelColumnName != null)
                Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName), "The label column name should be non-empty if you want to calculate the Nmi metric.");

            var eval = new ClusteringEvaluator(Environment, new ClusteringEvaluator.Arguments() { CalculateDbi = !string.IsNullOrEmpty(featureColumnName) });
            return eval.Evaluate(data, scoreColumnName, labelColumnName, featureColumnName);
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numberOfFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="samplingKeyColumnName"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumnName"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numberOfFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumnName">Optional label column for evaluation (clustering tasks may not always have a label).</param>
        /// <param name="featuresColumnName">Optional features column for evaluation (needed for calculating Dbi metric)</param>
        /// <param name="samplingKeyColumnName">Name of a column to use for grouping rows. If two examples share the same value of the <paramref name="samplingKeyColumnName"/>,
        /// they are guaranteed to appear in the same subset (train or test). This can be used to ensure no label leakage from the train to the test set.
        /// If <see langword="null"/> no row grouping will be performed.</param>
        /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
        public CrossValidationResult<ClusteringMetrics>[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numberOfFolds = 5, string labelColumnName = null, string featuresColumnName = null,
            string samplingKeyColumnName = null, int? seed = null)
        {
            var result = CrossValidateTrain(data, estimator, numberOfFolds, samplingKeyColumnName, seed);
            return result.Select(x => new CrossValidationResult<ClusteringMetrics>(x.Model,
                Evaluate(x.Scores, labelColumnName: labelColumnName, featureColumnName: featuresColumnName), x.Scores, x.Fold)).ToArray();
        }
    }

    /// <summary>
    /// The central catalog for multiclass classification tasks and trainers.
    /// </summary>
    public sealed class MulticlassClassificationCatalog : TrainCatalogBase
    {
        /// <summary>
        /// The list of trainers for performing multiclass classification.
        /// </summary>
        public MulticlassClassificationTrainers Trainers { get; }

        internal MulticlassClassificationCatalog(IHostEnvironment env)
            : base(env, nameof(MulticlassClassificationCatalog))
        {
            Trainers = new MulticlassClassificationTrainers(this);
        }

        public sealed class MulticlassClassificationTrainers : CatalogInstantiatorBase
        {
            internal MulticlassClassificationTrainers(MulticlassClassificationCatalog catalog)
                : base(catalog)
            {
            }
        }

        /// <summary>
        /// Evaluates scored multiclass classification data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="labelColumnName">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="scoreColumnName">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="predictedLabelColumnName">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <param name="topK">If given a positive value, the <see cref="MulticlassClassificationMetrics.TopKAccuracy"/> will be filled with
        /// the top-K accuracy, that is, the accuracy assuming we consider an example with the correct class within
        /// the top-K values as being stored "correctly."</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public MulticlassClassificationMetrics Evaluate(IDataView data, string labelColumnName = DefaultColumnNames.Label, string scoreColumnName = DefaultColumnNames.Score,
            string predictedLabelColumnName = DefaultColumnNames.PredictedLabel, int topK = 0)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            Environment.CheckNonEmpty(scoreColumnName, nameof(scoreColumnName));
            Environment.CheckNonEmpty(predictedLabelColumnName, nameof(predictedLabelColumnName));

            var args = new MulticlassClassificationEvaluator.Arguments() { };
            if (topK > 0)
                args.OutputTopKAcc = topK;
            var eval = new MulticlassClassificationEvaluator(Environment, args);
            return eval.Evaluate(data, labelColumnName, scoreColumnName, predictedLabelColumnName);
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numberOfFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="samplingKeyColumnName"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumnName"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numberOfFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumnName">The label column (for evaluation).</param>
        /// <param name="samplingKeyColumnName">Name of a column to use for grouping rows. If two examples share the same value of the <paramref name="samplingKeyColumnName"/>,
        /// they are guaranteed to appear in the same subset (train or test). This can be used to ensure no label leakage from the train to the test set.
        /// If <see langword="null"/> no row grouping will be performed.</param>
        /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public CrossValidationResult<MulticlassClassificationMetrics>[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numberOfFolds = 5, string labelColumnName = DefaultColumnNames.Label,
            string samplingKeyColumnName = null, int? seed = null)
        {
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            var result = CrossValidateTrain(data, estimator, numberOfFolds, samplingKeyColumnName, seed);
            return result.Select(x => new CrossValidationResult<MulticlassClassificationMetrics>(x.Model,
                Evaluate(x.Scores, labelColumnName), x.Scores, x.Fold)).ToArray();
        }
    }

    /// <summary>
    /// The central catalog for regression tasks and trainers.
    /// </summary>
    public sealed class RegressionCatalog : TrainCatalogBase
    {
        /// <summary>
        /// The list of trainers for performing regression.
        /// </summary>
        public RegressionTrainers Trainers { get; }

        internal RegressionCatalog(IHostEnvironment env)
            : base(env, nameof(RegressionCatalog))
        {
            Trainers = new RegressionTrainers(this);
        }

        public sealed class RegressionTrainers : CatalogInstantiatorBase
        {
            internal RegressionTrainers(RegressionCatalog catalog)
                : base(catalog)
            {
            }
        }

        /// <summary>
        /// Evaluates scored regression data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="labelColumnName">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="scoreColumnName">The name of the score column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public RegressionMetrics Evaluate(IDataView data, string labelColumnName = DefaultColumnNames.Label, string scoreColumnName = DefaultColumnNames.Score)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            Environment.CheckNonEmpty(scoreColumnName, nameof(scoreColumnName));

            var eval = new RegressionEvaluator(Environment, new RegressionEvaluator.Arguments() { });
            return eval.Evaluate(data, labelColumnName, scoreColumnName);
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numberOfFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="samplingKeyColumnName"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumnName"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numberOfFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumnName">The label column (for evaluation).</param>
        /// <param name="samplingKeyColumnName">Name of a column to use for grouping rows. If two examples share the same value of the <paramref name="samplingKeyColumnName"/>,
        /// they are guaranteed to appear in the same subset (train or test). This can be used to ensure no label leakage from the train to the test set.
        /// If <see langword="null"/> no row grouping will be performed.</param>
        /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public CrossValidationResult<RegressionMetrics>[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numberOfFolds = 5, string labelColumnName = DefaultColumnNames.Label,
            string samplingKeyColumnName = null, int? seed = null)
        {
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            var result = CrossValidateTrain(data, estimator, numberOfFolds, samplingKeyColumnName, seed);
            return result.Select(x => new CrossValidationResult<RegressionMetrics>(x.Model,
                Evaluate(x.Scores, labelColumnName), x.Scores, x.Fold)).ToArray();
        }
    }

    /// <summary>
    /// The central catalog for ranking tasks and trainers.
    /// </summary>
    public sealed class RankingCatalog : TrainCatalogBase
    {
        /// <summary>
        /// The list of trainers for performing regression.
        /// </summary>
        public RankingTrainers Trainers { get; }

        internal RankingCatalog(IHostEnvironment env)
            : base(env, nameof(RankingCatalog))
        {
            Trainers = new RankingTrainers(this);
        }

        public sealed class RankingTrainers : CatalogInstantiatorBase
        {
            internal RankingTrainers(RankingCatalog catalog)
                : base(catalog)
            {
            }
        }

        /// <summary>
        /// Evaluates scored ranking data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="labelColumnName">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="rowGroupColumnName">The name of the groupId column in <paramref name="data"/>.</param>
        /// <param name="scoreColumnName">The name of the score column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public RankingMetrics Evaluate(IDataView data,
            string labelColumnName = DefaultColumnNames.Label,
            string rowGroupColumnName = DefaultColumnNames.GroupId,
            string scoreColumnName = DefaultColumnNames.Score)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            Environment.CheckNonEmpty(scoreColumnName, nameof(scoreColumnName));
            Environment.CheckNonEmpty(rowGroupColumnName, nameof(rowGroupColumnName));

            var eval = new RankingEvaluator(Environment, new RankingEvaluator.Arguments() { });
            return eval.Evaluate(data, labelColumnName, rowGroupColumnName, scoreColumnName);
        }
    }

    /// <summary>
    /// The central catalog for anomaly detection tasks and trainers.
    /// </summary>
    public sealed class AnomalyDetectionCatalog : TrainCatalogBase
    {
        /// <summary>
        /// The list of trainers for anomaly detection.
        /// </summary>
        public AnomalyDetectionTrainers Trainers { get; }

        internal AnomalyDetectionCatalog(IHostEnvironment env)
            : base(env, nameof(AnomalyDetectionCatalog))
        {
            Trainers = new AnomalyDetectionTrainers(this);
        }

        public sealed class AnomalyDetectionTrainers : CatalogInstantiatorBase
        {
            internal AnomalyDetectionTrainers(AnomalyDetectionCatalog catalog)
                : base(catalog)
            {
            }
        }

        /// <summary>
        /// Evaluates scored anomaly detection data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="labelColumnName">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="scoreColumnName">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="predictedLabelColumnName">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <param name="k">The number of false positives to compute the <see cref="AnomalyDetectionMetrics.DetectionRateAtKFalsePositives"/> metric. </param>
        /// <returns>Evaluation results.</returns>
        public AnomalyDetectionMetrics Evaluate(IDataView data, string labelColumnName = DefaultColumnNames.Label, string scoreColumnName = DefaultColumnNames.Score,
            string predictedLabelColumnName = DefaultColumnNames.PredictedLabel, int k = 10)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(labelColumnName, nameof(labelColumnName));
            Environment.CheckNonEmpty(scoreColumnName, nameof(scoreColumnName));
            Environment.CheckNonEmpty(predictedLabelColumnName, nameof(predictedLabelColumnName));

            var args = new AnomalyDetectionEvaluator.Arguments();
            args.K = k;

            var eval = new AnomalyDetectionEvaluator(Environment, args);
            return eval.Evaluate(data, labelColumnName, scoreColumnName, predictedLabelColumnName);
        }
    }
}
