// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Evaluators.Metrics;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;

namespace Microsoft.ML
{
    /// <summary>
    /// A training catalog is an object instantiable by a user to do various tasks relating to a particular
    /// "area" of machine learning. A subclass would represent a particular task in machine learning. The idea
    /// is that a user can instantiate that particular area, and get trainers and evaluators.
    /// </summary>
    public abstract class TrainCatalogBase
    {
        [BestFriend]
        internal IHostEnvironment Environment { get; }

        /// <summary>
        /// A pair of datasets, for the train and test set.
        /// </summary>
        public struct TrainTestData
        {
            /// <summary>
            /// Training set.
            /// </summary>
            public readonly IDataView TrainSet;
            /// <summary>
            /// Testing set.
            /// </summary>
            public readonly IDataView TestSet;
            /// <summary>
            /// Create pair of datasets.
            /// </summary>
            /// <param name="trainSet">Training set.</param>
            /// <param name="testSet">Testing set.</param>
            internal TrainTestData(IDataView trainSet, IDataView testSet)
            {
                TrainSet = trainSet;
                TestSet = testSet;
            }
        }

        /// <summary>
        /// Split the dataset into the train set and test set according to the given fraction.
        /// Respects the <paramref name="stratificationColumn"/> if provided.
        /// </summary>
        /// <param name="data">The dataset to split.</param>
        /// <param name="testFraction">The fraction of data to go into the test set.</param>
        /// <param name="stratificationColumn">Optional name of the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        public TrainTestData TrainTestSplit(IDataView data, double testFraction = 0.1, string stratificationColumn = null, uint? seed = null)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckParam(0 < testFraction && testFraction < 1, nameof(testFraction), "Must be between 0 and 1 exclusive");
            Environment.CheckValueOrNull(stratificationColumn);

            EnsureStratificationColumn(ref data, ref stratificationColumn, seed);

            var trainFilter = new RangeFilter(Environment, new RangeFilter.Options()
            {
                Column = stratificationColumn,
                Min = 0,
                Max = testFraction,
                Complement = true
            }, data);
            var testFilter = new RangeFilter(Environment, new RangeFilter.Options()
            {
                Column = stratificationColumn,
                Min = 0,
                Max = testFraction,
                Complement = false
            }, data);

            return new TrainTestData(trainFilter, testFilter);
        }

        /// <summary>
        /// Results for specific cross-validation fold.
        /// </summary>
        protected internal struct CrossValidationResult
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
        protected internal CrossValidationResult[] CrossValidateTrain(IDataView data, IEstimator<ITransformer> estimator,
            int numFolds, string stratificationColumn, uint? seed = null)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckValue(estimator, nameof(estimator));
            Environment.CheckParam(numFolds > 1, nameof(numFolds), "Must be more than 1");
            Environment.CheckValueOrNull(stratificationColumn);

            EnsureStratificationColumn(ref data, ref stratificationColumn, seed);

            Func<int, CrossValidationResult> foldFunction =
                fold =>
                {
                    var trainFilter = new RangeFilter(Environment, new RangeFilter.Options
                    {
                        Column = stratificationColumn,
                        Min = (double)fold / numFolds,
                        Max = (double)(fold + 1) / numFolds,
                        Complement = true
                    }, data);
                    var testFilter = new RangeFilter(Environment, new RangeFilter.Options
                    {
                        Column = stratificationColumn,
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
        /// Make sure the provided <paramref name="stratificationColumn"/> is valid
        /// for <see cref="RangeFilter"/>, hash it if needed, or introduce a new one
        /// if needed.
        /// </summary>
        private void EnsureStratificationColumn(ref IDataView data, ref string stratificationColumn, uint? seed = null)
        {
            // We need to handle two cases: if the stratification column is provided, we use hashJoin to
            // build a single hash of it. If it is not, we generate a random number.

            if (stratificationColumn == null)
            {
                stratificationColumn = data.Schema.GetTempColumnName("StratificationColumn");
                data = new GenerateNumberTransform(Environment, data, stratificationColumn, seed);
            }
            else
            {
                if (!data.Schema.TryGetColumnIndex(stratificationColumn, out int stratCol))
                    throw Environment.ExceptSchemaMismatch(nameof(stratificationColumn), "stratification", stratificationColumn);

                var type = data.Schema[stratCol].Type;
                if (!RangeFilter.IsValidRangeFilterColumnType(Environment, type))
                {
                    // Hash the stratification column.
                    // REVIEW: this could currently crash, since Hash only accepts a limited set
                    // of column types. It used to be HashJoin, but we should probably extend Hash
                    // instead of having two hash transformations.
                    var origStratCol = stratificationColumn;
                    int tmp;
                    int inc = 0;

                    // Generate a new column with the hashed stratification column.
                    while (data.Schema.TryGetColumnIndex(stratificationColumn, out tmp))
                        stratificationColumn = string.Format("{0}_{1:000}", origStratCol, ++inc);
                    HashingEstimator.ColumnInfo columnInfo;
                    if (seed.HasValue)
                        columnInfo = new HashingEstimator.ColumnInfo(stratificationColumn, origStratCol, 30, seed.Value);
                    else
                        columnInfo = new HashingEstimator.ColumnInfo(stratificationColumn, origStratCol, 30);
                    data = new HashingEstimator(Environment, columnInfo).Fit(data).Transform(data);
                }
            }
        }

        /// <summary>
        /// Subclasses of <see cref="TrainContext"/> will provide little "extension method" hookable objects
        /// (for example, something like <see cref="BinaryClassificationCatalog.Trainers"/>). User code will only
        /// interact with these objects by invoking the extension methods. The actual component code can work
        /// through <see cref="CatalogUtils"/> to get more "hidden" information from this object,
        /// for example, the environment.
        /// </summary>
        public abstract class CatalogInstantiatorBase
        {
            [BestFriend]
            internal TrainCatalogBase Owner { get; }

            internal protected CatalogInstantiatorBase(TrainCatalogBase catalog)
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
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="probability">The name of the probability column in <paramref name="data"/>, the calibrated version of <paramref name="score"/>.</param>
        /// <param name="predictedLabel">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public CalibratedBinaryClassificationMetrics Evaluate(IDataView data, string label = DefaultColumnNames.Label, string score = DefaultColumnNames.Score,
            string probability = DefaultColumnNames.Probability, string predictedLabel = DefaultColumnNames.PredictedLabel)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(label, nameof(label));
            Environment.CheckNonEmpty(score, nameof(score));
            Environment.CheckNonEmpty(probability, nameof(probability));
            Environment.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var eval = new BinaryClassifierEvaluator(Environment, new BinaryClassifierEvaluator.Arguments() { });
            return eval.Evaluate(data, label, score, probability, predictedLabel);
        }

        /// <summary>
        /// Evaluates scored binary classification data, without probability-based metrics.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="predictedLabel">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these uncalibrated outputs.</returns>
        public BinaryClassificationMetrics EvaluateNonCalibrated(IDataView data, string label = DefaultColumnNames.Label, string score = DefaultColumnNames.Score,
            string predictedLabel = DefaultColumnNames.PredictedLabel)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(label, nameof(label));
            Environment.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var eval = new BinaryClassifierEvaluator(Environment, new BinaryClassifierEvaluator.Arguments() { });
            return eval.Evaluate(data, label, score, predictedLabel);
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="stratificationColumn"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumn"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumn">The label column (for evaluation).</param>
        /// <param name="stratificationColumn">Optional name of the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public CrossValidationResult<BinaryClassificationMetrics>[] CrossValidateNonCalibrated(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = DefaultColumnNames.Label,
            string stratificationColumn = null, uint? seed = null)
        {
            Environment.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn, seed);
            return result.Select(x => new CrossValidationResult<BinaryClassificationMetrics>(x.Model,
                EvaluateNonCalibrated(x.Scores, labelColumn), x.Scores, x.Fold)).ToArray();
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="stratificationColumn"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumn"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumn">The label column (for evaluation).</param>
        /// <param name="stratificationColumn">Optional stratification column.</param>
        /// <remarks>If two examples share the same value of the <paramref name="stratificationColumn"/> (if provided),
        /// they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from
        /// train to the test set.</remarks>
        /// <param name="seed">If <paramref name="stratificationColumn"/> not present in dataset we will generate random filled column based on provided <paramref name="seed"/>.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public CrossValidationResult<CalibratedBinaryClassificationMetrics>[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = DefaultColumnNames.Label,
            string stratificationColumn = null, uint? seed = null)
        {
            Environment.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn, seed);
            return result.Select(x => new CrossValidationResult<CalibratedBinaryClassificationMetrics>(x.Model,
                Evaluate(x.Scores, labelColumn), x.Scores, x.Fold)).ToArray();
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
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="label">The name of the optional label column in <paramref name="data"/>.
        /// If present, the <see cref="ClusteringMetrics.Nmi"/> metric will be computed.</param>
        /// <param name="features">The name of the optional features column in <paramref name="data"/>.
        /// If present, the <see cref="ClusteringMetrics.Dbi"/> metric will be computed.</param>
        /// <returns>The evaluation result.</returns>
        public ClusteringMetrics Evaluate(IDataView data,
            string label = null,
            string score = DefaultColumnNames.Score,
            string features = null)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(score, nameof(score));

            if (features != null)
                Environment.CheckNonEmpty(features, nameof(features), "The features column name should be non-empty if you want to calculate the Dbi metric.");

            if (label != null)
                Environment.CheckNonEmpty(label, nameof(label), "The label column name should be non-empty if you want to calculate the Nmi metric.");

            var eval = new ClusteringEvaluator(Environment, new ClusteringEvaluator.Arguments() { CalculateDbi = !string.IsNullOrEmpty(features) });
            return eval.Evaluate(data, score, label, features);
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="stratificationColumn"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumn"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumn">Optional label column for evaluation (clustering tasks may not always have a label).</param>
        /// <param name="featuresColumn">Optional features column for evaluation (needed for calculating Dbi metric)</param>
        /// <param name="stratificationColumn">Optional name of the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public CrossValidationResult<ClusteringMetrics>[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = null, string featuresColumn = null,
            string stratificationColumn = null, uint? seed = null)
        {
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn, seed);
            return result.Select(x => new CrossValidationResult<ClusteringMetrics>(x.Model,
                Evaluate(x.Scores, label: labelColumn, features: featuresColumn), x.Scores, x.Fold)).ToArray();
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
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="predictedLabel">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <param name="topK">If given a positive value, the <see cref="MultiClassClassifierMetrics.TopKAccuracy"/> will be filled with
        /// the top-K accuracy, that is, the accuracy assuming we consider an example with the correct class within
        /// the top-K values as being stored "correctly."</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public MultiClassClassifierMetrics Evaluate(IDataView data, string label = DefaultColumnNames.Label, string score = DefaultColumnNames.Score,
            string predictedLabel = DefaultColumnNames.PredictedLabel, int topK = 0)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(label, nameof(label));
            Environment.CheckNonEmpty(score, nameof(score));
            Environment.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var args = new MultiClassClassifierEvaluator.Arguments() { };
            if (topK > 0)
                args.OutputTopKAcc = topK;
            var eval = new MultiClassClassifierEvaluator(Environment, args);
            return eval.Evaluate(data, label, score, predictedLabel);
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="stratificationColumn"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumn"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumn">The label column (for evaluation).</param>
        /// <param name="stratificationColumn">Optional name of the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public CrossValidationResult<MultiClassClassifierMetrics>[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = DefaultColumnNames.Label,
            string stratificationColumn = null, uint? seed = null)
        {
            Environment.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn, seed);
            return result.Select(x => new CrossValidationResult<MultiClassClassifierMetrics>(x.Model,
                Evaluate(x.Scores, labelColumn), x.Scores, x.Fold)).ToArray();
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
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public RegressionMetrics Evaluate(IDataView data, string label = DefaultColumnNames.Label, string score = DefaultColumnNames.Score)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(label, nameof(label));
            Environment.CheckNonEmpty(score, nameof(score));

            var eval = new RegressionEvaluator(Environment, new RegressionEvaluator.Arguments() { });
            return eval.Evaluate(data, label, score);
        }

        /// <summary>
        /// Run cross-validation over <paramref name="numFolds"/> folds of <paramref name="data"/>, by fitting <paramref name="estimator"/>,
        /// and respecting <paramref name="stratificationColumn"/> if provided.
        /// Then evaluate each sub-model against <paramref name="labelColumn"/> and return metrics.
        /// </summary>
        /// <param name="data">The data to run cross-validation on.</param>
        /// <param name="estimator">The estimator to fit.</param>
        /// <param name="numFolds">Number of cross-validation folds.</param>
        /// <param name="labelColumn">The label column (for evaluation).</param>
        /// <param name="stratificationColumn">Optional name of the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public CrossValidationResult<RegressionMetrics>[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = DefaultColumnNames.Label,
            string stratificationColumn = null, uint? seed = null)
        {
            Environment.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn, seed);
            return result.Select(x => new CrossValidationResult<RegressionMetrics>(x.Model,
                Evaluate(x.Scores, labelColumn), x.Scores, x.Fold)).ToArray();
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
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="groupId">The name of the groupId column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public RankerMetrics Evaluate(IDataView data, string label, string groupId, string score = DefaultColumnNames.Score)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(label, nameof(label));
            Environment.CheckNonEmpty(score, nameof(score));
            Environment.CheckNonEmpty(groupId, nameof(groupId));

            var eval = new RankerEvaluator(Environment, new RankerEvaluator.Arguments() { });
            return eval.Evaluate(data, label, groupId, score);
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
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="predictedLabel">The name of the predicted label column in <paramref name="data"/>.</param>
        /// <param name="k">The number of false positives to compute the <see cref="AnomalyDetectionMetrics.DrAtK"/> metric. </param>
        /// <returns>Evaluation results.</returns>
        public AnomalyDetectionMetrics Evaluate(IDataView data, string label = DefaultColumnNames.Label, string score = DefaultColumnNames.Score,
            string predictedLabel = DefaultColumnNames.PredictedLabel, int k = 10)
        {
            Environment.CheckValue(data, nameof(data));
            Environment.CheckNonEmpty(label, nameof(label));
            Environment.CheckNonEmpty(score, nameof(score));
            Environment.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var args = new AnomalyDetectionEvaluator.Arguments();
            args.K = k;

            var eval = new AnomalyDetectionEvaluator(Environment, args);
            return eval.Evaluate(data, label, score, predictedLabel);
        }
    }
}
