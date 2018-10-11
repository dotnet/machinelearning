// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML
{
    /// <summary>
    /// A training context is an object instantiable by a user to do various tasks relating to a particular
    /// "area" of machine learning. A subclass would represent a particular task in machine learning. The idea
    /// is that a user can instantiate that particular area, and get trainers and evaluators.
    /// </summary>
    public abstract class TrainContextBase
    {
        protected readonly IHost Host;
        internal IHostEnvironment Environment => Host;

        /// <summary>
        /// Split the dataset into the train set and test set according to the given fraction.
        /// Respects the <paramref name="stratificationColumn"/> if provided.
        /// </summary>
        /// <param name="data">The dataset to split.</param>
        /// <param name="testFraction">The fraction of data to go into the test set.</param>
        /// <param name="stratificationColumn">Optional stratification column.</param>
        /// <remarks>If two examples share the same value of the <paramref name="stratificationColumn"/> (if provided),
        /// they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from
        /// train to the test set.</remarks>
        /// <returns>A pair of datasets, for the train and test set.</returns>
        public (IDataView trainSet, IDataView testSet) TrainTestSplit(IDataView data, double testFraction = 0.1, string stratificationColumn = null)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckParam(0 < testFraction && testFraction < 1, nameof(testFraction), "Must be between 0 and 1 exclusive");
            Host.CheckValueOrNull(stratificationColumn);

            EnsureStratificationColumn(ref data, ref stratificationColumn);

            var trainFilter = new RangeFilter(Host, new RangeFilter.Arguments()
            {
                Column = stratificationColumn,
                Min = 0,
                Max = testFraction,
                Complement = true
            }, data);
            var testFilter = new RangeFilter(Host, new RangeFilter.Arguments()
            {
                Column = stratificationColumn,
                Min = 0,
                Max = testFraction,
                Complement = false
            }, data);

            return (trainFilter, testFilter);
        }

        /// <summary>
        /// Train the <paramref name="estimator"/> on <paramref name="numFolds"/> folds of the data sequentially.
        /// Return each model and each scored test dataset.
        /// </summary>
        protected (IDataView scoredTestSet, ITransformer model)[] CrossValidateTrain(IDataView data, IEstimator<ITransformer> estimator,
            int numFolds, string stratificationColumn)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckValue(estimator, nameof(estimator));
            Host.CheckParam(numFolds > 1, nameof(numFolds), "Must be more than 1");
            Host.CheckValueOrNull(stratificationColumn);

            EnsureStratificationColumn(ref data, ref stratificationColumn);

            Func<int, (IDataView scores, ITransformer model)> foldFunction =
                fold =>
                {
                    var trainFilter = new RangeFilter(Host, new RangeFilter.Arguments
                    {
                        Column = stratificationColumn,
                        Min = (double)fold / numFolds,
                        Max = (double)(fold + 1) / numFolds,
                        Complement = true
                    }, data);
                    var testFilter = new RangeFilter(Host, new RangeFilter.Arguments
                    {
                        Column = stratificationColumn,
                        Min = (double)fold / numFolds,
                        Max = (double)(fold + 1) / numFolds,
                        Complement = false
                    }, data);

                    var model = estimator.Fit(trainFilter);
                    var scoredTest = model.Transform(testFilter);
                    return (scoredTest, model);
                };

            // Sequential per-fold training.
            // REVIEW: we could have a parallel implementation here. We would need to
            // spawn off a separate host per fold in that case.
            var result = new List<(IDataView scores, ITransformer model)>();
            for (int fold = 0; fold < numFolds; fold++)
                result.Add(foldFunction(fold));

            return result.ToArray();
        }

        protected TrainContextBase(IHostEnvironment env, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(registrationName, nameof(registrationName));
            Host = env.Register(registrationName);
        }

        /// <summary>
        /// Make sure the provided <paramref name="stratificationColumn"/> is valid
        /// for <see cref="RangeFilter"/>, hash it if needed, or introduce a new one
        /// if needed.
        /// </summary>
        private void EnsureStratificationColumn(ref IDataView data, ref string stratificationColumn)
        {
            // We need to handle two cases: if the stratification column is provided, we use hashJoin to
            // build a single hash of it. If it is not, we generate a random number.

            if (stratificationColumn == null)
            {
                stratificationColumn = data.Schema.GetTempColumnName("StratificationColumn");
                data = new GenerateNumberTransform(Host, data, stratificationColumn);
            }
            else
            {
                if (!data.Schema.TryGetColumnIndex(stratificationColumn, out int stratCol))
                    throw Host.ExceptSchemaMismatch(nameof(stratificationColumn), "stratification", stratificationColumn);

                var type = data.Schema.GetColumnType(stratCol);
                if (!RangeFilter.IsValidRangeFilterColumnType(Host, type))
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
                    data = new HashEstimator(Host, origStratCol, stratificationColumn, 30).Fit(data).Transform(data);
                }
            }
        }

        /// <summary>
        /// Subclasses of <see cref="TrainContext"/> will provide little "extension method" hookable objects
        /// (for example, something like <see cref="BinaryClassificationContext.Trainers"/>). User code will only
        /// interact with these objects by invoking the extension methods. The actual component code can work
        /// through <see cref="TrainContextComponentUtils"/> to get more "hidden" information from this object,
        /// for example, the environment.
        /// </summary>
        public abstract class ContextInstantiatorBase
        {
            internal TrainContextBase Owner { get; }

            protected ContextInstantiatorBase(TrainContextBase ctx)
            {
                Owner = ctx;
            }
        }
    }

    /// <summary>
    /// Utilities for component authors that want to be able to instantiate components using these context
    /// objects. These utilities are not hidden from non-component authoring users per see, but are at least
    /// registered somewhat less obvious so that they are not confused by the presence.
    /// </summary>
    /// <seealso cref="TrainContextBase"/>
    public static class TrainContextComponentUtils
    {
        /// <summary>
        /// Gets the environment hidden within the instantiator's context.
        /// </summary>
        /// <param name="obj">The extension method hook object for a context.</param>
        /// <returns>An environment that can be used when instantiating components.</returns>
        public static IHostEnvironment GetEnvironment(TrainContextBase.ContextInstantiatorBase obj)
        {
            Contracts.CheckValue(obj, nameof(obj));
            return obj.Owner.Environment;
        }

        /// <summary>
        /// Gets the environment hidden within the context.
        /// </summary>
        /// <param name="ctx">The context.</param>
        /// <returns>An environment that can be used when instantiating components.</returns>
        public static IHostEnvironment GetEnvironment(TrainContextBase ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            return ctx.Environment;
        }
    }

    /// <summary>
    /// The central context for binary classification trainers.
    /// </summary>
    public sealed class BinaryClassificationContext : TrainContextBase
    {
        /// <summary>
        /// For trainers for performing binary classification.
        /// </summary>
        public BinaryClassificationTrainers Trainers { get; }

        public BinaryClassificationContext(IHostEnvironment env)
            : base(env, nameof(BinaryClassificationContext))
        {
            Trainers = new BinaryClassificationTrainers(this);
        }

        public sealed class BinaryClassificationTrainers : ContextInstantiatorBase
        {
            internal BinaryClassificationTrainers(BinaryClassificationContext ctx)
                : base(ctx)
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
        public BinaryClassifierEvaluator.CalibratedResult Evaluate(IDataView data, string label, string score = DefaultColumnNames.Score,
            string probability = DefaultColumnNames.Probability, string predictedLabel = DefaultColumnNames.PredictedLabel)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));
            Host.CheckNonEmpty(probability, nameof(probability));
            Host.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var eval = new BinaryClassifierEvaluator(Host, new BinaryClassifierEvaluator.Arguments() { });
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
        public BinaryClassifierEvaluator.Result EvaluateNonCalibrated(IDataView data, string label, string score = DefaultColumnNames.Score,
            string predictedLabel = DefaultColumnNames.PredictedLabel)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));
            Host.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var eval = new BinaryClassifierEvaluator(Host, new BinaryClassifierEvaluator.Arguments() { });
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
        /// <param name="stratificationColumn">Optional stratification column.</param>
        /// <remarks>If two examples share the same value of the <paramref name="stratificationColumn"/> (if provided),
        /// they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from
        /// train to the test set.</remarks>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public (BinaryClassifierEvaluator.Result metrics, ITransformer model, IDataView scoredTestData)[] CrossValidateNonCalibrated(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = DefaultColumnNames.Label, string stratificationColumn = null)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn);
            return result.Select(x => (EvaluateNonCalibrated(x.scoredTestSet, labelColumn), x.model, x.scoredTestSet)).ToArray();
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
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public (BinaryClassifierEvaluator.CalibratedResult metrics, ITransformer model, IDataView scoredTestData)[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = DefaultColumnNames.Label, string stratificationColumn = null)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn);
            return result.Select(x => (Evaluate(x.scoredTestSet, labelColumn), x.model, x.scoredTestSet)).ToArray();
        }
    }

    /// <summary>
    /// The central context for clustering trainers.
    /// </summary>
    public sealed class ClusteringContext : TrainContextBase
    {
        /// <summary>
        /// List of trainers for performing clustering.
        /// </summary>
        public ClusteringTrainers Trainers { get; }

        /// <summary>
        /// The clustering context.
        /// </summary>
        public ClusteringContext(IHostEnvironment env)
            : base(env, nameof(ClusteringContext))
        {
            Trainers = new ClusteringTrainers(this);
        }

        public sealed class ClusteringTrainers : ContextInstantiatorBase
        {
            internal ClusteringTrainers(ClusteringContext ctx)
                : base(ctx)
            {
            }
        }

        /// <summary>
        /// Evaluates scored clustering data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <param name="label">The name of the optional label column in <paramref name="data"/>.
        /// If present, the <see cref="ClusteringEvaluator.Result.Nmi"/> metric will be computed.</param>
        /// <param name="features">The name of the optional features column in <paramref name="data"/>.
        /// If present, the <see cref="ClusteringEvaluator.Result.Dbi"/> metric will be computed.</param>
        /// <returns>The evaluation result.</returns>
        public ClusteringEvaluator.Result Evaluate(IDataView data,
            string label = null,
            string score = DefaultColumnNames.Score,
            string features = null )
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(score, nameof(score));

            if(features != null)
                Host.CheckNonEmpty(features, nameof(features), "The features column name should be non-empty, if provided, if you want to calculate the Dbi metric.");

            if (label != null)
                Host.CheckNonEmpty(label, nameof(label), "The features column name should be non-empty, if provided, if you want to calculate the Nmi metric.");

            var eval = new ClusteringEvaluator(Host, new ClusteringEvaluator.Arguments() { CalculateDbi = !string.IsNullOrEmpty(features) });
            return eval.Evaluate(data, score, label, features);
        }
    }

    /// <summary>
    /// The central context for multiclass classification trainers.
    /// </summary>
    public sealed class MulticlassClassificationContext : TrainContextBase
    {
        /// <summary>
        /// For trainers for performing multiclass classification.
        /// </summary>
        public MulticlassClassificationTrainers Trainers { get; }

        public MulticlassClassificationContext(IHostEnvironment env)
            : base(env, nameof(MulticlassClassificationContext))
        {
            Trainers = new MulticlassClassificationTrainers(this);
        }

        public sealed class MulticlassClassificationTrainers : ContextInstantiatorBase
        {
            internal MulticlassClassificationTrainers(MulticlassClassificationContext ctx)
                : base(ctx)
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
        /// <param name="topK">If given a positive value, the <see cref="MultiClassClassifierEvaluator.Result.TopKAccuracy"/> will be filled with
        /// the top-K accuracy, that is, the accuracy assuming we consider an example with the correct class within
        /// the top-K values as being stored "correctly."</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public MultiClassClassifierEvaluator.Result Evaluate(IDataView data, string label, string score = DefaultColumnNames.Score,
            string predictedLabel = DefaultColumnNames.PredictedLabel, int topK = 0)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));
            Host.CheckNonEmpty(predictedLabel, nameof(predictedLabel));

            var args = new MultiClassClassifierEvaluator.Arguments() { };
            if (topK > 0)
                args.OutputTopKAcc = topK;
            var eval = new MultiClassClassifierEvaluator(Host, args);
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
        /// <param name="stratificationColumn">Optional stratification column.</param>
        /// <remarks>If two examples share the same value of the <paramref name="stratificationColumn"/> (if provided),
        /// they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from
        /// train to the test set.</remarks>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public (MultiClassClassifierEvaluator.Result metrics, ITransformer model, IDataView scoredTestData)[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = DefaultColumnNames.Label, string stratificationColumn = null)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn);
            return result.Select(x => (Evaluate(x.scoredTestSet, labelColumn), x.model, x.scoredTestSet)).ToArray();
        }
    }

    /// <summary>
    /// The central context for regression trainers.
    /// </summary>
    public sealed class RegressionContext : TrainContextBase
    {
        /// <summary>
        /// For trainers for performing regression.
        /// </summary>
        public RegressionTrainers Trainers { get; }

        public RegressionContext(IHostEnvironment env)
            : base(env, nameof(RegressionContext))
        {
            Trainers = new RegressionTrainers(this);
        }

        public sealed class RegressionTrainers : ContextInstantiatorBase
        {
            internal RegressionTrainers(RegressionContext ctx)
                : base(ctx)
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
        public RegressionEvaluator.Result Evaluate(IDataView data, string label, string score = DefaultColumnNames.Score)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));

            var eval = new RegressionEvaluator(Host, new RegressionEvaluator.Arguments() { });
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
        /// <param name="stratificationColumn">Optional stratification column.</param>
        /// <remarks>If two examples share the same value of the <paramref name="stratificationColumn"/> (if provided),
        /// they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from
        /// train to the test set.</remarks>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public (RegressionEvaluator.Result metrics, ITransformer model, IDataView scoredTestData)[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = DefaultColumnNames.Label, string stratificationColumn = null)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn);
            return result.Select(x => (Evaluate(x.scoredTestSet, labelColumn), x.model, x.scoredTestSet)).ToArray();
        }
    }

    /// <summary>
    /// The central context for regression trainers.
    /// </summary>
    public sealed class RankerContext : TrainContextBase
    {
        /// <summary>
        /// For trainers for performing regression.
        /// </summary>
        public RankerTrainers Trainers { get; }

        public RankerContext(IHostEnvironment env)
            : base(env, nameof(RankerContext))
        {
            Trainers = new RankerTrainers(this);
        }

        public sealed class RankerTrainers : ContextInstantiatorBase
        {
            internal RankerTrainers(RankerContext ctx)
                : base(ctx)
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
        public RankerEvaluator.Result Evaluate(IDataView data, string label, string groupId, string score = DefaultColumnNames.Score)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckNonEmpty(label, nameof(label));
            Host.CheckNonEmpty(score, nameof(score));
            Host.CheckNonEmpty(groupId, nameof(groupId));

            var eval = new RankerEvaluator(Host, new RankerEvaluator.Arguments() { });
            return eval.Evaluate(data, label, groupId, score);
        }
    }
}
