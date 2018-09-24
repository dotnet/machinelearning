// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;

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
            Host.CheckParam(0 < testFraction && testFraction < 1, nameof(testFraction), "Must be between 0 and 1");
            Host.CheckValueOrNull(stratificationColumn);

            // We need to handle two cases: if the stratification column is provided, we use hashJoin to
            // build a single hash of it. If it is not, we generate a random number.

            if (stratificationColumn != null)
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

        protected TrainContextBase(IHostEnvironment env, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(registrationName, nameof(registrationName));
            Host = env.Register(registrationName);
        }

        /// <summary>
        /// Subclasses of <see cref="TrainContext"/> will provide little "extension method" hookable objects
        /// (e.g., something like <see cref="BinaryClassificationContext.Trainers"/>). User code will only
        /// interact with these objects by invoking the extension methods. The actual component code can work
        /// through <see cref="TrainContextComponentUtils"/> to get more "hidden" information from this object,
        /// e.g., the environment.
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
        /// <remarks>
        /// Component authors that have written binary classification. They are great people.
        /// </remarks>
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
    }
}
