// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Microsoft.ML
{
    public static class RecommenderCatalog
    {

        /// <summary>
        /// Trainers and tasks specific to ranking problems.
        /// </summary>
        public static RecommendationContext Recommendation(this MLContext ctx) => new RecommendationContext(ctx);
    }

    /// <summary>
    /// The central context for regression trainers.
    /// </summary>
    public sealed class RecommendationContext : TrainContextBase
    {
        /// <summary>
        /// For trainers for performing regression.
        /// </summary>
        public RecommendationTrainers Trainers { get; }

        public RecommendationContext(IHostEnvironment env)
            : base(env, nameof(RecommendationContext))
        {
            Trainers = new RecommendationTrainers(this);
        }

        public sealed class RecommendationTrainers : ContextInstantiatorBase
        {
            internal RecommendationTrainers(RecommendationContext ctx)
                : base(ctx)
            {
            }

            /// <summary>
            /// Train a matrix factorization model. It factorizes the training matrix into the product of two low-rank matrices.
            /// </summary>
            /// <remarks>
            /// <para>The basic idea of matrix factorization is finding two low-rank factor marcies to apporimate the training matrix.</para>
            /// <para>In this module, the expected training data is a list of tuples. Every tuple consists of a column index, a row index,
            /// and the value at the location specified by the two indexes.
            /// </para>
            /// </remarks>
            /// <param name="matrixColumnIndexColumnName">The name of the column hosting the matrix's column IDs.</param>
            /// <param name="matrixRowIndexColumnName">The name of the column hosting the matrix's row IDs.</param>
            /// <param name="labelColumn">The name of the label column.</param>
            /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
            public MatrixFactorizationTrainer MatrixFactorization(
                string matrixColumnIndexColumnName,
                string matrixRowIndexColumnName,
                string labelColumn = DefaultColumnNames.Label,
                Action<MatrixFactorizationTrainer.Arguments> advancedSettings = null)
                    => new MatrixFactorizationTrainer(Owner.Environment, matrixColumnIndexColumnName, matrixRowIndexColumnName, labelColumn, advancedSettings);
        }

        /// <summary>
        /// Evaluates the scored recommendation data.
        /// </summary>
        /// <param name="data">The scored data.</param>
        /// <param name="label">The name of the label column in <paramref name="data"/>.</param>
        /// <param name="score">The name of the score column in <paramref name="data"/>.</param>
        /// <returns>The evaluation results for these calibrated outputs.</returns>
        public RegressionMetrics Evaluate(IDataView data, string label = DefaultColumnNames.Label, string score = DefaultColumnNames.Score)
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
        /// <param name="stratificationColumn">Optional name of the column to use as a stratification column. If two examples share the same value of the <paramref name="stratificationColumn"/>
        /// (if provided), they are guaranteed to appear in the same subset (train or test). Use this to make sure there is no label leakage from train to the test set.
        /// If this optional parameter is not provided, a stratification columns will be generated, and its values will be random numbers .</param>
        /// <param name="seed">Optional parameter used in combination with the <paramref name="stratificationColumn"/>.
        /// If the <paramref name="stratificationColumn"/> is not provided, the random numbers generated to create it, will use this seed as value.
        /// And if it is not provided, the default value will be used.</param>
        /// <returns>Per-fold results: metrics, models, scored datasets.</returns>
        public (RegressionMetrics metrics, ITransformer model, IDataView scoredTestData)[] CrossValidate(
            IDataView data, IEstimator<ITransformer> estimator, int numFolds = 5, string labelColumn = DefaultColumnNames.Label,
            string stratificationColumn = null, uint? seed = null)
        {
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn, seed);
            return result.Select(x => (Evaluate(x.scoredTestSet, labelColumn), x.model, x.scoredTestSet)).ToArray();
        }
    }
}
