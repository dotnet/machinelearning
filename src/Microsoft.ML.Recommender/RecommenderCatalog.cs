﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML
{
    public static class RecommenderCatalog
    {

        /// <summary>
        /// Trainers and tasks specific to recommendation problems.
        /// </summary>
        public static RecommendationCatalog Recommendation(this MLContext ctx) => new RecommendationCatalog(ctx);
    }

    /// <summary>
    /// The central catalog for recommendation trainers and tasks.
    /// </summary>
    public sealed class RecommendationCatalog : TrainCatalogBase
    {
        /// <summary>
        /// The list of trainers for performing recommendation.
        /// </summary>
        public RecommendationTrainers Trainers { get; }

        internal RecommendationCatalog(IHostEnvironment env)
            : base(env, nameof(RecommendationCatalog))
        {
            Trainers = new RecommendationTrainers(this);
        }

        public sealed class RecommendationTrainers : CatalogInstantiatorBase
        {
            internal RecommendationTrainers(RecommendationCatalog catalog)
                : base(catalog)
            {
            }

            /// <summary>
            /// Train a matrix factorization model. It factorizes the training matrix into the product of two low-rank matrices.
            /// </summary>
            /// <remarks>
            /// <para>The basic idea of matrix factorization is finding two low-rank factor matrices to apporimate the training matrix.</para>
            /// <para>In this module, the expected training data is a list of tuples. Every tuple consists of a column index, a row index,
            /// and the value at the location specified by the two indexes.
            /// </para>
            /// </remarks>
            /// <param name="labelColumn">The name of the label column.</param>
            /// <param name="matrixColumnIndexColumnName">The name of the column hosting the matrix's column IDs.</param>
            /// <param name="matrixRowIndexColumnName">The name of the column hosting the matrix's row IDs.</param>
            /// <param name="approximationRank">Rank of approximation matrixes.</param>
            /// <param name="learningRate">Initial learning rate. It specifies the speed of the training algorithm.</param>
            /// <param name="numberOfIterations">Number of training iterations.</param>
            /// <example>
            /// <format type="text/markdown">
            /// <![CDATA[
            ///  [!code-csharp[MatrixFactorization](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Recommendation/MatrixFactorization.cs)]
            /// ]]></format>
            /// </example>
            public MatrixFactorizationTrainer MatrixFactorization(
                string labelColumn,
                string matrixColumnIndexColumnName,
                string matrixRowIndexColumnName,
                int approximationRank = MatrixFactorizationTrainer.Defaults.ApproximationRank,
                double learningRate = MatrixFactorizationTrainer.Defaults.LearningRate,
                int numberOfIterations = MatrixFactorizationTrainer.Defaults.NumIterations)
                    => new MatrixFactorizationTrainer(Owner.GetEnvironment(), labelColumn, matrixColumnIndexColumnName, matrixRowIndexColumnName,
                        approximationRank, learningRate, numberOfIterations);

            /// <summary>
            /// Train a matrix factorization model. It factorizes the training matrix into the product of two low-rank matrices.
            /// </summary>
            /// <remarks>
            /// <para>The basic idea of matrix factorization is finding two low-rank factor matrices to apporimate the training matrix.</para>
            /// <para>In this module, the expected training data is a list of tuples. Every tuple consists of a column index, a row index,
            /// and the value at the location specified by the two indexes. The training configuration is encoded in <see cref="MatrixFactorizationTrainer.Options"/>.
            /// </para>
            /// </remarks>
            /// <param name="options">Advanced arguments to the algorithm.</param>
            /// <example>
            /// <format type="text/markdown">
            /// <![CDATA[
            ///  [!code-csharp[MatrixFactorization](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Trainers/Recommendation/MatrixFactorizationWithOptions.cs)]
            /// ]]></format>
            /// </example>
            public MatrixFactorizationTrainer MatrixFactorization(
                MatrixFactorizationTrainer.Options options)
                    => new MatrixFactorizationTrainer(Owner.GetEnvironment(), options);
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
            string stratificationColumn = null, int? seed = null)
        {
            Environment.CheckNonEmpty(labelColumn, nameof(labelColumn));
            var result = CrossValidateTrain(data, estimator, numFolds, stratificationColumn, seed);
            return result.Select(x => new CrossValidationResult<RegressionMetrics>(x.Model, Evaluate(x.Scores, labelColumn), x.Scores, x.Fold)).ToArray();
        }
    }
}
