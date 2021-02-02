// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal class CrossValSummaryRunner<TMetrics> : IRunner<RunDetail<TMetrics>>
        where TMetrics : class
    {
        private readonly MLContext _context;
        private readonly IDataView[] _trainDatasets;
        private readonly IDataView[] _validDatasets;
        private readonly IMetricsAgent<TMetrics> _metricsAgent;
        private readonly IEstimator<ITransformer> _preFeaturizer;
        private readonly ITransformer[] _preprocessorTransforms;
        private readonly string _groupIdColumn;
        private readonly string _labelColumn;
        private readonly OptimizingMetricInfo _optimizingMetricInfo;
        private readonly IChannel _logger;
        private readonly DataViewSchema _modelInputSchema;

        public CrossValSummaryRunner(MLContext context,
            IDataView[] trainDatasets,
            IDataView[] validDatasets,
            IMetricsAgent<TMetrics> metricsAgent,
            IEstimator<ITransformer> preFeaturizer,
            ITransformer[] preprocessorTransforms,
            string groupIdColumn,
            string labelColumn,
            OptimizingMetricInfo optimizingMetricInfo,
            IChannel logger)
        {
            _context = context;
            _trainDatasets = trainDatasets;
            _validDatasets = validDatasets;
            _metricsAgent = metricsAgent;
            _preFeaturizer = preFeaturizer;
            _preprocessorTransforms = preprocessorTransforms;
            _groupIdColumn = groupIdColumn;
            _labelColumn = labelColumn;
            _optimizingMetricInfo = optimizingMetricInfo;
            _logger = logger;
            _modelInputSchema = trainDatasets[0].Schema;
        }

        public (SuggestedPipelineRunDetail suggestedPipelineRunDetail, RunDetail<TMetrics> runDetail)
            Run(SuggestedPipeline pipeline, DirectoryInfo modelDirectory, int iterationNum)
        {
            var trainResults = new List<(ModelContainer model, TMetrics metrics, Exception exception, double score)>();

            for (var i = 0; i < _trainDatasets.Length; i++)
            {
                var modelFileInfo = RunnerUtil.GetModelFileInfo(modelDirectory, iterationNum, i + 1);
                var trainResult = RunnerUtil.TrainAndScorePipeline(pipeline.GetContext(), pipeline, _trainDatasets[i], _validDatasets[i],
                    _groupIdColumn, _labelColumn, _metricsAgent, _preprocessorTransforms?.ElementAt(i), modelFileInfo, _modelInputSchema,
                    _logger);
                trainResults.Add(trainResult);
            }

            var allRunsSucceeded = trainResults.All(r => r.exception == null);
            if (!allRunsSucceeded)
            {
                var firstException = trainResults.First(r => r.exception != null).exception;
                var errorRunDetail = new SuggestedPipelineRunDetail<TMetrics>(pipeline, double.NaN, false, null, null, firstException);
                return (errorRunDetail, errorRunDetail.ToIterationResult(_preFeaturizer));
            }

            // Get the model from the best fold
            var bestFoldIndex = BestResultUtil.GetIndexOfBestScore(trainResults.Select(r => r.score), _optimizingMetricInfo.IsMaximizing);
            // bestFoldIndex will be -1 if the optimization metric for all folds is NaN.
            // In this case, return model from the first fold.
            bestFoldIndex = bestFoldIndex != -1 ? bestFoldIndex : 0;
            var bestModel = trainResults.ElementAt(bestFoldIndex).model;

            // Get the average metrics across all folds
            var avgScore = GetAverageOfNonNaNScores(trainResults.Select(x => x.score));
            var indexClosestToAvg = GetIndexClosestToAverage(trainResults.Select(r => r.score), avgScore);
            var metricsClosestToAvg = trainResults[indexClosestToAvg].metrics;
            var avgMetrics = GetAverageMetrics(trainResults.Select(x => x.metrics), metricsClosestToAvg);

            // Build result objects
            var suggestedPipelineRunDetail = new SuggestedPipelineRunDetail<TMetrics>(pipeline, avgScore, allRunsSucceeded, avgMetrics, bestModel, null);
            var runDetail = suggestedPipelineRunDetail.ToIterationResult(_preFeaturizer);
            return (suggestedPipelineRunDetail, runDetail);
        }

        private static TMetrics GetAverageMetrics(IEnumerable<TMetrics> metrics, TMetrics metricsClosestToAvg)
        {
            if (typeof(TMetrics) == typeof(BinaryClassificationMetrics))
            {
                var newMetrics = metrics.Select(x => x as BinaryClassificationMetrics);
                Contracts.Assert(newMetrics != null);

                var result = new BinaryClassificationMetrics(
                    auc: GetAverageOfNonNaNScores(newMetrics.Select(x => x.AreaUnderRocCurve)),
                    accuracy: GetAverageOfNonNaNScores(newMetrics.Select(x => x.Accuracy)),
                    positivePrecision: GetAverageOfNonNaNScores(newMetrics.Select(x => x.PositivePrecision)),
                    positiveRecall: GetAverageOfNonNaNScores(newMetrics.Select(x => x.PositiveRecall)),
                    negativePrecision: GetAverageOfNonNaNScores(newMetrics.Select(x => x.NegativePrecision)),
                    negativeRecall: GetAverageOfNonNaNScores(newMetrics.Select(x => x.NegativeRecall)),
                    f1Score: GetAverageOfNonNaNScores(newMetrics.Select(x => x.F1Score)),
                    auprc: GetAverageOfNonNaNScores(newMetrics.Select(x => x.AreaUnderPrecisionRecallCurve)),
                    // Return ConfusionMatrix from the fold closest to average score
                    confusionMatrix: (metricsClosestToAvg as BinaryClassificationMetrics).ConfusionMatrix);
                return result as TMetrics;
            }

            if (typeof(TMetrics) == typeof(MulticlassClassificationMetrics))
            {
                var newMetrics = metrics.Select(x => x as MulticlassClassificationMetrics);
                Contracts.Assert(newMetrics != null);

                var result = new MulticlassClassificationMetrics(
                    accuracyMicro: GetAverageOfNonNaNScores(newMetrics.Select(x => x.MicroAccuracy)),
                    accuracyMacro: GetAverageOfNonNaNScores(newMetrics.Select(x => x.MacroAccuracy)),
                    logLoss: GetAverageOfNonNaNScores(newMetrics.Select(x => x.LogLoss)),
                    logLossReduction: GetAverageOfNonNaNScores(newMetrics.Select(x => x.LogLossReduction)),
                    topKPredictionCount: newMetrics.ElementAt(0).TopKPredictionCount,
                    topKAccuracies: GetAverageOfNonNaNScoresInNestedEnumerable(newMetrics.Select(x => x.TopKAccuracyForAllK)),
                    perClassLogLoss: (metricsClosestToAvg as MulticlassClassificationMetrics).PerClassLogLoss.ToArray(),
                    confusionMatrix: (metricsClosestToAvg as MulticlassClassificationMetrics).ConfusionMatrix);
                return result as TMetrics;
            }

            if (typeof(TMetrics) == typeof(RegressionMetrics))
            {
                var newMetrics = metrics.Select(x => x as RegressionMetrics);
                Contracts.Assert(newMetrics != null);

                var result = new RegressionMetrics(
                    l1: GetAverageOfNonNaNScores(newMetrics.Select(x => x.MeanAbsoluteError)),
                    l2: GetAverageOfNonNaNScores(newMetrics.Select(x => x.MeanSquaredError)),
                    rms: GetAverageOfNonNaNScores(newMetrics.Select(x => x.RootMeanSquaredError)),
                    lossFunction: GetAverageOfNonNaNScores(newMetrics.Select(x => x.LossFunction)),
                    rSquared: GetAverageOfNonNaNScores(newMetrics.Select(x => x.RSquared)));
                return result as TMetrics;
            }

            if (typeof(TMetrics) == typeof(RankingMetrics))
            {
                var newMetrics = metrics.Select(x => x as RankingMetrics);
                Contracts.Assert(newMetrics != null);

                var result = new RankingMetrics(
                    dcg: GetAverageOfNonNaNScoresInNestedEnumerable(newMetrics.Select(x => x.DiscountedCumulativeGains)),
                    ndcg: GetAverageOfNonNaNScoresInNestedEnumerable(newMetrics.Select(x => x.NormalizedDiscountedCumulativeGains)));
                return result as TMetrics;
            }

            throw new NotImplementedException($"Metric {typeof(TMetrics)} not implemented");
        }

        private static double[] GetAverageOfNonNaNScoresInNestedEnumerable(IEnumerable<IEnumerable<double>> results)
        {
            if (results.All(result => result == null))
            {
                // If all nested enumerables are null, we say the average is a null enumerable as well.
                // This is expected to happen on Multiclass metrics where the TopKAccuracyForAllK
                // array can be null if the topKPredictionCount isn't a valid number.
                // In that case all of the "results" enumerables will be null anyway, and so
                // returning null is the expected solution.
                return null;
            }

            // In case there are only some null elements, we'll ignore them:
            results = results.Where(result => result != null);

            double[] arr = new double[results.ElementAt(0).Count()];
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = GetAverageOfNonNaNScores(results.Select(x => x.ElementAt(i)));
            }
            return arr;
        }

        private static double GetAverageOfNonNaNScores(IEnumerable<double> results)
        {
            var newResults = results.Where(r => !double.IsNaN(r));
            // Return NaN iff all scores are NaN
            if (newResults.Count() == 0)
                return double.NaN;
            // Return average of non-NaN scores otherwise
            return newResults.Average(r => r);
        }

        /// <summary>
        /// return the index of value from <paramref name="values"/> that closest to <paramref name="average"/>. If <paramref name="average"/> is NaN, +/- inf, the first, max/min value's index will be return.
        /// </summary>
        private static int GetIndexClosestToAverage(IEnumerable<double> values, double average)
        {
            // Average will be NaN iff all values are NaN.
            // Return the first index in this case.
            if (double.IsNaN(average))
                return 0;

            // Return the max value's index if average is positive inf.
            if (double.IsPositiveInfinity(average))
                return values.ToList().IndexOf(values.Max());

            // Return the min value's index if average is negative inf.
            if (double.IsNegativeInfinity(average))
                return values.ToList().IndexOf(values.Min());

            int avgFoldIndex = -1;
            var smallestDistFromAvg = double.PositiveInfinity;
            for (var i = 0; i < values.Count(); i++)
            {
                var value = values.ElementAt(i);
                if (double.IsNaN(value))
                    continue;
                var distFromAvg = Math.Abs(value - average);
                if (distFromAvg < smallestDistFromAvg)
                {
                    smallestDistFromAvg = distFromAvg;
                    avgFoldIndex = i;
                }
            }
            return avgFoldIndex;
        }
    }
}
