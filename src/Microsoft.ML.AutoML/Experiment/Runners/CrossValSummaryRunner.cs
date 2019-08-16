// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
                var trainResult = RunnerUtil.TrainAndScorePipeline(_context, pipeline, _trainDatasets[i], _validDatasets[i],
                    _labelColumn, _metricsAgent, _preprocessorTransforms?.ElementAt(i), modelFileInfo, _modelInputSchema,
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
            var bestModel = trainResults.ElementAt(bestFoldIndex).model;

            // Get the metrics from the fold whose score is closest to avg of all fold scores
            var avgScore = trainResults.Average(r => r.score);
            var indexClosestToAvg = GetIndexClosestToAverage(trainResults.Select(r => r.score), avgScore);
            var metricsClosestToAvg = trainResults[indexClosestToAvg].metrics;

            // Build result objects
            var suggestedPipelineRunDetail = new SuggestedPipelineRunDetail<TMetrics>(pipeline, avgScore, allRunsSucceeded, metricsClosestToAvg, bestModel, null);
            var runDetail = suggestedPipelineRunDetail.ToIterationResult(_preFeaturizer);
            return (suggestedPipelineRunDetail, runDetail);
        }

        private static int GetIndexClosestToAverage(IEnumerable<double> values, double average)
        {
            int avgFoldIndex = -1;
            var smallestDistFromAvg = double.PositiveInfinity;
            for (var i = 0; i < values.Count(); i++)
            {
                var distFromAvg = Math.Abs(values.ElementAt(i) - average);
                if (distFromAvg < smallestDistFromAvg || smallestDistFromAvg == double.PositiveInfinity)
                {
                    smallestDistFromAvg = distFromAvg;
                    avgFoldIndex = i;
                }
            }
            return avgFoldIndex;
        }
    }
}