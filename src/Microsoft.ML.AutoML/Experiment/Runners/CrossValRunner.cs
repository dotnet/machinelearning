// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal class CrossValRunner<TMetrics> : IRunner<CrossValidationRunDetail<TMetrics>>
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
        private readonly IChannel _logger;
        private readonly DataViewSchema _modelInputSchema;

        public CrossValRunner(MLContext context,
            IDataView[] trainDatasets,
            IDataView[] validDatasets,
            IMetricsAgent<TMetrics> metricsAgent,
            IEstimator<ITransformer> preFeaturizer,
            ITransformer[] preprocessorTransforms,
            string groupIdColumn,
            string labelColumn,
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
            _logger = logger;
            _modelInputSchema = trainDatasets[0].Schema;
        }

        public (SuggestedPipelineRunDetail suggestedPipelineRunDetail, CrossValidationRunDetail<TMetrics> runDetail)
            Run(SuggestedPipeline pipeline, DirectoryInfo modelDirectory, int iterationNum)
        {
            var trainResults = new List<SuggestedPipelineTrainResult<TMetrics>>();

            for (var i = 0; i < _trainDatasets.Length; i++)
            {
                var modelFileInfo = RunnerUtil.GetModelFileInfo(modelDirectory, iterationNum, i + 1);
                var trainResult = RunnerUtil.TrainAndScorePipeline(_context, pipeline, _trainDatasets[i], _validDatasets[i],
                    _groupIdColumn, _labelColumn, _metricsAgent, _preprocessorTransforms?[i], modelFileInfo, _modelInputSchema, _logger);
                trainResults.Add(new SuggestedPipelineTrainResult<TMetrics>(trainResult.model, trainResult.metrics, trainResult.exception, trainResult.score));
            }

            var avgScore = CalcAverageScore(trainResults.Select(r => r.Score));
            var allRunsSucceeded = trainResults.All(r => r.Exception == null);

            var suggestedPipelineRunDetail = new SuggestedPipelineCrossValRunDetail<TMetrics>(pipeline, avgScore, allRunsSucceeded, trainResults);
            var runDetail = suggestedPipelineRunDetail.ToIterationResult(_preFeaturizer);
            return (suggestedPipelineRunDetail, runDetail);
        }

        private static double CalcAverageScore(IEnumerable<double> scores)
        {
            var newScores = scores.Where(r => !double.IsNaN(r));
            // Return NaN iff all scores are NaN
            if (newScores.Count() == 0)
                return double.NaN;
            return newScores.Average();
        }
    }
}
