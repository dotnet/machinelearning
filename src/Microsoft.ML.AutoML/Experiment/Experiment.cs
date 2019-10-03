// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal class Experiment<TRunDetail, TMetrics> where TRunDetail : RunDetail
    {
        private readonly MLContext _context;
        private readonly OptimizingMetricInfo _optimizingMetricInfo;
        private readonly TaskKind _task;
        private readonly IProgress<TRunDetail> _progressCallback;
        private readonly ExperimentSettings _experimentSettings;
        private readonly IMetricsAgent<TMetrics> _metricsAgent;
        private readonly IEnumerable<TrainerName> _trainerWhitelist;
        private readonly DirectoryInfo _modelDirectory;
        private readonly DatasetColumnInfo[] _datasetColumnInfo;
        private readonly IRunner<TRunDetail> _runner;
        private readonly IList<SuggestedPipelineRunDetail> _history;
        private readonly IChannel _logger;

        public Experiment(MLContext context,
            TaskKind task,
            OptimizingMetricInfo metricInfo,
            IProgress<TRunDetail> progressCallback,
            ExperimentSettings experimentSettings,
            IMetricsAgent<TMetrics> metricsAgent,
            IEnumerable<TrainerName> trainerWhitelist,
            DatasetColumnInfo[] datasetColumnInfo,
            IRunner<TRunDetail> runner,
            IChannel logger)
        {
            _context = context;
            _history = new List<SuggestedPipelineRunDetail>();
            _optimizingMetricInfo = metricInfo;
            _task = task;
            _progressCallback = progressCallback;
            _experimentSettings = experimentSettings;
            _metricsAgent = metricsAgent;
            _trainerWhitelist = trainerWhitelist;
            _modelDirectory = GetModelDirectory(_experimentSettings.CacheDirectory);
            _datasetColumnInfo = datasetColumnInfo;
            _runner = runner;
            _logger = logger;
        }

        public IList<TRunDetail> Execute()
        {
            var stopwatch = Stopwatch.StartNew();
            var iterationResults = new List<TRunDetail>();

            do
            {
                var iterationStopwatch = Stopwatch.StartNew();

                // get next pipeline
                var getPiplelineStopwatch = Stopwatch.StartNew();
                var pipeline = PipelineSuggester.GetNextInferredPipeline(_context, _history, _datasetColumnInfo, _task, _optimizingMetricInfo.IsMaximizing, _experimentSettings.CacheBeforeTrainer, _trainerWhitelist);
                var pipelineInferenceTimeInSeconds = getPiplelineStopwatch.Elapsed.TotalSeconds;

                // break if no candidates returned, means no valid pipeline available
                if (pipeline == null)
                {
                    break;
                }

                // evaluate pipeline
                _logger.Trace($"Evaluating pipeline {pipeline.ToString()}");
                (SuggestedPipelineRunDetail suggestedPipelineRunDetail, TRunDetail runDetail)
                    = _runner.Run(pipeline, _modelDirectory, _history.Count + 1);
                _history.Add(suggestedPipelineRunDetail);
                WriteIterationLog(pipeline, suggestedPipelineRunDetail, iterationStopwatch);

                runDetail.RuntimeInSeconds = iterationStopwatch.Elapsed.TotalSeconds;
                runDetail.PipelineInferenceTimeInSeconds = getPiplelineStopwatch.Elapsed.TotalSeconds;

                ReportProgress(runDetail);
                iterationResults.Add(runDetail);

                // if model is perfect, break
                if (_metricsAgent.IsModelPerfect(suggestedPipelineRunDetail.Score))
                {
                    break;
                }

                // If after third run, all runs have failed so far, throw exception
                if (_history.Count() == 3 && _history.All(r => !r.RunSucceded))
                {
                    throw new InvalidOperationException($"Training failed with the exception: {_history.Last().Exception}");
                }

            } while (_history.Count < _experimentSettings.MaxModels &&
                    !_experimentSettings.CancellationToken.IsCancellationRequested &&
                    stopwatch.Elapsed.TotalSeconds < _experimentSettings.MaxExperimentTimeInSeconds);

            return iterationResults;
        }

        private static DirectoryInfo GetModelDirectory(DirectoryInfo rootDir)
        {
            if (rootDir == null)
            {
                return null;
            }
            var subdirs = rootDir.Exists ?
                new HashSet<string>(rootDir.EnumerateDirectories().Select(d => d.Name)) :
                new HashSet<string>();
            string experimentDir;
            for (var i = 0; ; i++)
            {
                experimentDir = $"experiment{i}";
                if (!subdirs.Contains(experimentDir))
                {
                    break;
                }
            }
            var experimentDirFullPath = Path.Combine(rootDir.FullName, experimentDir);
            var experimentDirInfo = new DirectoryInfo(experimentDirFullPath);
            if (!experimentDirInfo.Exists)
            {
                experimentDirInfo.Create();
            }
            return experimentDirInfo;
        }

        private void ReportProgress(TRunDetail iterationResult)
        {
            try
            {
                _progressCallback?.Report(iterationResult);
            }
            catch (Exception ex)
            {
                _logger.Error($"Progress report callback reported exception {ex}");
            }
        }

        private void WriteIterationLog(SuggestedPipeline pipeline, SuggestedPipelineRunDetail runResult, Stopwatch stopwatch)
        {
            _logger.Trace($"{_history.Count}\t{runResult.Score}\t{stopwatch.Elapsed}\t{pipeline.ToString()}");
        }
    }
}
