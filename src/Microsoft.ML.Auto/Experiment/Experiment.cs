// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Microsoft.ML.Auto
{
    internal class Experiment<TRunDetails, TMetrics> where TRunDetails : RunDetails
    {
        private readonly MLContext _context;
        private readonly OptimizingMetricInfo _optimizingMetricInfo;
        private readonly TaskKind _task;
        private readonly IProgress<TRunDetails> _progressCallback;
        private readonly ExperimentSettings _experimentSettings;
        private readonly IMetricsAgent<TMetrics> _metricsAgent;
        private readonly IEnumerable<TrainerName> _trainerWhitelist;
        private readonly DirectoryInfo _modelDirectory;
        private readonly DatasetColumnInfo[] _datasetColumnInfo;
        private readonly IRunner<TRunDetails> _runner;
        private readonly IList<SuggestedPipelineRunDetails> _history = new List<SuggestedPipelineRunDetails>();


        public Experiment(MLContext context,
            TaskKind task,
            OptimizingMetricInfo metricInfo,
            IProgress<TRunDetails> progressCallback,
            ExperimentSettings experimentSettings,
            IMetricsAgent<TMetrics> metricsAgent,
            IEnumerable<TrainerName> trainerWhitelist,
            DatasetColumnInfo[] datasetColumnInfo,
            IRunner<TRunDetails> runner)
        {
            _context = context;
            _optimizingMetricInfo = metricInfo;
            _task = task;
            _progressCallback = progressCallback;
            _experimentSettings = experimentSettings;
            _metricsAgent = metricsAgent;
            _trainerWhitelist = trainerWhitelist;
            _modelDirectory = GetModelDirectory(_experimentSettings.CacheDirectory);
            _datasetColumnInfo = datasetColumnInfo;
            _runner = runner;
        }

        public IList<TRunDetails> Execute()
        {
            var stopwatch = Stopwatch.StartNew();
            var iterationResults = new List<TRunDetails>();

            do
            {
                var iterationStopwatch = Stopwatch.StartNew();

                // get next pipeline
                var getPiplelineStopwatch = Stopwatch.StartNew();
                var pipeline = PipelineSuggester.GetNextInferredPipeline(_context, _history, _datasetColumnInfo, _task, _optimizingMetricInfo.IsMaximizing, _trainerWhitelist, _experimentSettings.CacheBeforeTrainer);
                var pipelineInferenceTimeInSeconds = getPiplelineStopwatch.Elapsed.TotalSeconds;

                // break if no candidates returned, means no valid pipeline available
                if (pipeline == null)
                {
                    break;
                }

                // evaluate pipeline
                Log(LogSeverity.Debug, $"Evaluating pipeline {pipeline.ToString()}");
                (SuggestedPipelineRunDetails suggestedPipelineRunDetails, TRunDetails runDetails)
                    = _runner.Run(pipeline, _modelDirectory, _history.Count + 1);
                _history.Add(suggestedPipelineRunDetails);
                WriteIterationLog(pipeline, suggestedPipelineRunDetails, iterationStopwatch);

                runDetails.RuntimeInSeconds = iterationStopwatch.Elapsed.TotalSeconds;
                runDetails.PipelineInferenceTimeInSeconds = getPiplelineStopwatch.Elapsed.TotalSeconds;

                ReportProgress(runDetails);
                iterationResults.Add(runDetails);

                // if model is perfect, break
                if (_metricsAgent.IsModelPerfect(suggestedPipelineRunDetails.Score))
                {
                    break;
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

        private void ReportProgress(TRunDetails iterationResult)
        {
            try
            {
                _progressCallback?.Report(iterationResult);
            }
            catch (Exception ex)
            {
                Log(LogSeverity.Error, $"Progress report callback reported exception {ex}");
            }
        }

        private void WriteIterationLog(SuggestedPipeline pipeline, SuggestedPipelineRunDetails runResult, Stopwatch stopwatch)
        {
            Log(LogSeverity.Debug, $"{_history.Count}\t{runResult.Score}\t{stopwatch.Elapsed}\t{pipeline.ToString()}");
        }

        private void Log(LogSeverity severity, string message)
        {
            if(_experimentSettings?.DebugLogger == null)
            {
                return;
            }

            _experimentSettings.DebugLogger.Log(severity, message);
        }
    }
}
