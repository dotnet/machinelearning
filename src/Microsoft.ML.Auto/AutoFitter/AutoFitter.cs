// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Auto
{
    internal class AutoFitter<T> where T : class
    {
        private readonly IList<SuggestedPipelineResult<T>> _history;
        private readonly ColumnInformation _columnInfo;
        private readonly MLContext _context;
        private readonly OptimizingMetricInfo _optimizingMetricInfo;
        private readonly TaskKind _task;
        private readonly IEstimator<ITransformer> _preFeaturizers;
        private readonly IProgress<RunResult<T>> _progressCallback;
        private readonly ExperimentSettings _experimentSettings;
        private readonly IMetricsAgent<T> _metricsAgent;
        private readonly IEnumerable<TrainerName> _trainerWhitelist;

        private IDataView _trainData;
        private IDataView _validationData;

        List<RunResult<T>> iterationResults = new List<RunResult<T>>();

        public AutoFitter(MLContext context,
            TaskKind task,
            IDataView trainData,
            ColumnInformation columnInfo,
            IDataView validationData,
            IEstimator<ITransformer> preFeaturizers,
            OptimizingMetricInfo metricInfo,
            IProgress<RunResult<T>> progressCallback,
            ExperimentSettings experimentSettings,
            IMetricsAgent<T> metricsAgent,
            IEnumerable<TrainerName> trainerWhitelist)
        {
            if (validationData == null)
            {
                (trainData, validationData) = context.Regression.TestValidateSplit(trainData);
            }
            _trainData = trainData;
            _validationData = validationData;

            _history = new List<SuggestedPipelineResult<T>>();
            _columnInfo = columnInfo;
            _context = context;
            _optimizingMetricInfo = metricInfo;
            _task = task;
            _preFeaturizers = preFeaturizers;
            _progressCallback = progressCallback;
            _experimentSettings = experimentSettings;
            _metricsAgent = metricsAgent;
            _trainerWhitelist = trainerWhitelist;
        }

        public List<RunResult<T>> Fit()
        {
            ITransformer preprocessorTransform = null;
            if (_preFeaturizers != null)
            {
                // preprocess train and validation data
                preprocessorTransform = _preFeaturizers.Fit(_trainData);
                _trainData = preprocessorTransform.Transform(_trainData);
                _validationData = preprocessorTransform.Transform(_validationData);
            }

            var stopwatch = Stopwatch.StartNew();
            var columns = AutoMlUtils.GetColumnInfoTuples(_context, _trainData, _columnInfo);

            do
            {
                SuggestedPipeline pipeline = null;
                SuggestedPipelineResult<T> runResult = null;

                try
                {
                    var iterationStopwatch = Stopwatch.StartNew();
                    var getPiplelineStopwatch = Stopwatch.StartNew();

                    // get next pipeline
                    pipeline = PipelineSuggester.GetNextInferredPipeline(_history, columns, _task, _optimizingMetricInfo.IsMaximizing, _trainerWhitelist);

                    getPiplelineStopwatch.Stop();

                    // break if no candidates returned, means no valid pipeline available
                    if (pipeline == null)
                    {
                        break;
                    }

                    // evaluate pipeline
                    runResult = ProcessPipeline(pipeline);

                    if (preprocessorTransform != null)
                    {
                        runResult.Model = preprocessorTransform.Append(runResult.Model);
                    }

                    runResult.RuntimeInSeconds = (int)iterationStopwatch.Elapsed.TotalSeconds;
                    runResult.PipelineInferenceTimeInSeconds = (int)getPiplelineStopwatch.Elapsed.TotalSeconds;
                }
                catch (Exception ex)
                {
                    WriteDebugLog(DebugStream.Exception, $"{pipeline?.Trainer} Crashed {ex}");
                    runResult = new SuggestedPipelineResult<T>(null, null, pipeline, -1, ex);
                }

                var iterationResult = runResult.ToIterationResult();
                ReportProgress(iterationResult);
                iterationResults.Add(iterationResult);
            } while (_history.Count < _experimentSettings.MaxModels &&
                    !_experimentSettings.CancellationToken.IsCancellationRequested &&
                    stopwatch.Elapsed.TotalSeconds < _experimentSettings.MaxInferenceTimeInSeconds);

            return iterationResults;
        }

        private void ReportProgress(RunResult<T> iterationResult)
        {
            try
            {
                _progressCallback?.Report(iterationResult);
            }
            catch (Exception ex)
            {
                WriteDebugLog(DebugStream.Exception, $"Progress report callback reported exception {ex}");
            }
        }

        private SuggestedPipelineResult<T> ProcessPipeline(SuggestedPipeline pipeline)
        {
            // run pipeline
            var stopwatch = Stopwatch.StartNew();

            var commandLineStr = $"{string.Join(" xf=", pipeline.Transforms)} tr={pipeline.Trainer}";

            WriteDebugLog(DebugStream.RunResult, $"Processing pipeline {commandLineStr}.");

            SuggestedPipelineResult<T> runResult;
            try
            {
                var pipelineModel = pipeline.Fit(_trainData);
                var scoredValidationData = pipelineModel.Transform(_validationData);
                var metrics = GetEvaluatedMetrics(scoredValidationData);
                var score = _metricsAgent.GetScore(metrics);
                runResult = new SuggestedPipelineResult<T>(metrics, pipelineModel, pipeline, score, null);
            }
            catch(Exception ex)
            {
                WriteDebugLog(DebugStream.Exception, $"{pipeline.Trainer} Crashed {ex}");
                runResult = new SuggestedPipelineResult<T>(null, null, pipeline, 0, ex);
            }

            // save pipeline run
            _history.Add(runResult);
            WriteIterationLog(pipeline, runResult, stopwatch);

            return runResult;
        }

        private T GetEvaluatedMetrics(IDataView scoredData)
        {
            switch(_task)
            {
                case TaskKind.BinaryClassification:
                    return _context.BinaryClassification.EvaluateNonCalibrated(scoredData) as T;
                case TaskKind.MulticlassClassification:
                    return _context.MulticlassClassification.Evaluate(scoredData) as T;
                case TaskKind.Regression:
                    return _context.Regression.Evaluate(scoredData) as T;
                // should not be possible to reach here
                default:
                    throw new InvalidOperationException($"unsupported machine learning task type {_task}");
            }
        }

        private void WriteIterationLog(SuggestedPipeline pipeline, SuggestedPipelineResult runResult, Stopwatch stopwatch)
        {
            // debug log pipeline result
            if (runResult.RunSucceded)
            {
                var transformsSb = new StringBuilder();
                foreach (var transform in pipeline.Transforms)
                {
                    transformsSb.Append("xf=");
                    transformsSb.Append(transform);
                    transformsSb.Append(" ");
                }
                var commandLineStr = $"{transformsSb.ToString()} tr={pipeline.Trainer}";
                WriteDebugLog(DebugStream.RunResult, $"{_history.Count}\t{runResult.Score}\t{stopwatch.Elapsed}\t{commandLineStr}");
            }
        }

        private void WriteDebugLog(DebugStream stream, string message)
        {
            if(_experimentSettings?.DebugLogger == null)
            {
                return;
            }

            _experimentSettings.DebugLogger.Log(stream, message);
        }
    }
}