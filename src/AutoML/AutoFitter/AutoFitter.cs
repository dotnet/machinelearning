// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class AutoFitter<T> where T : class
    {
        private readonly IDebugLogger _debugLogger;
        private readonly IList<SuggestedPipelineResult<T>> _history;
        private readonly string _label;
        private readonly MLContext _context;
        private readonly OptimizingMetricInfo _optimizingMetricInfo;
        private readonly IDictionary<string, ColumnPurpose> _purposeOverrides;
        private readonly AutoFitSettings _settings;
        private readonly TaskKind _task;
        private readonly IEstimator<ITransformer> _preFeaturizers;

        private IDataView _trainData;
        private IDataView _validationData;

        public AutoFitter(TaskKind task, 
            IDataView trainData,
            string label,
            IDataView validationData,
            AutoFitSettings settings,
            IEstimator<ITransformer> preFeaturizers,
            IEnumerable<(string, ColumnPurpose)> purposeOverrides,
            OptimizingMetric metric,
            IDebugLogger debugLogger)
        {
            _debugLogger = debugLogger;
            _history = new List<SuggestedPipelineResult<T>>();
            _label = label;
            _context = new MLContext();
            _optimizingMetricInfo = new OptimizingMetricInfo(metric);
            _settings = settings ?? new AutoFitSettings();
            _purposeOverrides = purposeOverrides?.ToDictionary(p => p.Item1, p => p.Item2);
            _trainData = trainData;
            _task = task;
            _validationData = validationData;
            _preFeaturizers = preFeaturizers;
        }

        public IEnumerable<IterationResult<T>> Fit()
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
            var columns = AutoMlUtils.GetColumnInfoTuples(_context, _trainData, _label, _purposeOverrides);

            do
            {
                SuggestedPipeline pipeline = null;
                SuggestedPipelineResult<T> runResult = null;

                try
                {
                    var iterationStopwatch = Stopwatch.StartNew();
                    var getPiplelineStopwatch = Stopwatch.StartNew();

                    // get next pipeline
                    var iterationsRemaining = (int)_settings.StoppingCriteria.MaxIterations - _history.Count;
                    pipeline = PipelineSuggester.GetNextInferredPipeline(_history, columns, _task, iterationsRemaining, _optimizingMetricInfo.IsMaximizing);

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
                    runResult.GetPipelineTimeInSeconds = (int)getPiplelineStopwatch.Elapsed.TotalSeconds;
                }
                catch (Exception ex)
                {
                    WriteDebugLog(DebugStream.Exception, $"{pipeline?.Trainer} Crashed {ex}");

                    if (runResult == null)
                    {
                        runResult = new SuggestedPipelineResult<T>(null, null, pipeline, -1, ex);
                    }
                    else
                    {
                        runResult = new SuggestedPipelineResult<T>(runResult.EvaluatedMetrics, runResult.Model, runResult.Pipeline, runResult.Score, ex);
                    }
                }

                yield return runResult.ToIterationResult();
            } while (_history.Count < _settings.StoppingCriteria.MaxIterations &&
                    stopwatch.Elapsed.TotalMinutes < _settings.StoppingCriteria.TimeOutInMinutes);
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
                var evaluatedMetrics = GetEvaluatedMetrics(scoredValidationData);
                var score = GetPipelineScore(evaluatedMetrics);
                runResult = new SuggestedPipelineResult<T>(evaluatedMetrics, pipelineModel, pipeline, score, null);
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

        private double GetPipelineScore(object evaluatedMetrics)
        {
            var type = evaluatedMetrics.GetType();
            if(type == typeof(BinaryClassificationMetrics))
            {
                return ((BinaryClassificationMetrics)evaluatedMetrics).Accuracy;
            }
            if (type == typeof(MultiClassClassifierMetrics))
            {
                return ((MultiClassClassifierMetrics)evaluatedMetrics).AccuracyMicro;
            }
            if (type == typeof(RegressionMetrics))
            {
                return ((RegressionMetrics)evaluatedMetrics).RSquared;
            }
            
            // should not be possible to reach here
            throw new InvalidOperationException($"unsupported machine learning task type {_task}");
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
            if(_debugLogger == null)
            {
                return;
            }

            _debugLogger.Log(stream, message);
        }
    }
}