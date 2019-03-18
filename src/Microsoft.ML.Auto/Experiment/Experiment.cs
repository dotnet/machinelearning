// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;

namespace Microsoft.ML.Auto
{
    internal class Experiment<T> where T : class
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
        private readonly DirectoryInfo _modelDirectory;

        private IDataView _trainData;
        private IDataView _validationData;
        private ITransformer _preprocessorTransform;

        List<RunResult<T>> iterationResults = new List<RunResult<T>>();

        public Experiment(MLContext context,
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
                (trainData, validationData) = context.Regression.TestValidateSplit(context, trainData, columnInfo);
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
            _modelDirectory = GetModelDirectory(_experimentSettings.ModelDirectory);
        }

        public List<RunResult<T>> Execute()
        {
            if (_preFeaturizers != null)
            {
                // preprocess train and validation data
                _preprocessorTransform = _preFeaturizers.Fit(_trainData);
                _trainData = _preprocessorTransform.Transform(_trainData);
                _validationData = _preprocessorTransform.Transform(_validationData);
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
                    pipeline = PipelineSuggester.GetNextInferredPipeline(_context, _history, columns, _task, _optimizingMetricInfo.IsMaximizing, _trainerWhitelist, _experimentSettings.EnableCaching);

                    getPiplelineStopwatch.Stop();

                    // break if no candidates returned, means no valid pipeline available
                    if (pipeline == null)
                    {
                        break;
                    }

                    // evaluate pipeline
                    runResult = ProcessPipeline(pipeline);

                    runResult.RuntimeInSeconds = iterationStopwatch.Elapsed.TotalSeconds;
                    runResult.PipelineInferenceTimeInSeconds = getPiplelineStopwatch.Elapsed.TotalSeconds;
                }
                catch (Exception ex)
                {
                    WriteDebugLog(DebugStream.Exception, $"{pipeline?.Trainer} Crashed {ex}");
                    runResult = new SuggestedPipelineResult<T>(null, null, null, pipeline, -1, ex);
                }

                var iterationResult = runResult.ToIterationResult();
                ReportProgress(iterationResult);
                iterationResults.Add(iterationResult);

                // if model is perfect, break
                if (_metricsAgent.IsModelPerfect(iterationResult.ValidationMetrics))
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

        private FileInfo GetNextModelFileInfo()
        {
            if (_experimentSettings.ModelDirectory == null)
            {
                return null;
            }

            return new FileInfo(Path.Combine(_modelDirectory.FullName, 
                $"Model{_history.Count + 1}.zip"));
        }

        private SuggestedPipelineResult<T> ProcessPipeline(SuggestedPipeline pipeline)
        {
            // run pipeline
            var stopwatch = Stopwatch.StartNew();

            WriteDebugLog(DebugStream.RunResult, $"Processing pipeline {pipeline.ToString()}");

            SuggestedPipelineResult<T> runResult;

            try
            {
                var model = pipeline.ToEstimator().Fit(_trainData);
                var scoredValidationData = model.Transform(_validationData);
                var metrics = GetEvaluatedMetrics(scoredValidationData);
                var score = _metricsAgent.GetScore(metrics);

                var estimator = pipeline.ToEstimator();
                if (_preFeaturizers != null)
                {
                    estimator = _preFeaturizers.Append(estimator);
                    model = _preprocessorTransform.Append(model);
                }

                var modelFileInfo = GetNextModelFileInfo();
                var modelContainer = modelFileInfo == null ?
                    new ModelContainer(_context, model) :
                    new ModelContainer(_context, modelFileInfo, model);

                runResult = new SuggestedPipelineResult<T>(metrics, estimator, modelContainer, pipeline, score, null);
            }
            catch(Exception ex)
            {
                WriteDebugLog(DebugStream.Exception, $"{pipeline.Trainer} Crashed {ex}");
                runResult = new SuggestedPipelineResult<T>(null, pipeline.ToEstimator(), null, pipeline, 0, ex);
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
                    return _context.BinaryClassification.EvaluateNonCalibrated(scoredData, label: _columnInfo.LabelColumn) as T;
                case TaskKind.MulticlassClassification:
                    return _context.MulticlassClassification.Evaluate(scoredData, label: _columnInfo.LabelColumn) as T;
                case TaskKind.Regression:
                    return _context.Regression.Evaluate(scoredData, label: _columnInfo.LabelColumn) as T;
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
                WriteDebugLog(DebugStream.RunResult, $"{_history.Count}\t{runResult.Score}\t{stopwatch.Elapsed}\t{pipeline.ToString()}");
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