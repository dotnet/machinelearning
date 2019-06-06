// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.AutoML;
using Microsoft.ML.CLI.Data;
using Microsoft.ML.CLI.ShellProgressBar;
using Microsoft.ML.CLI.Utilities;
using Microsoft.ML.Data;
using NLog;

namespace Microsoft.ML.CLI.CodeGenerator
{
    internal class AutoMLEngine : IAutoMLEngine
    {
        private NewCommandSettings _settings;
        private TaskKind _taskKind;
        private CacheBeforeTrainer _cacheBeforeTrainer;
        private static Logger _logger = LogManager.GetCurrentClassLogger();

        public AutoMLEngine(NewCommandSettings settings)
        {
            _settings = settings;
            _taskKind = Utils.GetTaskKind(settings.MlTask);
            _cacheBeforeTrainer = Utils.GetCacheSettings(settings.Cache);
        }

        public ColumnInferenceResults InferColumns(MLContext context, ColumnInformation columnInformation)
        {
            // Check what overload method of InferColumns needs to be called.
            _logger.Log(LogLevel.Trace, Strings.InferColumns);
            ColumnInferenceResults columnInference = null;
            var dataset = _settings.Dataset.FullName;
            if (columnInformation.LabelColumnName != null)
            {
                columnInference = context.Auto().InferColumns(dataset, columnInformation, groupColumns: false);
            }
            else
            {
                columnInference = context.Auto().InferColumns(dataset, _settings.LabelColumnIndex, hasHeader: _settings.HasHeader, groupColumns: false);
            }

            return columnInference;
        }

        void IAutoMLEngine.ExploreBinaryClassificationModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, BinaryClassificationMetric optimizationMetric, ProgressHandlers.BinaryClassificationHandler handler, ProgressBar progressBar)
        {
            ExperimentResult<BinaryClassificationMetrics> result = context.Auto()
                .CreateBinaryClassificationExperiment(new BinaryExperimentSettings()
                {
                    MaxExperimentTimeInSeconds = _settings.MaxExplorationTime,
                    CacheBeforeTrainer = _cacheBeforeTrainer,
                    OptimizingMetric = optimizationMetric
                })
                .Execute(trainData, validationData, columnInformation, progressHandler: handler);

            _logger.Log(LogLevel.Trace, Strings.RetrieveBestPipeline);
        }

        void IAutoMLEngine.ExploreRegressionModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, RegressionMetric optimizationMetric, ProgressHandlers.RegressionHandler handler, ProgressBar progressBar)
        {
            ExperimentResult<RegressionMetrics> result = context.Auto()
                .CreateRegressionExperiment(new RegressionExperimentSettings()
                {
                    MaxExperimentTimeInSeconds = _settings.MaxExplorationTime,
                    OptimizingMetric = optimizationMetric,
                    CacheBeforeTrainer = _cacheBeforeTrainer
                }).Execute(trainData, validationData, columnInformation, progressHandler: handler);

            _logger.Log(LogLevel.Trace, Strings.RetrieveBestPipeline);
        }

        void IAutoMLEngine.ExploreMultiClassificationModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, MulticlassClassificationMetric optimizationMetric, ProgressHandlers.MulticlassClassificationHandler handler, ProgressBar progressBar)
        {
            ExperimentResult<MulticlassClassificationMetrics> result = context.Auto()
                .CreateMulticlassClassificationExperiment(new MulticlassExperimentSettings()
                {
                    MaxExperimentTimeInSeconds = _settings.MaxExplorationTime,
                    CacheBeforeTrainer = _cacheBeforeTrainer,
                    OptimizingMetric = optimizationMetric
                }).Execute(trainData, validationData, columnInformation, progressHandler: handler);

            _logger.Log(LogLevel.Trace, Strings.RetrieveBestPipeline);
        }

    }
}
