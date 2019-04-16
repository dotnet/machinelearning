// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.Data;
using Microsoft.ML.CLI.ShellProgressBar;
using Microsoft.ML.CLI.Utilities;
using Microsoft.ML.Data;
using NLog;

namespace Microsoft.ML.CLI.CodeGenerator
{
    internal class AutoMLEngine : IAutoMLEngine
    {
        private NewCommandSettings settings;
        private TaskKind taskKind;
        private CacheBeforeTrainer cacheBeforeTrainer;
        private static Logger logger = LogManager.GetCurrentClassLogger();

        public AutoMLEngine(NewCommandSettings settings)
        {
            this.settings = settings;
            this.taskKind = Utils.GetTaskKind(settings.MlTask);
            this.cacheBeforeTrainer = Utils.GetCacheSettings(settings.Cache);
        }

        public ColumnInferenceResults InferColumns(MLContext context, ColumnInformation columnInformation)
        {
            // Check what overload method of InferColumns needs to be called.
            logger.Log(LogLevel.Trace, Strings.InferColumns);
            ColumnInferenceResults columnInference = null;
            var dataset = settings.Dataset.FullName;
            if (columnInformation.LabelColumnName != null)
            {
                columnInference = context.Auto().InferColumns(dataset, columnInformation, groupColumns: false);
            }
            else
            {
                columnInference = context.Auto().InferColumns(dataset, settings.LabelColumnIndex, hasHeader: settings.HasHeader, groupColumns: false);
            }

            return columnInference;
        }

        ExperimentResult<BinaryClassificationMetrics> IAutoMLEngine.ExploreBinaryClassificationModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, BinaryClassificationMetric optimizationMetric, ProgressBar progressBar)
        {
            var progressReporter = new ProgressHandlers.BinaryClassificationHandler(optimizationMetric, progressBar);
            var result = context.Auto()
                .CreateBinaryClassificationExperiment(new BinaryExperimentSettings()
                {
                    MaxExperimentTimeInSeconds = settings.MaxExplorationTime,
                    CacheBeforeTrainer = this.cacheBeforeTrainer,
                    OptimizingMetric = optimizationMetric
                })
                .Execute(trainData, validationData, columnInformation, progressHandler: progressReporter);
            logger.Log(LogLevel.Trace, Strings.RetrieveBestPipeline);
            return result;
        }

        ExperimentResult<RegressionMetrics> IAutoMLEngine.ExploreRegressionModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, RegressionMetric optimizationMetric, ProgressBar progressBar)
        {
            var progressReporter = new ProgressHandlers.RegressionHandler(optimizationMetric, progressBar);
            var result = context.Auto()
                .CreateRegressionExperiment(new RegressionExperimentSettings()
                {
                    MaxExperimentTimeInSeconds = settings.MaxExplorationTime,
                    OptimizingMetric = optimizationMetric,
                    CacheBeforeTrainer = this.cacheBeforeTrainer
                }).Execute(trainData, validationData, columnInformation, progressHandler: progressReporter);
            logger.Log(LogLevel.Trace, Strings.RetrieveBestPipeline);
            return result;
        }

        ExperimentResult<MulticlassClassificationMetrics> IAutoMLEngine.ExploreMultiClassificationModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, MulticlassClassificationMetric optimizationMetric, ProgressBar progressBar)
        {
            var progressReporter = new ProgressHandlers.MulticlassClassificationHandler(optimizationMetric, progressBar);
            var result = context.Auto()
                .CreateMulticlassClassificationExperiment(new MulticlassExperimentSettings()
                {
                    MaxExperimentTimeInSeconds = settings.MaxExplorationTime,
                    CacheBeforeTrainer = this.cacheBeforeTrainer,
                    OptimizingMetric = optimizationMetric
                }).Execute(trainData, validationData, columnInformation, progressHandler: progressReporter);
            logger.Log(LogLevel.Trace, Strings.RetrieveBestPipeline);
            return result;
        }

    }
}
