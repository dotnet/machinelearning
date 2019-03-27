// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.Data;
using Microsoft.ML.CLI.Utilities;
using NLog;

namespace Microsoft.ML.CLI.CodeGenerator
{
    internal class AutoMLEngine : IAutoMLEngine
    {
        private NewCommandSettings settings;
        private TaskKind taskKind;
        private bool? enableCaching;
        private static Logger logger = LogManager.GetCurrentClassLogger();

        public AutoMLEngine(NewCommandSettings settings)
        {
            this.settings = settings;
            this.taskKind = Utils.GetTaskKind(settings.MlTask);
            this.enableCaching = Utils.GetCacheSettings(settings.Cache);
        }

        public ColumnInferenceResults InferColumns(MLContext context, ColumnInformation columnInformation)
        {
            //Check what overload method of InferColumns needs to be called.
            logger.Log(LogLevel.Info, Strings.InferColumns);
            ColumnInferenceResults columnInference = null;
            var dataset = settings.Dataset.FullName;
            if (columnInformation.LabelColumn != null)
            {
                columnInference = context.Auto().InferColumns(dataset, columnInformation, groupColumns: false);
            }
            else
            {
                columnInference = context.Auto().InferColumns(dataset, settings.LabelColumnIndex, hasHeader: settings.HasHeader, groupColumns: false);
            }

            return columnInference;
        }

        (Pipeline, ITransformer) IAutoMLEngine.ExploreModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation)
        {
            ITransformer model = null;

            Pipeline pipeline = null;

            if (taskKind == TaskKind.BinaryClassification)
            {
                var optimizationMetric = new BinaryExperimentSettings().OptimizingMetric;
                var progressReporter = new ProgressHandlers.BinaryClassificationHandler(optimizationMetric);
                var result = context.Auto()
                    .CreateBinaryClassificationExperiment(new BinaryExperimentSettings()
                    {
                        MaxExperimentTimeInSeconds = settings.MaxExplorationTime,
                        ProgressHandler = progressReporter,
                        EnableCaching = this.enableCaching,
                        OptimizingMetric = optimizationMetric
                    })
                    .Execute(trainData, validationData, columnInformation);
                logger.Log(LogLevel.Info, Strings.RetrieveBestPipeline);
                var bestIteration = result.Best();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (taskKind == TaskKind.Regression)
            {
                var optimizationMetric = new RegressionExperimentSettings().OptimizingMetric;
                var progressReporter = new ProgressHandlers.RegressionHandler(optimizationMetric);
                var result = context.Auto()
                    .CreateRegressionExperiment(new RegressionExperimentSettings()
                    {
                        MaxExperimentTimeInSeconds = settings.MaxExplorationTime,
                        ProgressHandler = progressReporter,
                        OptimizingMetric = optimizationMetric,
                        EnableCaching = this.enableCaching
                    }).Execute(trainData, validationData, columnInformation);
                logger.Log(LogLevel.Info, Strings.RetrieveBestPipeline);
                var bestIteration = result.Best();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (taskKind == TaskKind.MulticlassClassification)
            {
                var optimizationMetric = new MulticlassExperimentSettings().OptimizingMetric;
                var progressReporter = new ProgressHandlers.MulticlassClassificationHandler(optimizationMetric);
                var result = context.Auto()
                    .CreateMulticlassClassificationExperiment(new MulticlassExperimentSettings()
                    {
                        MaxExperimentTimeInSeconds = settings.MaxExplorationTime,
                        ProgressHandler = progressReporter,
                        EnableCaching = this.enableCaching,
                        OptimizingMetric = optimizationMetric
                    }).Execute(trainData, validationData, columnInformation);
                logger.Log(LogLevel.Info, Strings.RetrieveBestPipeline);
                var bestIteration = result.Best();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            return (pipeline, model);
        }
    }
}
