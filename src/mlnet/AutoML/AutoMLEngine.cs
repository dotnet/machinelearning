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
        private static Logger logger = LogManager.GetCurrentClassLogger();

        public AutoMLEngine(NewCommandSettings settings)
        {
            this.settings = settings;
            this.taskKind = Utils.GetTaskKind(settings.MlTask);
        }

        public ColumnInferenceResults InferColumns(MLContext context)
        {
            //Check what overload method of InferColumns needs to be called.
            logger.Log(LogLevel.Info, Strings.InferColumns);
            ColumnInferenceResults columnInference = null;
            var dataset = settings.Dataset.FullName;
            if (settings.LabelColumnName != null)
            {
                columnInference = context.Auto().InferColumns(dataset, settings.LabelColumnName, groupColumns: false);
            }
            else
            {
                columnInference = context.Auto().InferColumns(dataset, settings.LabelColumnIndex, hasHeader: settings.HasHeader, groupColumns: false);
            }

            return columnInference;
        }

        (Pipeline, ITransformer) IAutoMLEngine.ExploreModels(MLContext context, IDataView trainData, IDataView validationData, string labelName)
        {
            ITransformer model = null;

            Pipeline pipeline = null;

            if (taskKind == TaskKind.BinaryClassification)
            {
                var progressReporter = new ProgressHandlers.BinaryClassificationHandler();
                var result = context.Auto()
                    .CreateBinaryClassificationExperiment(new BinaryExperimentSettings()
                    {
                        MaxExperimentTimeInSeconds = settings.MaxExplorationTime,
                        ProgressHandler = progressReporter
                    })
                    .Execute(trainData, validationData, new ColumnInformation() { LabelColumn = labelName });
                logger.Log(LogLevel.Info, Strings.RetrieveBestPipeline);
                var bestIteration = result.Best();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (taskKind == TaskKind.Regression)
            {
                var progressReporter = new ProgressHandlers.RegressionHandler();
                var result = context.Auto()
                    .CreateRegressionExperiment(new RegressionExperimentSettings()
                    {
                        MaxExperimentTimeInSeconds = settings.MaxExplorationTime,
                        ProgressHandler = progressReporter
                    }).Execute(trainData, validationData, new ColumnInformation() { LabelColumn = labelName });
                logger.Log(LogLevel.Info, Strings.RetrieveBestPipeline);
                var bestIteration = result.Best();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (taskKind == TaskKind.MulticlassClassification)
            {
                var progressReporter = new ProgressHandlers.MulticlassClassificationHandler();

                var experimentSettings = new MulticlassExperimentSettings()
                {
                    MaxExperimentTimeInSeconds = settings.MaxExplorationTime,
                    ProgressHandler = progressReporter
                };

                // Inclusion list for currently supported learners. Need to remove once we have codegen support for all other learners.
                experimentSettings.Trainers.Clear();
                experimentSettings.Trainers.Add(MulticlassClassificationTrainer.LightGbm);
                experimentSettings.Trainers.Add(MulticlassClassificationTrainer.LogisticRegression);
                experimentSettings.Trainers.Add(MulticlassClassificationTrainer.StochasticDualCoordinateAscent);

                var result = context.Auto()
                .CreateMulticlassClassificationExperiment(experimentSettings)
                    .Execute(trainData, validationData, new ColumnInformation() { LabelColumn = labelName });
                logger.Log(LogLevel.Info, Strings.RetrieveBestPipeline);
                var bestIteration = result.Best();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            return (pipeline, model);
        }
    }
}
