// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.CodeGenerator.CSharp;
using Microsoft.ML.CLI.Data;
using Microsoft.ML.CLI.Utilities;
using Microsoft.ML.Data;
using NLog;

namespace Microsoft.ML.CLI.Commands.New
{
    internal class NewCommand : ICommand
    {
        private NewCommandSettings settings;
        private static Logger logger = LogManager.GetCurrentClassLogger();
        private TaskKind taskKind;

        internal NewCommand(NewCommandSettings settings)
        {
            this.settings = settings;
            this.taskKind = Utils.GetTaskKind(settings.MlTask);
        }

        public void Execute()
        {
            var context = new MLContext();

            // Infer columns
            ColumnInferenceResults columnInference = null;
            try
            {
                columnInference = InferColumns(context);
            }
            catch (Exception e)
            {
                logger.Log(LogLevel.Error, $"{Strings.InferColumnError}");
                logger.Log(LogLevel.Error, e.Message);
                logger.Log(LogLevel.Debug, e.ToString());
                logger.Log(LogLevel.Error, Strings.Exiting);
                return;
            }

            // Sanitize columns
            Array.ForEach(columnInference.TextLoaderOptions.Columns, t => t.Name = Utils.Sanitize(t.Name));

            var sanitized_Label_Name = Utils.Sanitize(columnInference.ColumnInformation.LabelColumn);

            // Load data
            (IDataView trainData, IDataView validationData) = LoadData(context, columnInference.TextLoaderOptions);

            // Explore the models
            (Pipeline, ITransformer) result = default;
            Console.WriteLine($"{Strings.ExplorePipeline}: {settings.MlTask}");
            try
            {
                result = ExploreModels(context, trainData, validationData, sanitized_Label_Name);
            }
            catch (Exception e)
            {
                logger.Log(LogLevel.Error, $"{Strings.ExplorePipelineException}:");
                logger.Log(LogLevel.Error, e.Message);
                logger.Log(LogLevel.Debug, e.ToString());
                logger.Log(LogLevel.Error, Strings.Exiting);
                return;
            }

            //Get the best pipeline
            Pipeline pipeline = null;
            pipeline = result.Item1;
            var model = result.Item2;

            // Save the model
            logger.Log(LogLevel.Info, Strings.SavingBestModel);
            var modelPath = new FileInfo(Path.Combine(settings.OutputPath.FullName, "model.zip"));
            Utils.SaveModel(model, modelPath, context);

            // Generate the Project
            GenerateProject(columnInference, pipeline, sanitized_Label_Name, modelPath);
        }

        internal ColumnInferenceResults InferColumns(MLContext context)
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

        internal void GenerateProject(ColumnInferenceResults columnInference, Pipeline pipeline, string labelName, FileInfo modelPath)
        {
            //Generate code
            logger.Log(LogLevel.Info, $"{Strings.GenerateProject} : {settings.OutputPath.FullName}");
            var codeGenerator = new CodeGenerator.CSharp.CodeGenerator(
                pipeline,
                columnInference,
                new CodeGeneratorSettings()
                {
                    TrainDataset = settings.Dataset.FullName,
                    MlTask = taskKind,
                    TestDataset = settings.TestDataset?.FullName,
                    OutputName = settings.Name,
                    OutputBaseDir = settings.OutputPath.FullName,
                    LabelName = labelName,
                    ModelPath = modelPath.FullName
                });
            codeGenerator.GenerateOutput();
        }

        internal (Pipeline, ITransformer) ExploreModels(MLContext context, IDataView trainData, IDataView validationData, string labelName)
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

        internal (IDataView, IDataView) LoadData(MLContext context, TextLoader.Options textLoaderOptions)
        {
            logger.Log(LogLevel.Info, Strings.CreateDataLoader);
            var textLoader = context.Data.CreateTextLoader(textLoaderOptions);

            logger.Log(LogLevel.Info, Strings.LoadData);
            var trainData = textLoader.Load(settings.Dataset.FullName);
            var validationData = settings.ValidationDataset == null ? null : textLoader.Load(settings.ValidationDataset.FullName);

            return (trainData, validationData);
        }
    }
}
