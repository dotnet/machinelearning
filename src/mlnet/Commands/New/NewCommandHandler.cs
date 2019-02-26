// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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
        private NewCommandOptions options;
        private static Logger logger = LogManager.GetCurrentClassLogger();
        private TaskKind taskKind;

        internal NewCommand(NewCommandOptions options)
        {
            this.options = options;
            this.taskKind = Utils.GetTaskKind(options.MlTask);
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
            Array.ForEach(columnInference.TextLoaderArgs.Columns, t => t.Name = Utils.Sanitize(t.Name));

            var sanitized_Label_Name = Utils.Sanitize(columnInference.ColumnInformation.LabelColumn);

            // Load data
            (IDataView trainData, IDataView validationData) = LoadData(context, columnInference.TextLoaderArgs);

            // Explore the models
            (Pipeline, ITransformer) result = default;
            Console.WriteLine($"{Strings.ExplorePipeline}: {options.MlTask}");
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
            Utils.SaveModel(model, options.OutputPath.FullName, $"model.zip", context);

            // Generate the Project
            GenerateProject(columnInference, pipeline, sanitized_Label_Name);
        }

        internal ColumnInferenceResults InferColumns(MLContext context)
        {
            //Check what overload method of InferColumns needs to be called.
            logger.Log(LogLevel.Info, Strings.InferColumns);
            ColumnInferenceResults columnInference = null;
            var dataset = options.Dataset.FullName;
            if (options.LabelColumnName != null)
            {
                columnInference = context.AutoInference().InferColumns(dataset, options.LabelColumnName, groupColumns: false);
            }
            else
            {
                columnInference = context.AutoInference().InferColumns(dataset, options.LabelColumnIndex, hasHeader: options.HasHeader, groupColumns: false);
            }

            return columnInference;
        }

        internal void GenerateProject(ColumnInferenceResults columnInference, Pipeline pipeline, string labelName)
        {
            //Generate code
            logger.Log(LogLevel.Info, $"{Strings.GenerateProject} : {options.OutputPath.FullName}");
            var codeGenerator = new CodeGenerator.CSharp.CodeGenerator(
                pipeline,
                columnInference,
                new CodeGeneratorOptions()
                {
                    TrainDataset = options.Dataset,
                    MlTask = taskKind,
                    TestDataset = options.TestDataset,
                    OutputName = options.Name,
                    OutputBaseDir = options.OutputPath.FullName,
                    LabelName = labelName
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
                var result = context.AutoInference()
                    .CreateBinaryClassificationExperiment(new BinaryExperimentSettings()
                    {
                        MaxInferenceTimeInSeconds = options.MaxExplorationTime,
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
                var result = context.AutoInference()
                    .CreateRegressionExperiment(new RegressionExperimentSettings()
                    {
                        MaxInferenceTimeInSeconds = options.MaxExplorationTime,
                        ProgressHandler = progressReporter
                    }).Execute(trainData, validationData, new ColumnInformation() { LabelColumn = labelName });
                logger.Log(LogLevel.Info, Strings.RetrieveBestPipeline);
                var bestIteration = result.Best();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (taskKind == TaskKind.MulticlassClassification)
            {
                throw new NotImplementedException();
            }
            //Multi-class exploration here

            return (pipeline, model);
        }

        internal (IDataView, IDataView) LoadData(MLContext context, TextLoader.Arguments textLoaderArgs)
        {
            logger.Log(LogLevel.Info, Strings.CreateDataLoader);
            var textLoader = context.Data.CreateTextLoader(textLoaderArgs);

            logger.Log(LogLevel.Info, Strings.LoadData);
            var trainData = textLoader.Read(options.Dataset.FullName);
            var validationData = options.ValidationDataset == null ? null : textLoader.Read(options.ValidationDataset.FullName);

            return (trainData, validationData);
        }
    }
}
