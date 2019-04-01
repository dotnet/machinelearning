// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Data.DataView;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.CodeGenerator.CSharp;
using Microsoft.ML.CLI.Data;
using Microsoft.ML.CLI.ShellProgressBar;
using Microsoft.ML.CLI.Utilities;
using Microsoft.ML.Data;
using NLog;

namespace Microsoft.ML.CLI.CodeGenerator
{
    internal class CodeGenerationHelper
    {

        private IAutoMLEngine automlEngine;
        private NewCommandSettings settings;
        private static Logger logger = LogManager.GetCurrentClassLogger();
        private TaskKind taskKind;

        public CodeGenerationHelper(IAutoMLEngine automlEngine, NewCommandSettings settings)
        {
            this.automlEngine = automlEngine;
            this.settings = settings;
            this.taskKind = Utils.GetTaskKind(settings.MlTask);
        }

        public void GenerateCode()
        {
            var context = new MLContext();

            // Infer columns
            ColumnInferenceResults columnInference = null;
            try
            {
                var inputColumnInformation = new ColumnInformation();
                inputColumnInformation.LabelColumn = settings.LabelColumnName;
                foreach (var value in settings.IgnoreColumns)
                {
                    inputColumnInformation.IgnoredColumns.Add(value);
                }
                columnInference = automlEngine.InferColumns(context, inputColumnInformation);
            }
            catch (Exception e)
            {
                logger.Log(LogLevel.Error, $"{Strings.InferColumnError}");
                logger.Log(LogLevel.Error, e.Message);
                logger.Log(LogLevel.Debug, e.ToString());
                logger.Log(LogLevel.Error, Strings.Exiting);
                return;
            }

            var textLoaderOptions = columnInference.TextLoaderOptions;
            var columnInformation = columnInference.ColumnInformation;

            // Sanitization of input data.
            Array.ForEach(textLoaderOptions.Columns, t => t.Name = Utils.Sanitize(t.Name));
            columnInformation = Utils.GetSanitizedColumnInformation(columnInformation);

            // Load data
            (IDataView trainData, IDataView validationData) = LoadData(context, textLoaderOptions);

            // Explore the models

            // The reason why we are doing this way of defining 3 different results is because of the AutoML API 
            // i.e there is no common class/interface to handle all three tasks together.

            IEnumerable<RunResult<BinaryClassificationMetrics>> binaryRunResults = default;
            IEnumerable<RunResult<MultiClassClassifierMetrics>> multiRunResults = default;
            IEnumerable<RunResult<RegressionMetrics>> regressionRunResults = default;

            Console.WriteLine($"{Strings.ExplorePipeline}: {settings.MlTask}");
            try
            {
                var options = new ProgressBarOptions
                {
                    ForegroundColor = ConsoleColor.Yellow,
                    ForegroundColorDone = ConsoleColor.DarkGreen,
                    BackgroundColor = ConsoleColor.Gray,
                    ProgressCharacter = '\u2593',
                    BackgroundCharacter = '─',
                };
                var wait = TimeSpan.FromSeconds(settings.MaxExplorationTime);
                using (var pbar = new FixedDurationBar(wait, "", options))
                {
                    Task t = default;
                    switch (taskKind)
                    {
                        case TaskKind.BinaryClassification:
                            t = Task.Run(() => binaryRunResults = automlEngine.ExploreBinaryClassificationModels(context, trainData, validationData, columnInformation, new BinaryExperimentSettings().OptimizingMetric, pbar));
                            break;
                        case TaskKind.Regression:
                            t = Task.Run(() => regressionRunResults = automlEngine.ExploreRegressionModels(context, trainData, validationData, columnInformation, new RegressionExperimentSettings().OptimizingMetric, pbar));
                            break;
                        case TaskKind.MulticlassClassification:
                            t = Task.Run(() => multiRunResults = automlEngine.ExploreMultiClassificationModels(context, trainData, validationData, columnInformation, new MulticlassExperimentSettings().OptimizingMetric, pbar));
                            break;
                        default:
                            logger.Log(LogLevel.Error, Strings.UnsupportedMlTask);
                            break;
                    }

                    if (!pbar.CompletedHandle.WaitOne(wait))
                        Console.Error.WriteLine($"{nameof(FixedDurationBar)} did not signal {nameof(FixedDurationBar.CompletedHandle)} after {wait}");

                    if (t.IsCompleted == false)
                    {
                        string originalMessage = pbar.Message;
                        pbar.Message = " Waiting for the last iteration to complete ...";
                        t.Wait();
                    }
                }

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
            Pipeline bestPipeline = null;
            ITransformer bestModel = null;

            switch (taskKind)
            {
                case TaskKind.BinaryClassification:
                    var bestBinaryIteration = binaryRunResults.Best();
                    bestPipeline = bestBinaryIteration.Pipeline;
                    bestModel = bestBinaryIteration.Model;
                    ConsolePrinter.ExperimentResultsHeader(LogLevel.Info, settings.MlTask, settings.Dataset.Name, columnInformation.LabelColumn, settings.MaxExplorationTime.ToString(), binaryRunResults.Count());
                    ConsolePrinter.PrintIterationSummary(binaryRunResults, new BinaryExperimentSettings().OptimizingMetric, 5);
                    break;
                case TaskKind.Regression:
                    var bestRegressionIteration = regressionRunResults.Best();
                    bestPipeline = bestRegressionIteration.Pipeline;
                    bestModel = bestRegressionIteration.Model;
                    ConsolePrinter.ExperimentResultsHeader(LogLevel.Info, settings.MlTask, settings.Dataset.Name, columnInformation.LabelColumn, settings.MaxExplorationTime.ToString(), regressionRunResults.Count());
                    ConsolePrinter.PrintIterationSummary(regressionRunResults, new RegressionExperimentSettings().OptimizingMetric, 5);
                    break;
                case TaskKind.MulticlassClassification:
                    var bestMultiIteration = multiRunResults.Best();
                    bestPipeline = bestMultiIteration.Pipeline;
                    bestModel = bestMultiIteration.Model;
                    ConsolePrinter.ExperimentResultsHeader(LogLevel.Info, settings.MlTask, settings.Dataset.Name, columnInformation.LabelColumn, settings.MaxExplorationTime.ToString(), multiRunResults.Count());
                    ConsolePrinter.PrintIterationSummary(multiRunResults, new MulticlassExperimentSettings().OptimizingMetric, 5);
                    break;
            }

            // Save the model
            logger.Log(LogLevel.Info, Strings.SavingBestModel);
            var modelprojectDir = Path.Combine(settings.OutputPath.FullName, $"{settings.Name}.Model");
            var modelPath = new FileInfo(Path.Combine(modelprojectDir, "MLModel.zip"));
            Utils.SaveModel(bestModel, modelPath, context);

            // Generate the Project
            GenerateProject(columnInference, bestPipeline, columnInformation.LabelColumn, modelPath);
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
