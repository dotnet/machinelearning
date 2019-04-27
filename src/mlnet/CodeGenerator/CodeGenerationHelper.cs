// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.AutoML;
using Microsoft.ML.CLI.CodeGenerator.CSharp;
using Microsoft.ML.CLI.Data;
using Microsoft.ML.CLI.ShellProgressBar;
using Microsoft.ML.CLI.Utilities;
using Microsoft.ML.Data;
using NLog;
using NLog.Targets;

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
            Stopwatch watch = Stopwatch.StartNew();
            var context = new MLContext();
            ConsumeAutoMLSDKLogs(context);

            var verboseLevel = Utils.GetVerbosity(settings.Verbosity);

            // Infer columns
            ColumnInferenceResults columnInference = null;
            try
            {
                var inputColumnInformation = new ColumnInformation();
                inputColumnInformation.LabelColumnName = settings.LabelColumnName;
                foreach (var value in settings.IgnoreColumns)
                {
                    inputColumnInformation.IgnoredColumnNames.Add(value);
                }
                columnInference = automlEngine.InferColumns(context, inputColumnInformation);
            }
            catch (Exception e)
            {
                logger.Log(LogLevel.Error, $"{Strings.InferColumnError}");
                logger.Log(LogLevel.Error, e.Message);
                logger.Log(LogLevel.Trace, e.ToString());
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

            ExperimentResult<BinaryClassificationMetrics> binaryExperimentResult = default;
            ExperimentResult<MulticlassClassificationMetrics> multiExperimentResult = default;
            ExperimentResult<RegressionMetrics> regressionExperimentResult = default;
            if (verboseLevel > LogLevel.Trace)
            {
                Console.Write($"{Strings.ExplorePipeline}: ");
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"{settings.MlTask}");
                Console.ResetColor();
                Console.Write($"{Strings.FurtherLearning}: ");
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"{ Strings.LearningHttpLink}");
                Console.ResetColor();
            }

            logger.Log(LogLevel.Trace, $"{Strings.ExplorePipeline}: {settings.MlTask}");
            logger.Log(LogLevel.Trace, $"{Strings.FurtherLearning}: {Strings.LearningHttpLink}");
            try
            {
                var options = new ProgressBarOptions
                {
                    ForegroundColor = ConsoleColor.Yellow,
                    ForegroundColorDone = ConsoleColor.Yellow,
                    BackgroundColor = ConsoleColor.Gray,
                    ProgressCharacter = '\u2593',
                    BackgroundCharacter = '─',
                };
                var wait = TimeSpan.FromSeconds(settings.MaxExplorationTime);

                if (verboseLevel > LogLevel.Trace && !Console.IsOutputRedirected)
                {
                    using (var pbar = new FixedDurationBar(wait, "", options))
                    {
                        pbar.Message = Strings.WaitingForFirstIteration;
                        Thread t = default;
                        switch (taskKind)
                        {
                            case TaskKind.BinaryClassification:
                                t = new Thread(() => binaryExperimentResult = automlEngine.ExploreBinaryClassificationModels(context, trainData, validationData, columnInformation, new BinaryExperimentSettings().OptimizingMetric, pbar));
                                break;
                            case TaskKind.Regression:
                                t = new Thread(() => regressionExperimentResult = automlEngine.ExploreRegressionModels(context, trainData, validationData, columnInformation, new RegressionExperimentSettings().OptimizingMetric, pbar));
                                break;
                            case TaskKind.MulticlassClassification:
                                t = new Thread(() => multiExperimentResult = automlEngine.ExploreMultiClassificationModels(context, trainData, validationData, columnInformation, new MulticlassExperimentSettings().OptimizingMetric, pbar));
                                break;
                            default:
                                logger.Log(LogLevel.Error, Strings.UnsupportedMlTask);
                                break;
                        }
                        t.Start();

                        if (!pbar.CompletedHandle.WaitOne(wait))
                            pbar.Message = $"{nameof(FixedDurationBar)} did not signal {nameof(FixedDurationBar.CompletedHandle)} after {wait}";

                        if (t.IsAlive == true)
                        {
                            string waitingMessage = Strings.WaitingForLastIteration;
                            string originalMessage = pbar.Message;
                            pbar.Message = waitingMessage;
                            t.Join();
                            if (waitingMessage.Equals(pbar.Message))
                            {
                                // Corner cases where thread was alive but has completed all iterations.
                                pbar.Message = originalMessage;
                            }
                        }
                    }
                }
                else
                {
                    switch (taskKind)
                    {
                        case TaskKind.BinaryClassification:
                            binaryExperimentResult = automlEngine.ExploreBinaryClassificationModels(context, trainData, validationData, columnInformation, new BinaryExperimentSettings().OptimizingMetric);
                            break;
                        case TaskKind.Regression:
                            regressionExperimentResult = automlEngine.ExploreRegressionModels(context, trainData, validationData, columnInformation, new RegressionExperimentSettings().OptimizingMetric);
                            break;
                        case TaskKind.MulticlassClassification:
                            multiExperimentResult = automlEngine.ExploreMultiClassificationModels(context, trainData, validationData, columnInformation, new MulticlassExperimentSettings().OptimizingMetric);
                            break;
                        default:
                            logger.Log(LogLevel.Error, Strings.UnsupportedMlTask);
                            break;
                    }
                }


            }
            catch (Exception e)
            {
                logger.Log(LogLevel.Error, $"{Strings.ExplorePipelineException}:");
                logger.Log(LogLevel.Error, e.Message);
                logger.Log(LogLevel.Debug, e.ToString());
                logger.Log(LogLevel.Info, Strings.LookIntoLogFile);
                logger.Log(LogLevel.Error, Strings.Exiting);
                return;
            }

            var elapsedTime = watch.Elapsed.TotalSeconds;

            //Get the best pipeline
            Pipeline bestPipeline = null;
            ITransformer bestModel = null;
            try
            {
                switch (taskKind)
                {
                    case TaskKind.BinaryClassification:
                        var bestBinaryIteration = binaryExperimentResult.BestRun;
                        bestPipeline = bestBinaryIteration.Pipeline;
                        bestModel = bestBinaryIteration.Model;
                        ConsolePrinter.ExperimentResultsHeader(LogLevel.Info, settings.MlTask, settings.Dataset.Name, columnInformation.LabelColumnName, elapsedTime.ToString("F2"), binaryExperimentResult.RunDetails.Count());
                        ConsolePrinter.PrintIterationSummary(binaryExperimentResult.RunDetails, new BinaryExperimentSettings().OptimizingMetric, 5);
                        break;
                    case TaskKind.Regression:
                        var bestRegressionIteration = regressionExperimentResult.BestRun;
                        bestPipeline = bestRegressionIteration.Pipeline;
                        bestModel = bestRegressionIteration.Model;
                        ConsolePrinter.ExperimentResultsHeader(LogLevel.Info, settings.MlTask, settings.Dataset.Name, columnInformation.LabelColumnName, elapsedTime.ToString("F2"), regressionExperimentResult.RunDetails.Count());
                        ConsolePrinter.PrintIterationSummary(regressionExperimentResult.RunDetails, new RegressionExperimentSettings().OptimizingMetric, 5);
                        break;
                    case TaskKind.MulticlassClassification:
                        var bestMultiIteration = multiExperimentResult.BestRun;
                        bestPipeline = bestMultiIteration.Pipeline;
                        bestModel = bestMultiIteration.Model;
                        ConsolePrinter.ExperimentResultsHeader(LogLevel.Info, settings.MlTask, settings.Dataset.Name, columnInformation.LabelColumnName, elapsedTime.ToString("F2"), multiExperimentResult.RunDetails.Count());
                        ConsolePrinter.PrintIterationSummary(multiExperimentResult.RunDetails, new MulticlassExperimentSettings().OptimizingMetric, 5);
                        break;
                }
            }
            catch (Exception e)
            {
                logger.Log(LogLevel.Info, Strings.ErrorBestPipeline);
                logger.Log(LogLevel.Info, e.Message);
                logger.Log(LogLevel.Trace, e.ToString());
                logger.Log(LogLevel.Info, Strings.LookIntoLogFile);
                logger.Log(LogLevel.Error, Strings.Exiting);
                return;
            }

            // Save the model
            var modelprojectDir = Path.Combine(settings.OutputPath.FullName, $"{settings.Name}.Model");
            var modelPath = new FileInfo(Path.Combine(modelprojectDir, "MLModel.zip"));
            Utils.SaveModel(bestModel, modelPath, context, trainData.Schema);
            Console.ForegroundColor = ConsoleColor.Yellow;
            logger.Log(LogLevel.Info, $"{Strings.SavingBestModel}: {modelPath}");

            // Generate the Project
            GenerateProject(columnInference, bestPipeline, columnInformation.LabelColumnName, modelPath);
            logger.Log(LogLevel.Info, $"{Strings.GenerateModelConsumption}: { Path.Combine(settings.OutputPath.FullName, $"{settings.Name}.ConsoleApp")}");
            logger.Log(LogLevel.Info, $"{Strings.SeeLogFileForMoreInfo}: {settings.LogFilePath}");
            Console.ResetColor();
        }

        internal void GenerateProject(ColumnInferenceResults columnInference, Pipeline pipeline, string labelName, FileInfo modelPath)
        {
            // Generate code
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
            logger.Log(LogLevel.Trace, Strings.CreateDataLoader);
            var textLoader = context.Data.CreateTextLoader(textLoaderOptions);

            logger.Log(LogLevel.Trace, Strings.LoadData);
            var trainData = textLoader.Load(settings.Dataset.FullName);
            var validationData = settings.ValidationDataset == null ? null : textLoader.Load(settings.ValidationDataset.FullName);

            return (trainData, validationData);
        }

        private void ConsumeAutoMLSDKLogs(MLContext context)
        {
            context.Log += (object sender, LoggingEventArgs loggingEventArgs) =>
            {
                var logMessage = loggingEventArgs.Message;
                if (logMessage.Contains(AutoMLLogger.ChannelName))
                {
                    logger.Trace(loggingEventArgs.Message);
                }
            };
        }
    }
}
