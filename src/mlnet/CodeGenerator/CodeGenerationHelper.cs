﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML.AutoML;
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
            Stopwatch watch = Stopwatch.StartNew();
            var context = new MLContext();
            context.Log += ConsumeAutoMLSDKLog;

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
            catch (Exception)
            {
                logger.Log(LogLevel.Error, $"{Strings.InferColumnError}");
                throw;
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

            List<RunDetail<BinaryClassificationMetrics>> completedBinaryRuns = new List<RunDetail<BinaryClassificationMetrics>>();
            List<RunDetail<MulticlassClassificationMetrics>> completedMulticlassRuns = new List<RunDetail<MulticlassClassificationMetrics>>();
            List<RunDetail<RegressionMetrics>> completedRegressionRuns = new List<RunDetail<RegressionMetrics>>();

            ProgressHandlers.BinaryClassificationHandler binaryHandler = default;
            ProgressHandlers.RegressionHandler regressionHandler = default;
            ProgressHandlers.MulticlassClassificationHandler multiClassHandler = default;

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

            // TODO the below region needs more refactoring to be done especially with so many switch cases.

            #region RunAutoMLEngine
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
                    Exception ex = null;
                    using (var pbar = new FixedDurationBar(wait, "", options))
                    {
                        pbar.Message = Strings.WaitingForFirstIteration;
                        Thread t = default;
                        switch (taskKind)
                        {
                            // TODO: It may be a good idea to convert the below Threads to Tasks or get rid of this progress bar all together and use an existing one in opensource.
                            case TaskKind.BinaryClassification:
                                binaryHandler = new ProgressHandlers.BinaryClassificationHandler(new BinaryExperimentSettings().OptimizingMetric, completedBinaryRuns, pbar);
                                t = new Thread(() => SafeExecute(() => automlEngine.ExploreBinaryClassificationModels(context, trainData, validationData, columnInformation, new BinaryExperimentSettings().OptimizingMetric, binaryHandler, pbar), out ex, pbar));
                                break;
                            case TaskKind.Regression:
                                regressionHandler = new ProgressHandlers.RegressionHandler(new RegressionExperimentSettings().OptimizingMetric, completedRegressionRuns, pbar);
                                t = new Thread(() => SafeExecute(() => automlEngine.ExploreRegressionModels(context, trainData, validationData, columnInformation, new RegressionExperimentSettings().OptimizingMetric, regressionHandler, pbar), out ex, pbar));
                                break;
                            case TaskKind.MulticlassClassification:
                                multiClassHandler = new ProgressHandlers.MulticlassClassificationHandler(new MulticlassExperimentSettings().OptimizingMetric, completedMulticlassRuns, pbar);
                                t = new Thread(() => SafeExecute(() => automlEngine.ExploreMultiClassificationModels(context, trainData, validationData, columnInformation, new MulticlassExperimentSettings().OptimizingMetric, multiClassHandler, pbar), out ex, pbar));
                                break;
                            default:
                                logger.Log(LogLevel.Error, Strings.UnsupportedMlTask);
                                break;
                        }
                        t.Start();

                        pbar.CompletedHandle.WaitOne(wait);
                        context.Log -= ConsumeAutoMLSDKLog;
                        switch (taskKind)
                        {
                            case TaskKind.BinaryClassification:
                                binaryHandler.Stop();
                                break;
                            case TaskKind.Regression:
                                regressionHandler.Stop();
                                break;
                            case TaskKind.MulticlassClassification:
                                multiClassHandler.Stop();
                                break;
                            default:
                                logger.Log(LogLevel.Error, Strings.UnsupportedMlTask);
                                break;
                        }

                        if (ex != null)
                        {
                            throw ex;
                        }
                    }
                }
                else
                {
                    Exception ex = null;
                    Thread t = default;
                    switch (taskKind)
                    {
                        // TODO: It may be a good idea to convert the below Threads to Tasks or get rid of this progress bar all together and use an existing one in opensource.
                        case TaskKind.BinaryClassification:
                            binaryHandler = new ProgressHandlers.BinaryClassificationHandler(new BinaryExperimentSettings().OptimizingMetric, completedBinaryRuns, null);
                            t = new Thread(() => SafeExecute(() => automlEngine.ExploreBinaryClassificationModels(context, trainData, validationData, columnInformation, new BinaryExperimentSettings().OptimizingMetric, binaryHandler, null), out ex, null));
                            break;
                        case TaskKind.Regression:
                            regressionHandler = new ProgressHandlers.RegressionHandler(new RegressionExperimentSettings().OptimizingMetric, completedRegressionRuns, null);
                            t = new Thread(() => SafeExecute(() => automlEngine.ExploreRegressionModels(context, trainData, validationData, columnInformation, new RegressionExperimentSettings().OptimizingMetric, regressionHandler, null), out ex, null));
                            break;
                        case TaskKind.MulticlassClassification:
                            multiClassHandler = new ProgressHandlers.MulticlassClassificationHandler(new MulticlassExperimentSettings().OptimizingMetric, completedMulticlassRuns, null);
                            t = new Thread(() => SafeExecute(() => automlEngine.ExploreMultiClassificationModels(context, trainData, validationData, columnInformation, new MulticlassExperimentSettings().OptimizingMetric, multiClassHandler, null), out ex, null));
                            break;
                        default:
                            logger.Log(LogLevel.Error, Strings.UnsupportedMlTask);
                            break;
                    }
                    t.Start();
                    Thread.Sleep(wait);
                    context.Log -= ConsumeAutoMLSDKLog;
                    switch (taskKind)
                    {
                        case TaskKind.BinaryClassification:
                            binaryHandler.Stop();
                            break;
                        case TaskKind.Regression:
                            regressionHandler.Stop();
                            break;
                        case TaskKind.MulticlassClassification:
                            multiClassHandler.Stop();
                            break;
                        default:
                            logger.Log(LogLevel.Error, Strings.UnsupportedMlTask);
                            break;
                    }

                    if (ex != null)
                    {
                        throw ex;
                    }
                }
            }
            catch (Exception)
            {
                logger.Log(LogLevel.Error, $"{Strings.ExplorePipelineException}:");
                throw;
            }
            finally
            {
                context.Log -= ConsumeAutoMLSDKLog;
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
                        if (completedBinaryRuns.Count > 0)
                        {
                            var binaryMetric = new BinaryExperimentSettings().OptimizingMetric;
                            var bestBinaryIteration = BestResultUtil.GetBestRun(completedBinaryRuns, binaryMetric);
                            bestPipeline = bestBinaryIteration.Pipeline;
                            bestModel = bestBinaryIteration.Model;
                            ConsolePrinter.ExperimentResultsHeader(LogLevel.Info, settings.MlTask, settings.Dataset.Name, columnInformation.LabelColumnName, elapsedTime.ToString("F2"), completedBinaryRuns.Count());
                            ConsolePrinter.PrintIterationSummary(completedBinaryRuns, binaryMetric, 5);
                        }
                        else
                        {
                            logger.Log(LogLevel.Error, string.Format(Strings.CouldNotFinshOnTime, settings.MaxExplorationTime));
                            logger.Log(LogLevel.Info, Strings.Exiting);
                            return;
                        }
                        break;
                    case TaskKind.Regression:
                        if (completedRegressionRuns.Count > 0)
                        {
                            var regressionMetric = new RegressionExperimentSettings().OptimizingMetric;
                            var bestRegressionIteration = BestResultUtil.GetBestRun(completedRegressionRuns, regressionMetric);
                            bestPipeline = bestRegressionIteration.Pipeline;
                            bestModel = bestRegressionIteration.Model;
                            ConsolePrinter.ExperimentResultsHeader(LogLevel.Info, settings.MlTask, settings.Dataset.Name, columnInformation.LabelColumnName, elapsedTime.ToString("F2"), completedRegressionRuns.Count());
                            ConsolePrinter.PrintIterationSummary(completedRegressionRuns, regressionMetric, 5);
                        }
                        else
                        {
                            logger.Log(LogLevel.Error, string.Format(Strings.CouldNotFinshOnTime, settings.MaxExplorationTime));
                            logger.Log(LogLevel.Info, Strings.Exiting);
                            return;
                        }
                        break;
                    case TaskKind.MulticlassClassification:
                        if (completedMulticlassRuns.Count > 0)
                        {
                            var muliclassMetric = new MulticlassExperimentSettings().OptimizingMetric;
                            var bestMulticlassIteration = BestResultUtil.GetBestRun(completedMulticlassRuns, muliclassMetric);
                            bestPipeline = bestMulticlassIteration.Pipeline;
                            bestModel = bestMulticlassIteration.Model;
                            ConsolePrinter.ExperimentResultsHeader(LogLevel.Info, settings.MlTask, settings.Dataset.Name, columnInformation.LabelColumnName, elapsedTime.ToString("F2"), completedMulticlassRuns.Count());
                            ConsolePrinter.PrintIterationSummary(completedMulticlassRuns, muliclassMetric, 5);
                        }
                        else
                        {
                            logger.Log(LogLevel.Error, string.Format(Strings.CouldNotFinshOnTime, settings.MaxExplorationTime));
                            logger.Log(LogLevel.Info, Strings.Exiting);
                            return;
                        }
                        break;
                }
            }
            catch (Exception)
            {
                logger.Log(LogLevel.Info, Strings.ErrorBestPipeline);
                throw;
            }
            #endregion

            // Save the model
            var modelprojectDir = Path.Combine(settings.OutputPath.FullName, $"{settings.Name}.Model");
            var modelPath = new FileInfo(Path.Combine(modelprojectDir, "MLModel.zip"));

            try
            {
                Utils.SaveModel(bestModel, modelPath, context, trainData.Schema);
                Console.ForegroundColor = ConsoleColor.Yellow;
                logger.Log(LogLevel.Info, $"{Strings.SavingBestModel}: {modelPath}");
            }
            catch (Exception)
            {
                logger.Log(LogLevel.Info, Strings.ErrorSavingModel);
                throw;
            }
            finally
            {
                Console.ResetColor();
            }

            // Generate the Project
            try
            {
                GenerateProject(columnInference, bestPipeline, columnInformation.LabelColumnName, modelPath);
                Console.ForegroundColor = ConsoleColor.Yellow;
                logger.Log(LogLevel.Info, $"{Strings.GenerateModelConsumption}: { Path.Combine(settings.OutputPath.FullName, $"{settings.Name}.ConsoleApp")}");
                logger.Log(LogLevel.Info, $"{Strings.SeeLogFileForMoreInfo}: {settings.LogFilePath}");

            }
            catch (Exception)
            {
                logger.Log(LogLevel.Info, Strings.ErrorGeneratingProject);
                throw;
            }
            finally
            {
                Console.ResetColor();
            }
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

        private static void ConsumeAutoMLSDKLog(object sender, LoggingEventArgs args)
        {
            var logMessage = args.Message;
            if (logMessage.Contains(AutoMLLogger.ChannelName))
            {
                logger.Trace(args.Message);
            }
        }

        private void SafeExecute(Action p, out Exception ex, FixedDurationBar pbar)
        {
            try
            {
                p.Invoke();
                ex = null;
            }
            catch (Exception e)
            {
                ex = e;
                pbar?.Dispose();
                return;
            }
        }
    }
}
