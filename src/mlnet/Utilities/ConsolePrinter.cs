// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;
using NLog;

namespace Microsoft.ML.CLI.Utilities
{
    internal class ConsolePrinter
    {
        private static NLog.Logger logger = NLog.LogManager.GetCurrentClassLogger();


        internal static void PrintMetrics(int iteration, string trainerName, BinaryClassificationMetrics metrics, double bestMetric, double runtimeInSeconds, LogLevel logLevel)
        {
            logger.Log(logLevel, $"{iteration,-4} {trainerName,-35} {metrics?.Accuracy ?? double.NaN,9:F4} {metrics?.AreaUnderRocCurve ?? double.NaN,8:F4} {metrics?.AreaUnderPrecisionRecallCurve ?? double.NaN,8:F4} {metrics?.F1Score ?? double.NaN,9:F4} {bestMetric,8:F4} {runtimeInSeconds,9:F1}");
        }

        internal static void PrintMetrics(int iteration, string trainerName, MulticlassClassificationMetrics metrics, double bestMetric, double runtimeInSeconds, LogLevel logLevel)
        {
            logger.Log(logLevel, $"{iteration,-4} {trainerName,-35} {metrics?.MicroAccuracy ?? double.NaN,14:F4} {metrics?.MicroAccuracy ?? double.NaN,14:F4} {bestMetric,14:F4} {runtimeInSeconds,9:F1}");
        }

        internal static void PrintMetrics(int iteration, string trainerName, RegressionMetrics metrics, double bestMetric, double runtimeInSeconds, LogLevel logLevel)
        {
            logger.Log(logLevel, $"{iteration,-4} {trainerName,-35} {metrics?.RSquared ?? double.NaN,9:F4} {metrics?.LossFunction ?? double.NaN,12:F2} {metrics?.MeanAbsoluteError ?? double.NaN,15:F2} {metrics?.MeanSquaredError ?? double.NaN,15:F2} {metrics?.RootMeanSquaredError ?? double.NaN,12:F2} {bestMetric,12:F4} {runtimeInSeconds,9:F1}");
        }


        internal static void PrintBinaryClassificationMetricsHeader(LogLevel logLevel)
        {
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(logLevel, $"{Strings.MetricsForBinaryClassModels}");
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(logLevel, $"{" ",-4} {"Trainer",-35} {"Accuracy",9} {"AUC",8} {"AUPRC",8} {"F1-score",9} {"Best",8} {"Duration",9}");
        }

        internal static void PrintMulticlassClassificationMetricsHeader(LogLevel logLevel)
        {
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(logLevel, $"{Strings.MetricsForMulticlassModels}");
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(logLevel, $"{" ",-4} {"Trainer",-35} {"AccuracyMicro",14} {"AccuracyMacro",14} {"Best",14} {"Duration",9}");
        }

        internal static void PrintRegressionMetricsHeader(LogLevel logLevel)
        {
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(logLevel, $"{Strings.MetricsForRegressionModels}");
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(logLevel, $"{" ",-4} {"Trainer",-35} {"R2-Score",9} {"LossFn",12} {"Absolute-loss",15} {"Squared-loss",15} {"RMS-loss",12} {"Best",12} {"Duration",9}");
        }

        internal static void PrintBestPipelineHeader(LogLevel logLevel)
        {
            logger.Log(logLevel, $"*************************************************");
            logger.Log(logLevel, $"*       {Strings.BestPipeline}      ");
            logger.Log(logLevel, $"*------------------------------------------------");
        }

        internal static void PrintTopNHeader(int count)
        {
            throw new NotImplementedException();
        }

        internal static void ExperimentResultsHeader(LogLevel logLevel, string mltask, string datasetName, string labelName, string time, int numModelsExplored)
        {
            logger.Log(logLevel, $"==============================================Experiment Results==================================================");
            logger.Log(logLevel, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(logLevel, $"{"ML Task",-7} : {mltask,-20}");
            logger.Log(logLevel, $"{"Dataset",-7}: {datasetName,-25}");
            logger.Log(logLevel, $"{"Label",-6} : {labelName,-25}");
            logger.Log(logLevel, $"{"Exploration time",-20} : {time} Secs");
            logger.Log(logLevel, $"{"Total number of models explored",-30} : {numModelsExplored}");
            logger.Log(logLevel, $"------------------------------------------------------------------------------------------------------------------");
        }
        internal static void PrintIterationSummary(IEnumerable<RunDetails<BinaryClassificationMetrics>> results, BinaryClassificationMetric optimizationMetric, int count)
        {
            var metricsAgent = new BinaryMetricsAgent(null, optimizationMetric);
            var topNResults = BestResultUtil.GetTopNRunResults(results, metricsAgent, count, new OptimizingMetricInfo(optimizationMetric).IsMaximizing);
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(LogLevel.Info, $"Top {topNResults?.Count()} models explored                                          ");
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            PrintBinaryClassificationMetricsHeader(LogLevel.Info);
            int i = 0;
            foreach (var result in topNResults)
            {
                PrintMetrics(++i, result.TrainerName, result.ValidationMetrics, metricsAgent.GetScore(result.ValidationMetrics), result.RuntimeInSeconds, LogLevel.Info);
            }
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
        }

        internal static void PrintIterationSummary(IEnumerable<RunDetails<RegressionMetrics>> results, RegressionMetric optimizationMetric, int count)
        {
            var metricsAgent = new RegressionMetricsAgent(null, optimizationMetric);
            var topNResults = BestResultUtil.GetTopNRunResults(results, metricsAgent, count, new OptimizingMetricInfo(optimizationMetric).IsMaximizing);
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(LogLevel.Info, $"Top {topNResults?.Count()} models explored                                          ");
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            PrintRegressionMetricsHeader(LogLevel.Info);
            int i = 0;
            foreach (var result in topNResults)
            {
                PrintMetrics(++i, result.TrainerName, result.ValidationMetrics, metricsAgent.GetScore(result.ValidationMetrics), result.RuntimeInSeconds, LogLevel.Info);
            }
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
        }

        internal static void PrintIterationSummary(IEnumerable<RunDetails<MulticlassClassificationMetrics>> results, MulticlassClassificationMetric optimizationMetric, int count)
        {
            var metricsAgent = new MultiMetricsAgent(null, optimizationMetric);
            var topNResults = BestResultUtil.GetTopNRunResults(results, metricsAgent, count, new OptimizingMetricInfo(optimizationMetric).IsMaximizing);
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            logger.Log(LogLevel.Info, $"Top {topNResults?.Count()} models explored                                          ");
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
            PrintMulticlassClassificationMetricsHeader(LogLevel.Info);
            int i = 0;
            foreach (var result in topNResults)
            {
                PrintMetrics(++i, result.TrainerName, result.ValidationMetrics, metricsAgent.GetScore(result.ValidationMetrics), result.RuntimeInSeconds, LogLevel.Info);
            }
            logger.Log(LogLevel.Info, $"------------------------------------------------------------------------------------------------------------------");
        }
    }
}

