// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using NLog;

namespace Microsoft.ML.CLI.Utilities
{
    internal class ConsolePrinter
    {
        private const int Width = 114;
        private static NLog.Logger logger = NLog.LogManager.GetCurrentClassLogger();
        internal static readonly string TABLESEPERATOR = "------------------------------------------------------------------------------------------------------------------";

        internal static void PrintMetrics(int iteration, string trainerName, BinaryClassificationMetrics metrics, double bestMetric, double? runtimeInSeconds, LogLevel logLevel, int iterationNumber = -1)
        {
            logger.Log(logLevel, CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.Accuracy ?? double.NaN,9:F4} {metrics?.AreaUnderRocCurve ?? double.NaN,8:F4} {metrics?.AreaUnderPrecisionRecallCurve ?? double.NaN,8:F4} {metrics?.F1Score ?? double.NaN,9:F4} {runtimeInSeconds.Value,9:F1} {iterationNumber + 1,10}", Width));
        }

        internal static void PrintMetrics(int iteration, string trainerName, MulticlassClassificationMetrics metrics, double bestMetric, double? runtimeInSeconds, LogLevel logLevel, int iterationNumber = -1)
        {
            logger.Log(logLevel, CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.MicroAccuracy ?? double.NaN,14:F4} {metrics?.MacroAccuracy ?? double.NaN,14:F4} {runtimeInSeconds.Value,9:F1} {iterationNumber + 1,10}", Width));
        }

        internal static void PrintMetrics(int iteration, string trainerName, RegressionMetrics metrics, double bestMetric, double? runtimeInSeconds, LogLevel logLevel, int iterationNumber = -1)
        {
            logger.Log(logLevel, CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.RSquared ?? double.NaN,8:F4} {metrics?.MeanAbsoluteError ?? double.NaN,13:F2} {metrics?.MeanSquaredError ?? double.NaN,12:F2} {metrics?.RootMeanSquaredError ?? double.NaN,8:F2} {runtimeInSeconds.Value,9:F1} {iterationNumber + 1,10}", Width));
        }

        internal static void PrintBinaryClassificationMetricsHeader(LogLevel logLevel)
        {
            logger.Log(logLevel, CreateRow($"{"",-4} {"Trainer",-35} {"Accuracy",9} {"AUC",8} {"AUPRC",8} {"F1-score",9} {"Duration",9} {"#Iteration",10}", Width));
        }

        internal static void PrintMulticlassClassificationMetricsHeader(LogLevel logLevel)
        {
            logger.Log(logLevel, CreateRow($"{"",-4} {"Trainer",-35} {"MicroAccuracy",14} {"MacroAccuracy",14} {"Duration",9} {"#Iteration",10}", Width));
        }

        internal static void PrintRegressionMetricsHeader(LogLevel logLevel)
        {
            logger.Log(logLevel, CreateRow($"{"",-4} {"Trainer",-35} {"RSquared",8} {"Absolute-loss",13} {"Squared-loss",12} {"RMS-loss",8} {"Duration",9} {"#Iteration",10}", Width));
        }

        internal static void ExperimentResultsHeader(LogLevel logLevel, string mltask, string datasetName, string labelName, string time, int numModelsExplored)
        {
            logger.Log(logLevel, string.Empty);
            logger.Log(logLevel, $"===============================================Experiment Results=================================================");
            logger.Log(logLevel, TABLESEPERATOR);
            var header = "Summary";
            logger.Log(logLevel, CreateRow(header.PadLeft((Width / 2) + header.Length / 2), Width));
            logger.Log(logLevel, TABLESEPERATOR);
            logger.Log(logLevel, CreateRow($"{"ML Task",-7}: {mltask,-20}", Width));
            logger.Log(logLevel, CreateRow($"{"Dataset",-7}: {datasetName,-25}", Width));
            logger.Log(logLevel, CreateRow($"{"Label",-6}: {labelName,-25}", Width));
            logger.Log(logLevel, CreateRow($"{"Total experiment time",-22}: {time} Secs", Width));
            logger.Log(logLevel, CreateRow($"{"Total number of models explored",-30}: {numModelsExplored}", Width));
            logger.Log(logLevel, TABLESEPERATOR);
        }

        internal static string CreateRow(string message, int width)
        {
            return "|" + message.PadRight(width - 2) + "|";
        }

        internal static void PrintIterationSummary(IEnumerable<RunDetail<BinaryClassificationMetrics>> results, BinaryClassificationMetric optimizationMetric, int count)
        {
            var metricsAgent = new BinaryMetricsAgent(null, optimizationMetric);
            var topNResults = BestResultUtil.GetTopNRunResults(results, metricsAgent, count, new OptimizingMetricInfo(optimizationMetric).IsMaximizing);
            var header = $"Top {topNResults?.Count()} models explored";
            logger.Log(LogLevel.Info, CreateRow(header.PadLeft((Width / 2) + header.Length / 2), Width));
            logger.Log(LogLevel.Info, TABLESEPERATOR);

            PrintBinaryClassificationMetricsHeader(LogLevel.Info);
            int i = 0;
            foreach (var pair in topNResults)
            {
                var result = pair.Item1;
                if (i == 0)
                {
                    // Print top iteration colored.
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    PrintMetrics(++i, result?.TrainerName, result?.ValidationMetrics, metricsAgent.GetScore(result?.ValidationMetrics), result?.RuntimeInSeconds, LogLevel.Info, pair.Item2);
                    Console.ResetColor();
                    continue;
                }
                PrintMetrics(++i, result?.TrainerName, result?.ValidationMetrics, metricsAgent.GetScore(result?.ValidationMetrics), result?.RuntimeInSeconds, LogLevel.Info, pair.Item2);
            }
            logger.Log(LogLevel.Info, TABLESEPERATOR);
        }

        internal static void PrintIterationSummary(IEnumerable<RunDetail<RegressionMetrics>> results, RegressionMetric optimizationMetric, int count)
        {
            var metricsAgent = new RegressionMetricsAgent(null, optimizationMetric);
            var topNResults = BestResultUtil.GetTopNRunResults(results, metricsAgent, count, new OptimizingMetricInfo(optimizationMetric).IsMaximizing);
            var header = $"Top {topNResults?.Count()} models explored";
            logger.Log(LogLevel.Info, CreateRow(header.PadLeft((Width / 2) + header.Length / 2), Width));
            logger.Log(LogLevel.Info, TABLESEPERATOR);

            PrintRegressionMetricsHeader(LogLevel.Info);
            int i = 0;
            foreach (var pair in topNResults)
            {
                var result = pair.Item1;
                if (i == 0)
                {
                    // Print top iteration colored.
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    PrintMetrics(++i, result?.TrainerName, result?.ValidationMetrics, metricsAgent.GetScore(result?.ValidationMetrics), result?.RuntimeInSeconds, LogLevel.Info, pair.Item2);
                    Console.ResetColor();
                    continue;
                }
                PrintMetrics(++i, result?.TrainerName, result?.ValidationMetrics, metricsAgent.GetScore(result?.ValidationMetrics), result?.RuntimeInSeconds, LogLevel.Info, pair.Item2);
            }
            logger.Log(LogLevel.Info, TABLESEPERATOR);
        }

        internal static void PrintIterationSummary(IEnumerable<RunDetail<MulticlassClassificationMetrics>> results, MulticlassClassificationMetric optimizationMetric, int count)
        {
            var metricsAgent = new MultiMetricsAgent(null, optimizationMetric);
            var topNResults = BestResultUtil.GetTopNRunResults(results, metricsAgent, count, new OptimizingMetricInfo(optimizationMetric).IsMaximizing);
            var header = $"Top {topNResults?.Count()} models explored";
            logger.Log(LogLevel.Info, CreateRow(header.PadLeft((Width / 2) + header.Length / 2), Width));
            logger.Log(LogLevel.Info, TABLESEPERATOR);
            PrintMulticlassClassificationMetricsHeader(LogLevel.Info);
            int i = 0;
            foreach (var pair in topNResults)
            {
                var result = pair.Item1;
                if (i == 0)
                {
                    // Print top iteration colored.
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    PrintMetrics(++i, result?.TrainerName, result?.ValidationMetrics, metricsAgent.GetScore(result?.ValidationMetrics), result?.RuntimeInSeconds, LogLevel.Info, pair.Item2);
                    Console.ResetColor();
                    continue;
                }
                PrintMetrics(++i, result?.TrainerName, result?.ValidationMetrics, metricsAgent.GetScore(result?.ValidationMetrics), result?.RuntimeInSeconds, LogLevel.Info, pair.Item2);
            }
            logger.Log(LogLevel.Info, TABLESEPERATOR);
        }
        internal static void PrintException(Exception e, LogLevel logLevel)
        {
            logger.Log(logLevel, e.ToString());
        }
    }
}

