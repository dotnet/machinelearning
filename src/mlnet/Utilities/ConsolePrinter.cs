// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using NLog;

namespace Microsoft.ML.CLI.Utilities
{
    internal class ConsolePrinter
    {
        private static NLog.Logger logger = NLog.LogManager.GetCurrentClassLogger();


        internal static void PrintBinaryClassificationMetrics(int iteration, string trainerName, BinaryClassificationMetrics metrics)
        {
            logger.Log(LogLevel.Info, $"{iteration,-4} {trainerName,-35} {metrics?.Accuracy ?? double.NaN,9:F4} {metrics?.Auc ?? double.NaN,8:F4} {metrics?.Auprc ?? double.NaN,8:F4} {metrics?.F1Score ?? double.NaN,9:F4}");
        }

        internal static void PrintMulticlassClassificationMetrics(int iteration, string trainerName, MultiClassClassifierMetrics metrics)
        {
            logger.Log(LogLevel.Info, $"{iteration,-4} {trainerName,-35} {metrics?.AccuracyMicro ?? double.NaN,14:F4} {metrics?.AccuracyMacro ?? double.NaN,14:F4}");
        }

        internal static void PrintRegressionMetrics(int iteration, string trainerName, RegressionMetrics metrics)
        {
            logger.Log(LogLevel.Info, $"{iteration,-4} {trainerName,-35} {metrics?.RSquared ?? double.NaN,9:F4} {metrics?.LossFn ?? double.NaN,12:F2} {metrics?.L1 ?? double.NaN,15:F2} {metrics?.L2 ?? double.NaN,15:F2} {metrics?.Rms ?? double.NaN,12:F2}");
        }

        internal static void PrintBinaryClassificationMetricsHeader()
        {
            logger.Log(LogLevel.Info, $"*************************************************");
            logger.Log(LogLevel.Info, $"*       {Strings.MetricsForBinaryClassModels}     ");
            logger.Log(LogLevel.Info, $"*------------------------------------------------");
            logger.Log(LogLevel.Info, $"{" ",-4} {"Trainer",-35} {"Accuracy",9} {"AUC",8} {"AUPRC",8} {"F1-score",9}");
        }

        internal static void PrintMulticlassClassificationMetricsHeader()
        {
            logger.Log(LogLevel.Info, $"*************************************************");
            logger.Log(LogLevel.Info, $"*       {Strings.MetricsForMulticlassModels}     ");
            logger.Log(LogLevel.Info, $"*------------------------------------------------");
            logger.Log(LogLevel.Info, $"{" ",-4} {"Trainer",-35} {"AccuracyMicro",14} {"AccuracyMacro",14}");
        }

        internal static void PrintRegressionMetricsHeader()
        {
            logger.Log(LogLevel.Info, $"*************************************************");
            logger.Log(LogLevel.Info, $"*       {Strings.MetricsForRegressionModels}     ");
            logger.Log(LogLevel.Info, $"*------------------------------------------------");
            logger.Log(LogLevel.Info, $"{" ",-4} {"Trainer",-35} {"R2-Score",9} {"LossFn",12} {"Absolute-loss",15} {"Squared-loss",15} {"RMS-loss",12}");
        }

        internal static void PrintBestPipelineHeader()
        {
            logger.Log(LogLevel.Info, $"*************************************************");
            logger.Log(LogLevel.Info, $"*       {Strings.BestPipeline}      ");
            logger.Log(LogLevel.Info, $"*------------------------------------------------");
}
    }
}
