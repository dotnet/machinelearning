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
        internal static void PrintRegressionMetrics(int iteration, string trainerName, RegressionMetrics metrics)
        {
            logger.Log(LogLevel.Info, $"{iteration,-3}{trainerName,-35}{metrics.RSquared,-10:0.###}{metrics.LossFn,-8:0.##}{metrics.L1,-15:#.##}{metrics.L2,-15:#.##}{metrics.Rms,-10:#.##}");
        }

        internal static void PrintBinaryClassificationMetrics(int iteration, string trainerName, BinaryClassificationMetrics metrics)
        {
            logger.Log(LogLevel.Info, $"{iteration,-3}{trainerName,-35}{metrics.Accuracy,-10:0.###}{metrics.Auc,-8:0.##}");
        }

        internal static void PrintBinaryClassificationMetricsHeader()
        {
            logger.Log(LogLevel.Info, $"*************************************************");
            logger.Log(LogLevel.Info, $"*       {Strings.MetricsForBinaryClassModels}     ");
            logger.Log(LogLevel.Info, $"*------------------------------------------------");
            logger.Log(LogLevel.Info, $"{" ",-3}{"Trainer",-35}{"Accuracy",-10}{"Auc",-8}");
        }

        internal static void PrintRegressionMetricsHeader()
        {
            logger.Log(LogLevel.Info, $"*************************************************");
            logger.Log(LogLevel.Info, $"*       {Strings.MetricsForRegressionModels}     ");
            logger.Log(LogLevel.Info, $"*------------------------------------------------");
            logger.Log(LogLevel.Info, $"{" ",-3}{"Trainer",-35}{"R2-Score",-10}{"LossFn",-8}{"Absolute-loss",-15}{"Squared-loss",-15}{"RMS-loss",-10}");
        }

        internal static void PrintBestPipelineHeader()
        {
            logger.Log(LogLevel.Info, $"*************************************************");
            logger.Log(LogLevel.Info, $"*       {Strings.BestPipeline}      ");
            logger.Log(LogLevel.Info, $"*------------------------------------------------");
        }
    }
}
