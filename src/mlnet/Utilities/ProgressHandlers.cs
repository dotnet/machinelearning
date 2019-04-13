// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.ShellProgressBar;
using Microsoft.ML.Data;
using NLog;

namespace Microsoft.ML.CLI.Utilities
{
    internal class ProgressHandlers
    {
        private static int MetricComparator(double a, double b, bool isMaximizing)
        {
            return (isMaximizing ? a.CompareTo(b) : -a.CompareTo(b));
        }

        internal class RegressionHandler : IProgress<RunDetail<RegressionMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunDetail<RegressionMetrics>, double> GetScore;
            private RunDetail<RegressionMetrics> bestResult;
            private int iterationIndex;
            private ProgressBar progressBar;
            private string optimizationMetric = string.Empty;

            public RegressionHandler(RegressionMetric optimizationMetric, ShellProgressBar.ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric.ToString();
                this.progressBar = progressBar;
                GetScore = (RunDetail<RegressionMetrics> result) => new RegressionMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintRegressionMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetail<RegressionMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                if (progressBar != null)
                    progressBar.Message = $"Best {this.optimizationMetric}: {GetScore(bestResult):F4}, Best Algorithm: {bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, GetScore(bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                if (iterationResult.Exception != null)
                {
                    ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                }
            }

            private void UpdateBestResult(RunDetail<RegressionMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                {
                    bestResult = iterationResult;
                }
            }
        }

        internal class BinaryClassificationHandler : IProgress<RunDetail<BinaryClassificationMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunDetail<BinaryClassificationMetrics>, double> GetScore;
            private RunDetail<BinaryClassificationMetrics> bestResult;
            private int iterationIndex;
            private ProgressBar progressBar;
            private BinaryClassificationMetric optimizationMetric;

            public BinaryClassificationHandler(BinaryClassificationMetric optimizationMetric, ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric;
                this.progressBar = progressBar;
                GetScore = (RunDetail<BinaryClassificationMetrics> result) => new BinaryMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintBinaryClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetail<BinaryClassificationMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                if (progressBar != null)
                    progressBar.Message = GetProgressBarMessage(iterationResult);
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, GetScore(bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                if (iterationResult.Exception != null)
                {
                    ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                }
            }

            private string GetProgressBarMessage(RunDetail<BinaryClassificationMetrics> iterationResult)
            {
                if (optimizationMetric == BinaryClassificationMetric.Accuracy)
                {
                    return $"Best Accuracy: {GetScore(bestResult) * 100:F2}%, Best Algorithm: {bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                }

                return $"Best {this.optimizationMetric}: {GetScore(bestResult):F4}, Best Algorithm: {bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
            }

            private void UpdateBestResult(RunDetail<BinaryClassificationMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                {
                    bestResult = iterationResult;
                }
            }
        }

        internal class MulticlassClassificationHandler : IProgress<RunDetail<MulticlassClassificationMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunDetail<MulticlassClassificationMetrics>, double> GetScore;
            private RunDetail<MulticlassClassificationMetrics> bestResult;
            private int iterationIndex;
            private ProgressBar progressBar;
            private MulticlassClassificationMetric optimizationMetric;

            public MulticlassClassificationHandler(MulticlassClassificationMetric optimizationMetric, ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric;
                this.progressBar = progressBar;
                GetScore = (RunDetail<MulticlassClassificationMetrics> result) => new MultiMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintMulticlassClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetail<MulticlassClassificationMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                if (progressBar != null)
                    progressBar.Message = GetProgressBarMessage(iterationResult);
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, GetScore(bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                if (iterationResult.Exception != null)
                {
                    ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                }
            }

            private void UpdateBestResult(RunDetail<MulticlassClassificationMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                {
                    bestResult = iterationResult;
                }
            }

            private string GetProgressBarMessage(RunDetail<MulticlassClassificationMetrics> iterationResult)
            {
                if (optimizationMetric == MulticlassClassificationMetric.MicroAccuracy)
                {
                    return $"Best Accuracy: {GetScore(bestResult) * 100:F2}%, Best Algorithm: {bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                }

                return $"Best {this.optimizationMetric}: {GetScore(bestResult):F4}, Best Algorithm: {bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
            }
        }

    }
}