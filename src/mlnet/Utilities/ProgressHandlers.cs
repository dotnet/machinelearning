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

        internal class RegressionHandler : IProgress<RunDetails<RegressionMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunDetails<RegressionMetrics>, double> GetScore;
            private RunDetails<RegressionMetrics> bestResult;
            private int iterationIndex;
            private ProgressBar progressBar;
            private string optimizationMetric = string.Empty;

            public RegressionHandler(RegressionMetric optimizationMetric, ShellProgressBar.ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric.ToString();
                this.progressBar = progressBar;
                GetScore = (RunDetails<RegressionMetrics> result) => new RegressionMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintRegressionMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetails<RegressionMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                progressBar.Message = $"Best {this.optimizationMetric}: {GetScore(bestResult):F4}, Best Algorithm: {bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, GetScore(bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                if (iterationResult.Exception != null)
                {
                    ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                }
            }

            private void UpdateBestResult(RunDetails<RegressionMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                {
                    bestResult = iterationResult;
                }
            }
        }

        internal class BinaryClassificationHandler : IProgress<RunDetails<BinaryClassificationMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunDetails<BinaryClassificationMetrics>, double> GetScore;
            private RunDetails<BinaryClassificationMetrics> bestResult;
            private int iterationIndex;
            private ProgressBar progressBar;
            private string optimizationMetric = string.Empty;

            public BinaryClassificationHandler(BinaryClassificationMetric optimizationMetric, ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric.ToString();
                this.progressBar = progressBar;
                GetScore = (RunDetails<BinaryClassificationMetrics> result) => new BinaryMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintBinaryClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetails<BinaryClassificationMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                progressBar.Message = $"Best {this.optimizationMetric}: {GetScore(bestResult):F4}, Best Algorithm: {bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, GetScore(bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                if (iterationResult.Exception != null)
                {
                    ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                }
            }

            private void UpdateBestResult(RunDetails<BinaryClassificationMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                {
                    bestResult = iterationResult;
                }
            }
        }

        internal class MulticlassClassificationHandler : IProgress<RunDetails<MulticlassClassificationMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunDetails<MulticlassClassificationMetrics>, double> GetScore;
            private RunDetails<MulticlassClassificationMetrics> bestResult;
            private int iterationIndex;
            private ProgressBar progressBar;
            private string optimizationMetric = string.Empty;

            public MulticlassClassificationHandler(MulticlassClassificationMetric optimizationMetric, ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric.ToString();
                this.progressBar = progressBar;
                GetScore = (RunDetails<MulticlassClassificationMetrics> result) => new MultiMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintMulticlassClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetails<MulticlassClassificationMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                progressBar.Message = $"Best {this.optimizationMetric}: {GetScore(bestResult):F4}, Best Algorithm: {bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, GetScore(bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                if (iterationResult.Exception != null)
                {
                    ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                }
            }

            private void UpdateBestResult(RunDetails<MulticlassClassificationMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                {
                    bestResult = iterationResult;
                }
            }
        }

    }
}