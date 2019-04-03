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

        internal class RegressionHandler : IProgress<RunResult<RegressionMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunResult<RegressionMetrics>, double> GetScore;
            private RunResult<RegressionMetrics> bestResult;
            private int iterationIndex;
            private ProgressBar progressBar;
            private string optimizationMetric = string.Empty;

            public RegressionHandler(RegressionMetric optimizationMetric, ShellProgressBar.ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric.ToString();
                this.progressBar = progressBar;
                GetScore = (RunResult<RegressionMetrics> result) => new RegressionMetricsAgent(optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintRegressionMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunResult<RegressionMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics, GetScore(bestResult), iterationResult.RuntimeInSeconds, LogLevel.Trace);
            }

            private void UpdateBestResult(RunResult<RegressionMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                {
                    bestResult = iterationResult;
                    progressBar.Message = $"Best {this.optimizationMetric} : {GetScore(bestResult):F2} , Best Algorithm : {bestResult.TrainerName}, Last Algorithm : {bestResult.TrainerName}";
                }
                else
                {
                    progressBar.Message = $"Best {this.optimizationMetric} : {GetScore(bestResult):F2} , Best Algorithm : {bestResult.TrainerName}, Last Algorithm : {iterationResult.TrainerName}";
                }
            }
        }

        internal class BinaryClassificationHandler : IProgress<RunResult<BinaryClassificationMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunResult<BinaryClassificationMetrics>, double> GetScore;
            private RunResult<BinaryClassificationMetrics> bestResult;
            private int iterationIndex;
            private ProgressBar progressBar;
            private string optimizationMetric = string.Empty;

            public BinaryClassificationHandler(BinaryClassificationMetric optimizationMetric, ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric.ToString();
                this.progressBar = progressBar;
                GetScore = (RunResult<BinaryClassificationMetrics> result) => new BinaryMetricsAgent(optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintBinaryClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunResult<BinaryClassificationMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics, GetScore(bestResult), iterationResult.RuntimeInSeconds, LogLevel.Trace);
            }

            private void UpdateBestResult(RunResult<BinaryClassificationMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                {
                    bestResult = iterationResult;
                    progressBar.Message = $"Best {this.optimizationMetric} : {GetScore(bestResult):F2} , Best Algorithm : {bestResult.TrainerName}, Last Algorithm : {bestResult.TrainerName}";
                }
                else
                {
                    progressBar.Message = $"Best {this.optimizationMetric} : {GetScore(bestResult):F2} , Best Algorithm : {bestResult.TrainerName}, Last Algorithm : {iterationResult.TrainerName}";
                }
            }
        }

        internal class MulticlassClassificationHandler : IProgress<RunResult<MulticlassClassificationMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunResult<MulticlassClassificationMetrics>, double> GetScore;
            private RunResult<MulticlassClassificationMetrics> bestResult;
            private int iterationIndex;
            private ProgressBar progressBar;
            private string optimizationMetric = string.Empty;

            public MulticlassClassificationHandler(MulticlassClassificationMetric optimizationMetric, ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric.ToString();
                this.progressBar = progressBar;
                GetScore = (RunResult<MulticlassClassificationMetrics> result) => new MultiMetricsAgent(optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintMulticlassClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunResult<MulticlassClassificationMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics, GetScore(bestResult), iterationResult.RuntimeInSeconds, LogLevel.Trace);
            }

            private void UpdateBestResult(RunResult<MulticlassClassificationMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                {
                    bestResult = iterationResult;
                    progressBar.Message = $"Best {this.optimizationMetric} : {GetScore(bestResult):F2} , Best Algorithm : {bestResult.TrainerName}, Last Algorithm : {bestResult.TrainerName}";
                }
                else
                {
                    progressBar.Message = $"Best {this.optimizationMetric} : {GetScore(bestResult):F2} , Best Algorithm : {bestResult.TrainerName}, Last Algorithm : {iterationResult.TrainerName}";
                }
            }
        }

    }
}
