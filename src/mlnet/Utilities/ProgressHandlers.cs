// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.AutoML;
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
            private List<RunDetail<RegressionMetrics>> completedIterations;
            private ProgressBar progressBar;
            private string optimizationMetric = string.Empty;
            private bool isStopped;

            public RegressionHandler(RegressionMetric optimizationMetric, List<RunDetail<RegressionMetrics>> completedIterations, ShellProgressBar.ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric.ToString();
                this.completedIterations = completedIterations;
                this.progressBar = progressBar;
                GetScore = (RunDetail<RegressionMetrics> result) => new RegressionMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintRegressionMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetail<RegressionMetrics> iterationResult)
            {
                lock (this)
                {
                    if (this.isStopped)
                        return;

                    iterationIndex++;
                    completedIterations.Add(iterationResult);
                    UpdateBestResult(iterationResult);
                    if (progressBar != null)
                        progressBar.Message = $"Best quality({this.optimizationMetric}): {GetScore(bestResult):F4}, Best Algorithm: {bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                    ConsolePrinter.PrintMetrics(iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, GetScore(bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                    if (iterationResult.Exception != null)
                    {
                        ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                    }
                }
            }
            public void Stop()
            {
                lock (this)
                {
                    this.isStopped = true;
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
            private List<RunDetail<BinaryClassificationMetrics>> completedIterations;
            private bool isStopped;

            public BinaryClassificationHandler(BinaryClassificationMetric optimizationMetric, List<RunDetail<BinaryClassificationMetrics>> completedIterations, ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric;
                this.completedIterations = completedIterations;
                this.progressBar = progressBar;
                GetScore = (RunDetail<BinaryClassificationMetrics> result) => new BinaryMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintBinaryClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetail<BinaryClassificationMetrics> iterationResult)
            {
                lock (this)
                {
                    if (this.isStopped)
                        return;
                    iterationIndex++;
                    completedIterations.Add(iterationResult);
                    UpdateBestResult(iterationResult);
                    if (progressBar != null)
                        progressBar.Message = GetProgressBarMessage(iterationResult);
                    ConsolePrinter.PrintMetrics(iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, GetScore(bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                    if (iterationResult.Exception != null)
                    {
                        ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                    }
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

            public void Stop()
            {
                lock (this)
                {
                    this.isStopped = true;
                }
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
            private List<RunDetail<MulticlassClassificationMetrics>> completedIterations;
            private bool isStopped;

            public MulticlassClassificationHandler(MulticlassClassificationMetric optimizationMetric, List<RunDetail<MulticlassClassificationMetrics>> completedIterations, ProgressBar progressBar)
            {
                this.isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                this.optimizationMetric = optimizationMetric;
                this.completedIterations = completedIterations;
                this.progressBar = progressBar;
                GetScore = (RunDetail<MulticlassClassificationMetrics> result) => new MultiMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintMulticlassClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetail<MulticlassClassificationMetrics> iterationResult)
            {
                lock (this)
                {
                    if (this.isStopped)
                    {
                        return;
                    }

                    iterationIndex++;
                    completedIterations.Add(iterationResult);
                    UpdateBestResult(iterationResult);
                    if (progressBar != null)
                        progressBar.Message = GetProgressBarMessage(iterationResult);
                    ConsolePrinter.PrintMetrics(iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, GetScore(bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                    if (iterationResult.Exception != null)
                    {
                        ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                    }
                }
            }

            public void Stop()
            {
                lock (this)
                {
                    this.isStopped = true;
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