// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

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

            public RegressionHandler(RegressionMetric optimizationMetric)
            {
                isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                GetScore = (RunResult<RegressionMetrics> result) => new RegressionMetricsAgent(optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintRegressionMetricsHeader();
            }

            public void Report(RunResult<RegressionMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics, GetScore(bestResult), iterationResult.RuntimeInSeconds);
            }

            private void UpdateBestResult(RunResult<RegressionMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                    bestResult = iterationResult;
            }
        }

        internal class BinaryClassificationHandler : IProgress<RunResult<BinaryClassificationMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunResult<BinaryClassificationMetrics>, double> GetScore;
            private RunResult<BinaryClassificationMetrics> bestResult;
            private int iterationIndex;

            public BinaryClassificationHandler(BinaryClassificationMetric optimizationMetric)
            {
                isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                GetScore = (RunResult<BinaryClassificationMetrics> result) => new BinaryMetricsAgent(optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintBinaryClassificationMetricsHeader();
            }

            public void Report(RunResult<BinaryClassificationMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics, GetScore(bestResult), iterationResult.RuntimeInSeconds);
            }

            private void UpdateBestResult(RunResult<BinaryClassificationMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                    bestResult = iterationResult;
            }
        }

        internal class MulticlassClassificationHandler : IProgress<RunResult<MultiClassClassifierMetrics>>
        {
            private readonly bool isMaximizing;
            private readonly Func<RunResult<MultiClassClassifierMetrics>, double> GetScore;
            private RunResult<MultiClassClassifierMetrics> bestResult;
            private int iterationIndex;

            public MulticlassClassificationHandler(MulticlassClassificationMetric optimizationMetric)
            {
                isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                GetScore = (RunResult<MultiClassClassifierMetrics> result) => new MultiMetricsAgent(optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintMulticlassClassificationMetricsHeader();
            }

            public void Report(RunResult<MultiClassClassifierMetrics> iterationResult)
            {
                iterationIndex++;
                UpdateBestResult(iterationResult);
                ConsolePrinter.PrintMetrics(iterationIndex, iterationResult.TrainerName, iterationResult.ValidationMetrics, GetScore(bestResult), iterationResult.RuntimeInSeconds);
            }

            private void UpdateBestResult(RunResult<MultiClassClassifierMetrics> iterationResult)
            {
                if (MetricComparator(GetScore(iterationResult), GetScore(bestResult), isMaximizing) > 0)
                    bestResult = iterationResult;
            }
        }

    }
}
