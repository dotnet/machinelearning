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
            private readonly bool _isMaximizing;
            private readonly Func<RunDetail<RegressionMetrics>, double> _getScore;
            private RunDetail<RegressionMetrics> _bestResult;
            private int _iterationIndex;
            private List<RunDetail<RegressionMetrics>> _completedIterations;
            private ProgressBar _progressBar;
            private string _optimizationMetric;
            private bool _isStopped;

            public RegressionHandler(RegressionMetric optimizationMetric, List<RunDetail<RegressionMetrics>> completedIterations, ShellProgressBar.ProgressBar progressBar)
            {
                _isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                _optimizationMetric = optimizationMetric.ToString();
                _completedIterations = completedIterations;
                _progressBar = progressBar;
                _getScore = (RunDetail<RegressionMetrics> result) => new RegressionMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintRegressionMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetail<RegressionMetrics> iterationResult)
            {
                lock (this)
                {
                    if (_isStopped)
                        return;

                    _iterationIndex++;
                    _completedIterations.Add(iterationResult);
                    UpdateBestResult(iterationResult);
                    if (_progressBar != null)
                        _progressBar.Message = $"Best quality({_optimizationMetric}): {_getScore(_bestResult):F4}, Best Algorithm: {_bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                    ConsolePrinter.PrintMetrics(_iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, _getScore(_bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
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
                    _isStopped = true;
                }
            }

            private void UpdateBestResult(RunDetail<RegressionMetrics> iterationResult)
            {
                if (MetricComparator(_getScore(iterationResult), _getScore(_bestResult), _isMaximizing) > 0)
                {
                    _bestResult = iterationResult;
                }
            }
        }

        internal class BinaryClassificationHandler : IProgress<RunDetail<BinaryClassificationMetrics>>
        {
            private readonly bool _isMaximizing;
            private readonly Func<RunDetail<BinaryClassificationMetrics>, double> _getScore;
            private RunDetail<BinaryClassificationMetrics> _bestResult;
            private int _iterationIndex;
            private ProgressBar _progressBar;
            private BinaryClassificationMetric _optimizationMetric;
            private List<RunDetail<BinaryClassificationMetrics>> _completedIterations;
            private bool _isStopped;

            public BinaryClassificationHandler(BinaryClassificationMetric optimizationMetric, List<RunDetail<BinaryClassificationMetrics>> completedIterations, ProgressBar progressBar)
            {
                _isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                _optimizationMetric = optimizationMetric;
                _completedIterations = completedIterations;
                _progressBar = progressBar;
                _getScore = (RunDetail<BinaryClassificationMetrics> result) => new BinaryMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintBinaryClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetail<BinaryClassificationMetrics> iterationResult)
            {
                lock (this)
                {
                    if (_isStopped)
                        return;
                    _iterationIndex++;
                    _completedIterations.Add(iterationResult);
                    UpdateBestResult(iterationResult);
                    if (_progressBar != null)
                        _progressBar.Message = GetProgressBarMessage(iterationResult);
                    ConsolePrinter.PrintMetrics(_iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, _getScore(_bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
                    if (iterationResult.Exception != null)
                    {
                        ConsolePrinter.PrintException(iterationResult.Exception, LogLevel.Trace);
                    }
                }
            }

            private string GetProgressBarMessage(RunDetail<BinaryClassificationMetrics> iterationResult)
            {
                if (_optimizationMetric == BinaryClassificationMetric.Accuracy)
                {
                    return $"Best Accuracy: {_getScore(_bestResult) * 100:F2}%, Best Algorithm: {_bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                }

                return $"Best {_optimizationMetric}: {_getScore(_bestResult):F4}, Best Algorithm: {_bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
            }

            public void Stop()
            {
                lock (this)
                {
                    _isStopped = true;
                }
            }

            private void UpdateBestResult(RunDetail<BinaryClassificationMetrics> iterationResult)
            {
                if (MetricComparator(_getScore(iterationResult), _getScore(_bestResult), _isMaximizing) > 0)
                {
                    _bestResult = iterationResult;
                }
            }
        }

        internal class MulticlassClassificationHandler : IProgress<RunDetail<MulticlassClassificationMetrics>>
        {
            private readonly bool _isMaximizing;
            private readonly Func<RunDetail<MulticlassClassificationMetrics>, double> _getScore;
            private RunDetail<MulticlassClassificationMetrics> _bestResult;
            private int _iterationIndex;
            private ProgressBar _progressBar;
            private MulticlassClassificationMetric _optimizationMetric;
            private List<RunDetail<MulticlassClassificationMetrics>> _completedIterations;
            private bool _isStopped;

            public MulticlassClassificationHandler(MulticlassClassificationMetric optimizationMetric, List<RunDetail<MulticlassClassificationMetrics>> completedIterations, ProgressBar progressBar)
            {
                _isMaximizing = new OptimizingMetricInfo(optimizationMetric).IsMaximizing;
                _optimizationMetric = optimizationMetric;
                _completedIterations = completedIterations;
                _progressBar = progressBar;
                _getScore = (RunDetail<MulticlassClassificationMetrics> result) => new MultiMetricsAgent(null, optimizationMetric).GetScore(result?.ValidationMetrics);
                ConsolePrinter.PrintMulticlassClassificationMetricsHeader(LogLevel.Trace);
            }

            public void Report(RunDetail<MulticlassClassificationMetrics> iterationResult)
            {
                lock (this)
                {
                    if (_isStopped)
                    {
                        return;
                    }

                    _iterationIndex++;
                    _completedIterations.Add(iterationResult);
                    UpdateBestResult(iterationResult);
                    if (_progressBar != null)
                        _progressBar.Message = GetProgressBarMessage(iterationResult);
                    ConsolePrinter.PrintMetrics(_iterationIndex, iterationResult?.TrainerName, iterationResult?.ValidationMetrics, _getScore(_bestResult), iterationResult?.RuntimeInSeconds, LogLevel.Trace);
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
                    _isStopped = true;
                }
            }

            private void UpdateBestResult(RunDetail<MulticlassClassificationMetrics> iterationResult)
            {
                if (MetricComparator(_getScore(iterationResult), _getScore(_bestResult), _isMaximizing) > 0)
                {
                    _bestResult = iterationResult;
                }
            }

            private string GetProgressBarMessage(RunDetail<MulticlassClassificationMetrics> iterationResult)
            {
                if (_optimizationMetric == MulticlassClassificationMetric.MicroAccuracy)
                {
                    return $"Best Accuracy: {_getScore(_bestResult) * 100:F2}%, Best Algorithm: {_bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
                }

                return $"Best {_optimizationMetric}: {_getScore(_bestResult):F4}, Best Algorithm: {_bestResult?.TrainerName}, Last Algorithm: {iterationResult?.TrainerName}";
            }
        }

    }
}