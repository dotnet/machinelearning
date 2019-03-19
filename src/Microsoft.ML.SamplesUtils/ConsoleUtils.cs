using System;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.SamplesUtils
{
    /// <summary>
    /// Utilities for creating console outputs in samples' code.
    /// </summary>
    public static class ConsoleUtils
    {
        /// <summary>
        /// Pretty-print BinaryClassificationMetrics objects.
        /// </summary>
        /// <param name="metrics">Binary classification metrics.</param>
        public static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:F2}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F2}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}");
        }

        /// <summary>
        /// Pretty-print CalibratedBinaryClassificationMetrics objects.
        /// </summary>
        /// <param name="metrics"><see cref="CalibratedBinaryClassificationMetrics"/> object.</param>
        public static void PrintMetrics(CalibratedBinaryClassificationMetrics metrics)
        {
            PrintMetrics(metrics as BinaryClassificationMetrics);
            Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");
            Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction:F2}");
            Console.WriteLine($"Entropy: {metrics.Entropy:F2}");
        }

        /// <summary>
        /// Pretty-print MulticlassClassificationMetrics objects.
        /// </summary>
        /// <param name="metrics"><see cref="MulticlassClassificationMetrics"/> object.</param>
        public static void PrintMetrics(MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:F2}");
            Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:F2}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");
            Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction:F2}");
        }

        /// <summary>
        /// Pretty-print RegressionMetrics objects.
        /// </summary>
        /// <param name="metrics">Regression metrics.</param>
        public static void PrintMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError:F2}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError:F2}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:F2}");
            Console.WriteLine($"RSquared: {metrics.RSquared:F2}");
        }

        /// <summary>
        /// Pretty-print RankerMetrics objects.
        /// </summary>
        /// <param name="metrics">Ranker metrics.</param>
        public static void PrintMetrics(RankingMetrics metrics)
        {
            Console.WriteLine($"DCG: {string.Join(", ", metrics.DiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F2}").ToArray())}");
            Console.WriteLine($"NDCG: {string.Join(", ", metrics.NormalizedDiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F2}").ToArray())}");
        }
    }
}
