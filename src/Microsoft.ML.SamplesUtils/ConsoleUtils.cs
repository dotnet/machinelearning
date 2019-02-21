using System;
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
            Console.WriteLine($"AUC: {metrics.Auc:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:F2}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F2}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}");
        }

        /// <summary>
        /// Pretty-print RegressionMetrics objects.
        /// </summary>
        /// <param name="metrics">Regression metrics.</param>
        public static void PrintMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"L1: {metrics.L1:F2}");
            Console.WriteLine($"L2: {metrics.L2:F2}");
            Console.WriteLine($"LossFunction: {metrics.LossFn:F2}");
            Console.WriteLine($"RMS: {metrics.Rms:F2}");
            Console.WriteLine($"RSquared: {metrics.RSquared:F2}");
        }
    }
}
