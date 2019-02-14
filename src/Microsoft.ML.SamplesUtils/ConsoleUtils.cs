using System;
using System.Collections.Generic;
using System.Text;
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
    }
}
