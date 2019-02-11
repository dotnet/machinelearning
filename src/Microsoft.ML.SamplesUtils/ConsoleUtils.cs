using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.SamplesUtils
{
    public static class ConsoleUtils
    {
        public static void PrintBinaryClassificationMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.Auc:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:F2}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F2}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}");
        }

        public static void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"L1: {metrics.L1:F2}");
            Console.WriteLine($"L2: {metrics.L2:F2}");
            Console.WriteLine($"LossFunction: {metrics.LossFn:F2}");
            Console.WriteLine($"RMS: {metrics.Rms:F2}");
            Console.WriteLine($"RSquared: {metrics.RSquared:F2}");
        }
    }
}
