// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;

namespace Microsoft.ML.Auto
{
    public enum OptimizingMetric
    {
        Auc,
        Accuracy,
        AccuracyMacro,
        L1,
        L2,
        F1,
        AuPrc,
        TopKAccuracy,
        Rms,
        LossFn,
        RSquared,
        LogLoss,
        LogLossReduction,
        Ndcg,
        Dcg,
        PositivePrecision,
        PositiveRecall,
        NegativePrecision,
        NegativeRecall,
        DrAtK,
        DrAtPFpr,
        DrAtNumPos,
        NumAnomalies,
        ThreshAtK,
        ThreshAtP,
        ThreshAtNumPos,
        Nmi,
        AvgMinScore,
        Dbi
    };

    internal sealed class OptimizingMetricInfo
    {
        public string Name { get; }
        public bool IsMaximizing { get; }

        private static OptimizingMetric[] _minimizingMetrics = new OptimizingMetric[]
        {
            OptimizingMetric.L1,
            OptimizingMetric.L2,
            OptimizingMetric.Rms,
            OptimizingMetric.LossFn,
            OptimizingMetric.ThreshAtK,
            OptimizingMetric.ThreshAtP,
            OptimizingMetric.ThreshAtNumPos,
            OptimizingMetric.AvgMinScore,
            OptimizingMetric.Dbi
        };

        public OptimizingMetricInfo(OptimizingMetric optimizingMetric)
        {
            Name = optimizingMetric.ToString();
            IsMaximizing = !_minimizingMetrics.Contains(optimizingMetric);
        }

        public override string ToString() => Name;
    }
}
