// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class BinaryMetricsAgent : IMetricsAgent<BinaryClassificationMetrics>
    {
        private readonly BinaryClassificationMetric _optimizingMetric;

        public BinaryMetricsAgent(BinaryClassificationMetric optimizingMetric)
        {
            this._optimizingMetric = optimizingMetric;
        }

        public double GetScore(BinaryClassificationMetrics metrics)
        {
            switch(_optimizingMetric)
            {
                case BinaryClassificationMetric.Accuracy:
                    return metrics.Accuracy;
                case BinaryClassificationMetric.Auc:
                    return metrics.Auc;
                case BinaryClassificationMetric.Auprc:
                    return metrics.Auprc;
                case BinaryClassificationMetric.F1Score:
                    return metrics.F1Score;
                case BinaryClassificationMetric.NegativePrecision:
                    return metrics.NegativePrecision;
                case BinaryClassificationMetric.NegativeRecall:
                    return metrics.NegativeRecall;
                case BinaryClassificationMetric.PositivePrecision:
                    return metrics.PositivePrecision;
                case BinaryClassificationMetric.PositiveRecall:
                    return metrics.PositiveRecall;
            }

            // never expected to reach here
            throw new NotSupportedException($"{_optimizingMetric} is not a supported sweep metric");
        }
    }
}
