// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class MultiMetricsAgent : IMetricsAgent<MultiClassClassifierMetrics>
    {
        private readonly MulticlassClassificationMetric _optimizingMetric;

        public MultiMetricsAgent(MulticlassClassificationMetric optimizingMetric)
        {
            this._optimizingMetric = optimizingMetric;
        }

        public double GetScore(MultiClassClassifierMetrics metrics)
        {
            switch (_optimizingMetric)
            {
                case MulticlassClassificationMetric.AccuracyMacro:
                    return metrics.AccuracyMacro;
                case MulticlassClassificationMetric.AccuracyMicro:
                    return metrics.AccuracyMicro;
                case MulticlassClassificationMetric.LogLoss:
                    return metrics.LogLoss;
                case MulticlassClassificationMetric.LogLossReduction:
                    return metrics.LogLossReduction;
                case MulticlassClassificationMetric.TopKAccuracy:
                    return metrics.TopKAccuracy;
            }

            // never expected to reach here
            throw new NotSupportedException($"{_optimizingMetric} is not a supported sweep metric");
        }
    }
}
