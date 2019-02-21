// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class RegressionMetricsAgent : IMetricsAgent<RegressionMetrics>
    {
        private readonly RegressionMetric _optimizingMetric;

        public RegressionMetricsAgent(RegressionMetric optimizingMetric)
        {
            this._optimizingMetric = optimizingMetric;
        }

        public double GetScore(RegressionMetrics metrics)
        {
            switch(_optimizingMetric)
            {
                case RegressionMetric.L1:
                    return metrics.L1;
                case RegressionMetric.L2:
                    return metrics.L2;
                case RegressionMetric.Rms:
                    return metrics.Rms;
                case RegressionMetric.RSquared:
                    return metrics.RSquared;
            }

            // never expected to reach here
            throw new NotSupportedException($"{_optimizingMetric} is not a supported sweep metric");
        }
    }
}
