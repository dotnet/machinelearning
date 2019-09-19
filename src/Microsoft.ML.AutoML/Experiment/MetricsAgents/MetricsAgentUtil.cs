// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.AutoML
{
    internal static class MetricsAgentUtil
    {
        public static NotSupportedException BuildMetricNotSupportedException<T>(T optimizingMetric)
        {
            return new NotSupportedException($"{optimizingMetric} is not a supported sweep metric");
        }
    }
}
