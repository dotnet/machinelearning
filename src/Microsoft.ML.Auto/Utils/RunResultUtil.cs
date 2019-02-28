// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Auto
{
    internal class RunResultUtil
    {
        public static RunResult<T> GetBestRunResult<T>(IEnumerable<RunResult<T>> results,
            IMetricsAgent<T> metricsAgent)
        {
            results = results.Where(r => r.Metrics != null);
            if (!results.Any()) { return null; }
            double maxScore = results.Select(r => metricsAgent.GetScore(r.Metrics)).Max();
            return results.First(r => metricsAgent.GetScore(r.Metrics) == maxScore);
        }
    }
}
