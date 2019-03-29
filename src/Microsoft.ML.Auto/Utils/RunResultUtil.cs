// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Auto
{
    internal class RunResultUtil
    {
        public static RunResult<T> GetBestRunResult<T>(IEnumerable<RunResult<T>> results,
            IMetricsAgent<T> metricsAgent)
        {
            results = results.Where(r => r.ValidationMetrics != null);
            if (!results.Any()) { return null; }
            double maxScore = results.Select(r => metricsAgent.GetScore(r.ValidationMetrics)).Max();
            return results.First(r => Math.Abs(metricsAgent.GetScore(r.ValidationMetrics) - maxScore) < 1E-20);
        }

        public static IEnumerable<RunResult<T>> GetTopNRunResults<T>(IEnumerable<RunResult<T>> results,
            IMetricsAgent<T> metricsAgent, int n)
        {
            results = results.Where(r => r.ValidationMetrics != null);
            if (!results.Any()) { return null; }

            var orderedResults = results.OrderByDescending(t => metricsAgent.GetScore(t.ValidationMetrics));

            return orderedResults.Take(n);
        }
    }
}
