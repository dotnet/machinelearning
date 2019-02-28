// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class RunResultTests
    {
        [TestMethod]
        public void FindBestResultWithSomeNullMetrics()
        {
            var metrics1 = MetricsUtil.CreateRegressionMetrics(0.2, 0.2, 0.2, 0.2, 0.2);
            var metrics2 = MetricsUtil.CreateRegressionMetrics(0.3, 0.3, 0.3, 0.3, 0.3);
            var metrics3 = MetricsUtil.CreateRegressionMetrics(0.1, 0.1, 0.1, 0.1, 0.1);

            var runResults = new List<RunResult<RegressionMetrics>>()
            {
                new RunResult<RegressionMetrics>(null, null, null, null, 0, 0),
                new RunResult<RegressionMetrics>(null, metrics1, null, null, 0, 0),
                new RunResult<RegressionMetrics>(null, metrics2, null, null, 0, 0),
                new RunResult<RegressionMetrics>(null, metrics3, null, null, 0, 0),
            };

            var metricsAgent = new RegressionMetricsAgent(RegressionMetric.RSquared);
            var bestResult = RunResultUtil.GetBestRunResult(runResults, metricsAgent);
            Assert.AreEqual(0.3, bestResult.Metrics.RSquared);
        }

        [TestMethod]
        public void FindBestResultWithAllNullMetrics()
        {
            var runResults = new List<RunResult<RegressionMetrics>>()
            {
                new RunResult<RegressionMetrics>(null, null, null, null, 0, 0),
            };

            var metricsAgent = new RegressionMetricsAgent(RegressionMetric.RSquared);
            var bestResult = RunResultUtil.GetBestRunResult(runResults, metricsAgent);
            Assert.AreEqual(null, bestResult);
        }
    }
}
