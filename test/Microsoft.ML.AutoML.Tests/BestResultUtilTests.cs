// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class BestResultUtilTests
    {
        [TestMethod]
        public void FindBestResultWithSomeNullMetrics()
        {
            var metrics1 = MetricsUtil.CreateRegressionMetrics(0.2, 0.2, 0.2, 0.2, 0.2);
            var metrics2 = MetricsUtil.CreateRegressionMetrics(0.3, 0.3, 0.3, 0.3, 0.3);
            var metrics3 = MetricsUtil.CreateRegressionMetrics(0.1, 0.1, 0.1, 0.1, 0.1);

            var runResults = new List<RunDetail<RegressionMetrics>>()
            {
                new RunDetail<RegressionMetrics>(null, null, null, null, null, null),
                new RunDetail<RegressionMetrics>(null, null, null, null, metrics1, null),
                new RunDetail<RegressionMetrics>(null, null, null, null, metrics2, null),
                new RunDetail<RegressionMetrics>(null, null, null, null, metrics3, null),
            };

            var metricsAgent = new RegressionMetricsAgent(null, RegressionMetric.RSquared);
            var bestResult = BestResultUtil.GetBestRun(runResults, metricsAgent, true);
            Assert.AreEqual(0.3, bestResult.ValidationMetrics.RSquared);
        }

        [TestMethod]
        public void FindBestResultWithAllNullMetrics()
        {
            var runResults = new List<RunDetail<RegressionMetrics>>()
            {
                new RunDetail<RegressionMetrics>(null, null, null, null, null, null),
            };

            var metricsAgent = new RegressionMetricsAgent(null, RegressionMetric.RSquared);
            var bestResult = BestResultUtil.GetBestRun(runResults, metricsAgent, true);
            Assert.AreEqual(null, bestResult);
        }

        [TestMethod]
        public void GetIndexOfBestScoreMaximizingUtil()
        {
            var scores = new double[] { 0, 2, 5, 100, -100, -70 };
            var indexOfMaxScore = BestResultUtil.GetIndexOfBestScore(scores, true);
            Assert.AreEqual(3, indexOfMaxScore);
        }

        [TestMethod]
        public void GetIndexOfBestScoreMinimizingUtil()
        {
            var scores = new double[] { 0, 2, 5, 100, -100, -70 };
            var indexOfMaxScore = BestResultUtil.GetIndexOfBestScore(scores, false);
            Assert.AreEqual(4, indexOfMaxScore);
        }
    }
}
