// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class MetricsAgentsTests
    {
        [TestMethod]
        public void BinaryMetricsGetScoreTest()
        {
            var metrics = MetricsUtil.CreateBinaryClassificationMetrics(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8);
            Assert.AreEqual(0.1, GetScore(metrics, BinaryClassificationMetric.AreaUnderRocCurve));
            Assert.AreEqual(0.2, GetScore(metrics, BinaryClassificationMetric.Accuracy));
            Assert.AreEqual(0.3, GetScore(metrics, BinaryClassificationMetric.PositivePrecision));
            Assert.AreEqual(0.4, GetScore(metrics, BinaryClassificationMetric.PositiveRecall));
            Assert.AreEqual(0.5, GetScore(metrics, BinaryClassificationMetric.NegativePrecision));
            Assert.AreEqual(0.6, GetScore(metrics, BinaryClassificationMetric.NegativeRecall));
            Assert.AreEqual(0.7, GetScore(metrics, BinaryClassificationMetric.F1Score));
            Assert.AreEqual(0.8, GetScore(metrics, BinaryClassificationMetric.AreaUnderPrecisionRecallCurve));
        }

        [TestMethod]
        public void BinaryMetricsNonPerfectTest()
        {
            var metrics = MetricsUtil.CreateBinaryClassificationMetrics(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8);
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.Accuracy));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.AreaUnderRocCurve));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.AreaUnderPrecisionRecallCurve));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.F1Score));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.NegativePrecision));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.NegativeRecall));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.PositivePrecision));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.PositiveRecall));
        }

        [TestMethod]
        public void BinaryMetricsPerfectTest()
        {
            var metrics = MetricsUtil.CreateBinaryClassificationMetrics(1, 1, 1, 1, 1, 1, 1, 1);
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.Accuracy));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.AreaUnderRocCurve));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.AreaUnderPrecisionRecallCurve));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.F1Score));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.NegativePrecision));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.NegativeRecall));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.PositivePrecision));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.PositiveRecall));
        }

        [TestMethod]
        public void MulticlassMetricsGetScoreTest()
        {
            var metrics = MetricsUtil.CreateMulticlassClassificationMetrics(0.1, 0.2, 0.3, 0.4, 0, 0.5, new double[] {});
            Assert.AreEqual(0.1, GetScore(metrics, MulticlassClassificationMetric.MicroAccuracy));
            Assert.AreEqual(0.2, GetScore(metrics, MulticlassClassificationMetric.MacroAccuracy));
            Assert.AreEqual(0.3, GetScore(metrics, MulticlassClassificationMetric.LogLoss));
            Assert.AreEqual(0.4, GetScore(metrics, MulticlassClassificationMetric.LogLossReduction));
            Assert.AreEqual(0.5, GetScore(metrics, MulticlassClassificationMetric.TopKAccuracy));
        }

        [TestMethod]
        public void MulticlassMetricsNonPerfectTest()
        {
            var metrics = MetricsUtil.CreateMulticlassClassificationMetrics(0.1, 0.2, 0.3, 0.4, 0, 0.5, new double[] { });
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.MacroAccuracy));
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.MicroAccuracy));
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.LogLoss));
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.LogLossReduction));
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.TopKAccuracy));
        }

        [TestMethod]
        public void MulticlassMetricsPerfectTest()
        {
            var metrics = MetricsUtil.CreateMulticlassClassificationMetrics(1, 1, 0, 1, 0, 1, new double[] { });
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.MicroAccuracy));
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.MacroAccuracy));
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.LogLoss));
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.LogLossReduction));
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.TopKAccuracy));
        }

        [TestMethod]
        public void RegressionMetricsGetScoreTest()
        {
            var metrics = MetricsUtil.CreateRegressionMetrics(0.2, 0.3, 0.4, 0.5, 0.6);
            Assert.AreEqual(0.2, GetScore(metrics, RegressionMetric.MeanAbsoluteError));
            Assert.AreEqual(0.3, GetScore(metrics, RegressionMetric.MeanSquaredError));
            Assert.AreEqual(0.4, GetScore(metrics, RegressionMetric.RootMeanSquaredError));
            Assert.AreEqual(0.6, GetScore(metrics, RegressionMetric.RSquared));
        }

        [TestMethod]
        public void RegressionMetricsNonPerfectTest()
        {
            var metrics = MetricsUtil.CreateRegressionMetrics(0.2, 0.3, 0.4, 0.5, 0.6);
            Assert.AreEqual(false, IsPerfectModel(metrics, RegressionMetric.MeanAbsoluteError));
            Assert.AreEqual(false, IsPerfectModel(metrics, RegressionMetric.MeanSquaredError));
            Assert.AreEqual(false, IsPerfectModel(metrics, RegressionMetric.RootMeanSquaredError));
            Assert.AreEqual(false, IsPerfectModel(metrics, RegressionMetric.RSquared));
        }

        [TestMethod]
        public void RegressionMetricsPerfectTest()
        {
            var metrics = MetricsUtil.CreateRegressionMetrics(0, 0, 0, 0, 1);
            Assert.AreEqual(true, IsPerfectModel(metrics, RegressionMetric.MeanAbsoluteError));
            Assert.AreEqual(true, IsPerfectModel(metrics, RegressionMetric.MeanSquaredError));
            Assert.AreEqual(true, IsPerfectModel(metrics, RegressionMetric.RootMeanSquaredError));
            Assert.AreEqual(true, IsPerfectModel(metrics, RegressionMetric.RSquared));
        }

        [TestMethod]
        [ExpectedException(typeof(NotSupportedException))]
        public void ThrowNotSupportedMetricException()
        {
            throw MetricsAgentUtil.BuildMetricNotSupportedException(BinaryClassificationMetric.Accuracy);
        }        

        private static double GetScore(BinaryClassificationMetrics metrics, BinaryClassificationMetric metric)
        {
            return new BinaryMetricsAgent(null, metric).GetScore(metrics);
        }

        private static double GetScore(MulticlassClassificationMetrics metrics, MulticlassClassificationMetric metric)
        {
            return new MultiMetricsAgent(null, metric).GetScore(metrics);
        }

        private static double GetScore(RegressionMetrics metrics, RegressionMetric metric)
        {
            return new RegressionMetricsAgent(null, metric).GetScore(metrics);
        }

        private static bool IsPerfectModel(BinaryClassificationMetrics metrics, BinaryClassificationMetric metric)
        {
            var metricsAgent = new BinaryMetricsAgent(null, metric);
            return IsPerfectModel(metricsAgent, metrics);
        }

        private static bool IsPerfectModel(MulticlassClassificationMetrics metrics, MulticlassClassificationMetric metric)
        {
            var metricsAgent = new MultiMetricsAgent(null, metric);
            return IsPerfectModel(metricsAgent, metrics);
        }

        private static bool IsPerfectModel(RegressionMetrics metrics, RegressionMetric metric)
        {
            var metricsAgent = new RegressionMetricsAgent(null, metric);
            return IsPerfectModel(metricsAgent, metrics);
        }

        private static bool IsPerfectModel<TMetrics>(IMetricsAgent<TMetrics> metricsAgent, TMetrics metrics)
        {
            var score = metricsAgent.GetScore(metrics);
            return metricsAgent.IsModelPerfect(score);
        }
    }
}
