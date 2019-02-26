// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class MetricsAgentsTests
    {
        [TestMethod]
        public void BinaryMetricsGetScoreTest()
        {
            var metrics = CreateInstance<BinaryClassificationMetrics>(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8);
            Assert.AreEqual(0.1, GetScore(metrics, BinaryClassificationMetric.Auc));
            Assert.AreEqual(0.2, GetScore(metrics, BinaryClassificationMetric.Accuracy));
            Assert.AreEqual(0.3, GetScore(metrics, BinaryClassificationMetric.PositivePrecision));
            Assert.AreEqual(0.4, GetScore(metrics, BinaryClassificationMetric.PositiveRecall));
            Assert.AreEqual(0.5, GetScore(metrics, BinaryClassificationMetric.NegativePrecision));
            Assert.AreEqual(0.6, GetScore(metrics, BinaryClassificationMetric.NegativeRecall));
            Assert.AreEqual(0.7, GetScore(metrics, BinaryClassificationMetric.F1Score));
            Assert.AreEqual(0.8, GetScore(metrics, BinaryClassificationMetric.Auprc));
        }

        [TestMethod]
        public void BinaryMetricsNonPerfectTest()
        {
            var metrics = CreateInstance<BinaryClassificationMetrics>(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8);
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.Accuracy));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.Auc));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.Auprc));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.F1Score));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.NegativePrecision));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.NegativeRecall));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.PositivePrecision));
            Assert.AreEqual(false, IsPerfectModel(metrics, BinaryClassificationMetric.PositiveRecall));
        }

        [TestMethod]
        public void BinaryMetricsPerfectTest()
        {
            var metrics = CreateInstance<BinaryClassificationMetrics>(1, 1, 1, 1, 1, 1, 1, 1);
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.Accuracy));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.Auc));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.Auprc));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.F1Score));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.NegativePrecision));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.NegativeRecall));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.PositivePrecision));
            Assert.AreEqual(true, IsPerfectModel(metrics, BinaryClassificationMetric.PositiveRecall));
        }

        [TestMethod]
        public void MulticlassMetricsGetScoreTest()
        {
            var metrics = CreateInstance<MultiClassClassifierMetrics>(0.1, 0.2, 0.3, 0.4, 0, 0.5, new double[] {});
            Assert.AreEqual(0.1, GetScore(metrics, MulticlassClassificationMetric.AccuracyMicro));
            Assert.AreEqual(0.2, GetScore(metrics, MulticlassClassificationMetric.AccuracyMacro));
            Assert.AreEqual(0.3, GetScore(metrics, MulticlassClassificationMetric.LogLoss));
            Assert.AreEqual(0.4, GetScore(metrics, MulticlassClassificationMetric.LogLossReduction));
            Assert.AreEqual(0.5, GetScore(metrics, MulticlassClassificationMetric.TopKAccuracy));
        }

        [TestMethod]
        public void MulticlassMetricsNonPerfectTest()
        {
            var metrics = CreateInstance<MultiClassClassifierMetrics>(0.1, 0.2, 0.3, 0.4, 0, 0.5, new double[] { });
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.AccuracyMacro));
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.AccuracyMicro));
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.LogLoss));
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.LogLossReduction));
            Assert.AreEqual(false, IsPerfectModel(metrics, MulticlassClassificationMetric.TopKAccuracy));
        }

        [TestMethod]
        public void MulticlassMetricsPerfectTest()
        {
            var metrics = CreateInstance<MultiClassClassifierMetrics>(1, 1, 0, 1, 0, 1, new double[] { });
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.AccuracyMicro));
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.AccuracyMacro));
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.LogLoss));
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.LogLossReduction));
            Assert.AreEqual(true, IsPerfectModel(metrics, MulticlassClassificationMetric.TopKAccuracy));
        }

        [TestMethod]
        public void RegressionMetricsGetScoreTest()
        {
            var metrics = CreateInstance<RegressionMetrics>(0.2, 0.3, 0.4, 0.5, 0.6);
            Assert.AreEqual(0.2, GetScore(metrics, RegressionMetric.L1));
            Assert.AreEqual(0.3, GetScore(metrics, RegressionMetric.L2));
            Assert.AreEqual(0.4, GetScore(metrics, RegressionMetric.Rms));
            Assert.AreEqual(0.6, GetScore(metrics, RegressionMetric.RSquared));
        }

        [TestMethod]
        public void RegressionMetricsNonPerfectTest()
        {
            var metrics = CreateInstance<RegressionMetrics>(0.2, 0.3, 0.4, 0.5, 0.6);
            Assert.AreEqual(false, IsPerfectModel(metrics, RegressionMetric.L1));
            Assert.AreEqual(false, IsPerfectModel(metrics, RegressionMetric.L2));
            Assert.AreEqual(false, IsPerfectModel(metrics, RegressionMetric.Rms));
            Assert.AreEqual(false, IsPerfectModel(metrics, RegressionMetric.RSquared));
        }

        [TestMethod]
        public void RegressionMetricsPerfectTest()
        {
            var metrics = CreateInstance<RegressionMetrics>(0, 0, 0, 0, 1);
            Assert.AreEqual(true, IsPerfectModel(metrics, RegressionMetric.L1));
            Assert.AreEqual(true, IsPerfectModel(metrics, RegressionMetric.L2));
            Assert.AreEqual(true, IsPerfectModel(metrics, RegressionMetric.Rms));
            Assert.AreEqual(true, IsPerfectModel(metrics, RegressionMetric.RSquared));
        }

        [TestMethod]
        [ExpectedException(typeof(NotSupportedException))]
        public void ThrowNotSupportedMetricException()
        {
            throw MetricsAgentUtil.BuildMetricNotSupportedException(BinaryClassificationMetric.Accuracy);
        }

        private static T CreateInstance<T>(params object[] args)
        {
            var type = typeof(T);
            var instance = type.Assembly.CreateInstance(
                type.FullName, false,
                BindingFlags.Instance | BindingFlags.NonPublic,
                null, args, null, null);
            return (T)instance;
        }

        private static double GetScore(BinaryClassificationMetrics metrics, BinaryClassificationMetric metric)
        {
            return new BinaryMetricsAgent(metric).GetScore(metrics);
        }

        private static double GetScore(MultiClassClassifierMetrics metrics, MulticlassClassificationMetric metric)
        {
            return new MultiMetricsAgent(metric).GetScore(metrics);
        }

        private static double GetScore(RegressionMetrics metrics, RegressionMetric metric)
        {
            return new RegressionMetricsAgent(metric).GetScore(metrics);
        }

        private static bool IsPerfectModel(BinaryClassificationMetrics metrics, BinaryClassificationMetric metric)
        {
            return new BinaryMetricsAgent(metric).IsModelPerfect(metrics);
        }

        private static bool IsPerfectModel(MultiClassClassifierMetrics metrics, MulticlassClassificationMetric metric)
        {
            return new MultiMetricsAgent(metric).IsModelPerfect(metrics);
        }

        private static bool IsPerfectModel(RegressionMetrics metrics, RegressionMetric metric)
        {
            return new RegressionMetricsAgent(metric).IsModelPerfect(metrics);
        }
    }
}
