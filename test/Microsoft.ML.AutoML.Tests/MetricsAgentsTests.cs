// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class MetricsAgentsTests : BaseTestClass
    {
        public MetricsAgentsTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void BinaryMetricsGetScoreTest()
        {
            var metrics = MetricsUtil.CreateBinaryClassificationMetrics(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8);
            Assert.Equal(0.1, GetScore(metrics, BinaryClassificationMetric.AreaUnderRocCurve));
            Assert.Equal(0.2, GetScore(metrics, BinaryClassificationMetric.Accuracy));
            Assert.Equal(0.3, GetScore(metrics, BinaryClassificationMetric.PositivePrecision));
            Assert.Equal(0.4, GetScore(metrics, BinaryClassificationMetric.PositiveRecall));
            Assert.Equal(0.5, GetScore(metrics, BinaryClassificationMetric.NegativePrecision));
            Assert.Equal(0.6, GetScore(metrics, BinaryClassificationMetric.NegativeRecall));
            Assert.Equal(0.7, GetScore(metrics, BinaryClassificationMetric.F1Score));
            Assert.Equal(0.8, GetScore(metrics, BinaryClassificationMetric.AreaUnderPrecisionRecallCurve));
        }

        [Fact]
        public void BinaryMetricsNonPerfectTest()
        {
            var metrics = MetricsUtil.CreateBinaryClassificationMetrics(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8);
            Assert.False(IsPerfectModel(metrics, BinaryClassificationMetric.Accuracy));
            Assert.False(IsPerfectModel(metrics, BinaryClassificationMetric.AreaUnderRocCurve));
            Assert.False(IsPerfectModel(metrics, BinaryClassificationMetric.AreaUnderPrecisionRecallCurve));
            Assert.False(IsPerfectModel(metrics, BinaryClassificationMetric.F1Score));
            Assert.False(IsPerfectModel(metrics, BinaryClassificationMetric.NegativePrecision));
            Assert.False(IsPerfectModel(metrics, BinaryClassificationMetric.NegativeRecall));
            Assert.False(IsPerfectModel(metrics, BinaryClassificationMetric.PositivePrecision));
            Assert.False(IsPerfectModel(metrics, BinaryClassificationMetric.PositiveRecall));
        }

        [Fact]
        public void BinaryMetricsPerfectTest()
        {
            var metrics = MetricsUtil.CreateBinaryClassificationMetrics(1, 1, 1, 1, 1, 1, 1, 1);
            Assert.True(IsPerfectModel(metrics, BinaryClassificationMetric.Accuracy));
            Assert.True(IsPerfectModel(metrics, BinaryClassificationMetric.AreaUnderRocCurve));
            Assert.True(IsPerfectModel(metrics, BinaryClassificationMetric.AreaUnderPrecisionRecallCurve));
            Assert.True(IsPerfectModel(metrics, BinaryClassificationMetric.F1Score));
            Assert.True(IsPerfectModel(metrics, BinaryClassificationMetric.NegativePrecision));
            Assert.True(IsPerfectModel(metrics, BinaryClassificationMetric.NegativeRecall));
            Assert.True(IsPerfectModel(metrics, BinaryClassificationMetric.PositivePrecision));
            Assert.True(IsPerfectModel(metrics, BinaryClassificationMetric.PositiveRecall));
        }

        [Fact]
        public void MulticlassMetricsGetScoreTest()
        {
            var metrics = MetricsUtil.CreateMulticlassClassificationMetrics(0.1, 0.2, 0.3, 0.4, 0, new double[] { 0.5 }, new double[] { });
            Assert.Equal(0.1, GetScore(metrics, MulticlassClassificationMetric.MicroAccuracy));
            Assert.Equal(0.2, GetScore(metrics, MulticlassClassificationMetric.MacroAccuracy));
            Assert.Equal(0.3, GetScore(metrics, MulticlassClassificationMetric.LogLoss));
            Assert.Equal(0.4, GetScore(metrics, MulticlassClassificationMetric.LogLossReduction));
            Assert.Equal(0.5, GetScore(metrics, MulticlassClassificationMetric.TopKAccuracy));
        }

        [Fact]
        public void MulticlassMetricsNonPerfectTest()
        {
            var metrics = MetricsUtil.CreateMulticlassClassificationMetrics(0.1, 0.2, 0.3, 0.4, 0, new double[] { 0.5 }, new double[] { });
            Assert.False(IsPerfectModel(metrics, MulticlassClassificationMetric.MacroAccuracy));
            Assert.False(IsPerfectModel(metrics, MulticlassClassificationMetric.MicroAccuracy));
            Assert.False(IsPerfectModel(metrics, MulticlassClassificationMetric.LogLoss));
            Assert.False(IsPerfectModel(metrics, MulticlassClassificationMetric.LogLossReduction));
            Assert.False(IsPerfectModel(metrics, MulticlassClassificationMetric.TopKAccuracy));
        }

        [Fact]
        public void MulticlassMetricsPerfectTest()
        {
            var metrics = MetricsUtil.CreateMulticlassClassificationMetrics(1, 1, 0, 1, 0, new double[] { 1 }, new double[] { });
            Assert.True(IsPerfectModel(metrics, MulticlassClassificationMetric.MicroAccuracy));
            Assert.True(IsPerfectModel(metrics, MulticlassClassificationMetric.MacroAccuracy));
            Assert.True(IsPerfectModel(metrics, MulticlassClassificationMetric.LogLoss));
            Assert.True(IsPerfectModel(metrics, MulticlassClassificationMetric.LogLossReduction));
            Assert.True(IsPerfectModel(metrics, MulticlassClassificationMetric.TopKAccuracy));
        }

        [Fact]
        public void RegressionMetricsGetScoreTest()
        {
            var metrics = MetricsUtil.CreateRegressionMetrics(0.2, 0.3, 0.4, 0.5, 0.6);
            Assert.Equal(0.2, GetScore(metrics, RegressionMetric.MeanAbsoluteError));
            Assert.Equal(0.3, GetScore(metrics, RegressionMetric.MeanSquaredError));
            Assert.Equal(0.4, GetScore(metrics, RegressionMetric.RootMeanSquaredError));
            Assert.Equal(0.6, GetScore(metrics, RegressionMetric.RSquared));
        }

        [Fact]
        public void RegressionMetricsNonPerfectTest()
        {
            var metrics = MetricsUtil.CreateRegressionMetrics(0.2, 0.3, 0.4, 0.5, 0.6);
            Assert.False(IsPerfectModel(metrics, RegressionMetric.MeanAbsoluteError));
            Assert.False(IsPerfectModel(metrics, RegressionMetric.MeanSquaredError));
            Assert.False(IsPerfectModel(metrics, RegressionMetric.RootMeanSquaredError));
            Assert.False(IsPerfectModel(metrics, RegressionMetric.RSquared));
        }

        [Fact]
        public void RegressionMetricsPerfectTest()
        {
            var metrics = MetricsUtil.CreateRegressionMetrics(0, 0, 0, 0, 1);
            Assert.True(IsPerfectModel(metrics, RegressionMetric.MeanAbsoluteError));
            Assert.True(IsPerfectModel(metrics, RegressionMetric.MeanSquaredError));
            Assert.True(IsPerfectModel(metrics, RegressionMetric.RootMeanSquaredError));
            Assert.True(IsPerfectModel(metrics, RegressionMetric.RSquared));
        }

        [Fact]
        public void RankingMetricsGetScoreTest()
        {
            double[] ndcg = { 0.2, 0.3, 0.4 };
            double[] dcg = { 0.2, 0.3, 0.4 };
            var metrics = MetricsUtil.CreateRankingMetrics(dcg, ndcg);
            Assert.Equal(0.4, GetScore(metrics, RankingMetric.Dcg, 3));
            Assert.Equal(0.4, GetScore(metrics, RankingMetric.Ndcg, 3));

            double[] largeNdcg = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95 };
            double[] largeDcg = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95 };
            metrics = MetricsUtil.CreateRankingMetrics(largeDcg, largeNdcg);
            Assert.Equal(0.3, GetScore(metrics, RankingMetric.Dcg, 3));
            Assert.Equal(0.3, GetScore(metrics, RankingMetric.Ndcg, 3));
        }

        [Fact]
        public void RankingMetricsNonPerfectTest()
        {
            double[] ndcg = { 0.2, 0.3, 0.4 };
            double[] dcg = { 0.2, 0.3, 0.4 };
            var metrics = MetricsUtil.CreateRankingMetrics(dcg, ndcg);
            Assert.False(IsPerfectModel(metrics, RankingMetric.Dcg, 3));
            Assert.False(IsPerfectModel(metrics, RankingMetric.Ndcg, 3));
        }

        [Fact]
        public void RankingMetricsPerfectTest()
        {
            double[] ndcg = { 0.2, 0.3, 1 };
            double[] dcg = { 0.2, 0.3, 1 };
            var metrics = MetricsUtil.CreateRankingMetrics(dcg, ndcg);
            Assert.False(IsPerfectModel(metrics, RankingMetric.Dcg, 3)); //REVIEW: No true Perfect model
            Assert.True(IsPerfectModel(metrics, RankingMetric.Ndcg, 3));
        }

        [Fact]
        public void ThrowNotSupportedMetricException()
        {
            var ex = MetricsAgentUtil.BuildMetricNotSupportedException(BinaryClassificationMetric.Accuracy);
            Assert.Equal(typeof(NotSupportedException), ex.GetType());
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

        private static double GetScore(RankingMetrics metrics, RankingMetric metric, uint dcgTruncationLevel)
        {
            return new RankingMetricsAgent(null, metric, dcgTruncationLevel).GetScore(metrics);
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

        private static bool IsPerfectModel(RankingMetrics metrics, RankingMetric metric, uint dcgTruncationLevel)
        {
            var metricsAgent = new RankingMetricsAgent(null, metric, dcgTruncationLevel);
            return IsPerfectModel(metricsAgent, metrics);
        }

        private static bool IsPerfectModel<TMetrics>(IMetricsAgent<TMetrics> metricsAgent, TMetrics metrics)
        {
            var score = metricsAgent.GetScore(metrics);
            return metricsAgent.IsModelPerfect(score);
        }
    }
}
