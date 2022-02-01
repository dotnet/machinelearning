// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Test
{
    internal static class MetricsUtil
    {
        public static BinaryClassificationMetrics CreateBinaryClassificationMetrics(
            double auc, double accuracy, double positivePrecision,
            double positiveRecall, double negativePrecision,
            double negativeRecall, double f1Score, double auprc)
        {
            return CreateInstance<BinaryClassificationMetrics>(auc, accuracy,
                positivePrecision, positiveRecall, negativePrecision,
                negativeRecall, f1Score, auprc);
        }

        public static MulticlassClassificationMetrics CreateMulticlassClassificationMetrics(
            double accuracyMicro, double accuracyMacro, double logLoss,
            double logLossReduction, int topK, double[] topKAccuracy,
            double[] perClassLogLoss)
        {
            return CreateInstance<MulticlassClassificationMetrics>(accuracyMicro,
                accuracyMacro, logLoss, logLossReduction, topK,
                topKAccuracy, perClassLogLoss);
        }

        public static RegressionMetrics CreateRegressionMetrics(double l1,
            double l2, double rms, double lossFn, double rSquared)
        {
            return CreateInstance<RegressionMetrics>(l1, l2,
                rms, lossFn, rSquared);
        }

        public static RankingMetrics CreateRankingMetrics(double[] dcg,
            double[] ndcg)
        {
            return CreateInstance<RankingMetrics>(dcg, ndcg);
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
    }
}
