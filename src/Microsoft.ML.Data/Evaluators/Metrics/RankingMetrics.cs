// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;

namespace Microsoft.ML.Data
{
    public sealed class RankingMetrics
    {
        /// <summary>
        /// Array of normalized discounted cumulative gains where i-th element represent NDCG@i.
        /// <image src="https://github.com/dotnet/machinelearning/tree/master/docs/images/NDCG.png"></image>
        /// </summary>
        public double[] Ndcg { get; }

        /// <summary>
        ///Array of discounted cumulative gains where i-th element represent DCG@i.
        /// <a href="https://en.wikipedia.org/wiki/Discounted_cumulative_gain">Discounted Cumulative gain</a>
        /// is the sum of the gains, for all the instances i, normalized by the natural logarithm of the instance + 1.
        /// Note that unline the Wikipedia article, ML.Net uses the natural logarithm.
        /// <image src="https://github.com/dotnet/machinelearning/tree/master/docs/images/DCG.png"></image>
        /// </summary>
        public double[] Dcg { get; }

        private static T Fetch<T>(IExceptionContext ectx, DataViewRow row, string name)
        {
            if (!row.Schema.TryGetColumnIndex(name, out int col))
                throw ectx.Except($"Could not find column '{name}'");
            T val = default;
            row.GetGetter<T>(col)(ref val);
            return val;
        }

        internal RankingMetrics(IExceptionContext ectx, DataViewRow overallResult)
        {
            VBuffer<double> Fetch(string name) => Fetch<VBuffer<double>>(ectx, overallResult, name);

            Dcg = Fetch(RankingEvaluator.Dcg).GetValues().ToArray();
            Ndcg = Fetch(RankingEvaluator.Ndcg).GetValues().ToArray();
        }

        internal RankingMetrics(double[] dcg, double[] ndcg)
        {
            Dcg = new double[dcg.Length];
            dcg.CopyTo(Dcg, 0);
            Ndcg = new double[ndcg.Length];
            ndcg.CopyTo(Ndcg, 0);
        }
    }
}