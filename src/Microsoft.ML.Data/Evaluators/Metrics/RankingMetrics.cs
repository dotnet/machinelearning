// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Evaluation results for rankers.
    /// </summary>
    public sealed class RankingMetrics
    {
        /// <summary>
        /// Array of normalized discounted cumulative gains where i-th element represent NDCG@i.
        /// <image src="https://github.com/dotnet/machinelearning/tree/master/docs/images/NDCG.png"></image>
        /// </summary>
        public IReadOnlyList<double> NormalizedDiscountedCumulativeGains { get; }

        /// <summary>
        /// Array of discounted cumulative gains where i-th element represent DCG@i.
        /// Discounted Cumulative gain is the sum of the gains, for all the instances i,
        /// normalized by the natural logarithm of the instance + 1.
        /// Note that unline the Wikipedia article, ML.Net uses the natural logarithm.
        /// <image src="https://github.com/dotnet/machinelearning/tree/master/docs/images/DCG.png"></image>
        /// </summary>
        /// <remarks><a href="https://en.wikipedia.org/wiki/Discounted_cumulative_gain">Discounted Cumulative gain.</a></remarks>
        public IReadOnlyList<double> DiscountedCumulativeGains { get; }

        private static T Fetch<T>(IExceptionContext ectx, DataViewRow row, string name)
        {
            var column = row.Schema.GetColumnOrNull(name);
            if (!column.HasValue)
                throw ectx.Except($"Could not find column '{name}'");
            T val = default;
            row.GetGetter<T>(column.Value)(ref val);
            return val;
        }

        internal RankingMetrics(IExceptionContext ectx, DataViewRow overallResult)
        {
            VBuffer<double> Fetch(string name) => Fetch<VBuffer<double>>(ectx, overallResult, name);

            DiscountedCumulativeGains = Fetch(RankingEvaluator.Dcg).DenseValues().ToImmutableArray();
            NormalizedDiscountedCumulativeGains = Fetch(RankingEvaluator.Ndcg).DenseValues().ToImmutableArray();
        }

        internal RankingMetrics(double[] dcg, double[] ndcg)
        {
            DiscountedCumulativeGains = dcg.ToImmutableArray();
            NormalizedDiscountedCumulativeGains = ndcg.ToImmutableArray();
        }
    }
}