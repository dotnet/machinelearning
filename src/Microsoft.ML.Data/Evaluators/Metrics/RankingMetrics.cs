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
        /// <format type="text/markdown"><![CDATA[
        /// List of normalized discounted cumulative gains (NDCG), where the N-th element represents NDCG@N.
        /// Search resuls vary in length depending on query, so different rankers cannot be consistently compared
        /// using DCG alone unless the DCG is normalized. This is done by calculating the maximum DCG (also known
        /// as Ideal DCG), which is the DCG for the ideal ordering of search results sorted by their relative relevance.
        ///
        /// $NDCG@N = \frac{DCG@N}{MaxDCG@N}$
        /// ]]>
        /// </format>
        /// </summary>
        /// <remarks>
        /// <a href="https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG">Normalized Discounted Cumulative Gain</a>
        /// </remarks>
        public IReadOnlyList<double> NormalizedDiscountedCumulativeGains { get; }

        /// <summary>
        /// <format type="text/markdown"><![CDATA[
        /// List of discounted cumulative gains (DCG), where the N-th element represents DCG@N.
        /// Discounted Cumulative Gain is the sum of the relevance gains up to the N-th position for all the instances i,
        /// normalized by the natural logarithm of the instance + 1. DCG is an increasing metric,
        /// with a higher value indicating a better model.
        /// Note that unlike the Wikipedia article, ML.NET uses the natural logarithm.
        ///
        /// $DCG@N = \sum_{i = 1}^N \frac{g_i}{ln(i + 1)}$, where $g_i$ is the relevance gain at the i-th position.
        /// ]]>
        /// </format>
        /// </summary>
        /// <remarks>
        /// <a href="https://en.wikipedia.org/wiki/Discounted_cumulative_gain">Discounted Cumulative Gain</a>
        /// </remarks>
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
