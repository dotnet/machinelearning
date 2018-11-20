// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Text;

namespace Microsoft.ML
{
    public static class PermutationFeatureImportanceExtensions
    {
        /// <summary>
        /// Permutation Feature Importance is a technique that calculates how much each feature 'matters' to the predictions.
        /// Namely, how much the model's predictions will change if we randomly permute the values of one feature across the evaluation set.
        /// If the quality doesn't change much, this feature is not very important. If the quality drops drastically, this was a really important feature.
        /// </summary>
        /// <param name="ctx">The regression context.</param>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="label">Label column name.</param>
        /// <param name="features">Feature column names.</param>
        /// <param name="topExamples">Number of top (highest positive) contributions to output.</param>
        /// <returns></returns>
        public static ImmutableArray<RegressionEvaluator.Result>
            CalculateFeatureImportance(
                this RegressionContext ctx,
                ITransformer model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                int topExamples = 0)
        {
            return PermutationFeatureImportance<RegressionEvaluator.Result>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(ctx),
                            model,
                            data,
                            idv => ctx.Evaluate(idv, label),
                            RegressionDelta,
                            features,
                            topExamples);
        }

        private static RegressionEvaluator.Result RegressionDelta(
            RegressionEvaluator.Result a, RegressionEvaluator.Result b)
        {
            return new RegressionEvaluator.Result(
                l1: a.L1 - b.L1,
                l2: a.L2 - b.L2,
                rms: a.Rms - b.Rms,
                lossFunction: a.LossFn - b.LossFn,
                rSquared: a.RSquared - b.RSquared);
        }
    }
}
