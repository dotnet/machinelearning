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
        public static ImmutableArray<RegressionEvaluator.Result>
            CalculateFeatureImportance(
                this RegressionContext ctx,
                ITransformer model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                int topExamples = 0)
        {
            var pfi = new PermutationFeatureImportance<RegressionEvaluator.Result>(CatalogUtils.GetEnvironment(ctx));
            return pfi.GetImportanceMetricsMatrix(
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
