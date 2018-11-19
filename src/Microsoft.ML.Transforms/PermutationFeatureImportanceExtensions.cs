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
            PermutationFeatureImportance(
                this RegressionContext ctx,
                ITransformer model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                int topExamples = 0,
                int progressIterations = 10)
        {
            var pfi = new PermutationFeatureImportance<RegressionEvaluator.Result>(ctx.Environment);
            return pfi.GetImportanceMetricsMatrix(
                            model,
                            data,
                            idv => ctx.Evaluate(idv, label),
                            RegressionDelta,
                            features,
                            topExamples,
                            progressIterations);
        }

        private static RegressionEvaluator.Result RegressionDelta(
            RegressionEvaluator.Result a, RegressionEvaluator.Result b)
        {
            return new RegressionEvaluator.Result(
                l1: a.L1 - b.L1,
                l2: a.L2 - b.L2,
                rms: a.Rms - b.Rms,
                lossfn: a.LossFn - b.LossFn,
                rsquared: a.RSquared - b.RSquared);
        }
    }
}
