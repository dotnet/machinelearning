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
            ITransformer model, IDataView data,
            string label = DefaultColumnNames.Label, string features = DefaultColumnNames.Features,
            int topExamples = 0, int progressIterations = 10)
        {
            var pfi = new PermutationFeatureImportanceRegression(ctx.Environment);
            return pfi.GetImportanceMetricsMatrix(model, data, label, features);
        }
    }
}
