// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System.Collections.Immutable;

namespace Microsoft.ML
{
    public static class PermutationFeatureImportanceExtensions
    {
        /// <summary>
        /// Permutation feature importance (PFI) is a technique to determine the global importance features in a trained
        /// machine learning model. PFI works by taking a labeled dataset, and then, going feature by feature, the values
        /// for that feature are permuted, and the resulting change in the metric values for the task is computed. The
        /// larger the change in the evaluation metric, the more important the feature is to the model. This is a simple
        /// feature importance scheme motivated by Breiman in his Random Forest paper, in section 10
        /// (Breiman. "Random Forests." Machine Learning, 2001.) The advantage of the PFI method is that it is model
        /// agnostic -- it works with any model that can be evaluated -- and it can use any dataset, not just the training
        /// set, to compute feature importance metrics.
        /// /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[PFI](~/../docs/samples/Microsoft.ML.Samples/Dynamic/PermutationFeatureImportance.cs)]
        /// ]]>
        /// </format>
        /// </example>
        /// </summary>
        /// <param name="ctx">The regression context.</param>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="data">The evaluation data set.</param>
        /// <param name="label">Label column name.</param>
        /// <param name="features">Feature column names.</param>
        /// <param name="useFeatureWeightFilter">Use features weight to pre-filter features.</param>
        /// <param name="topExamples">Limit the number of examples to evaluate on. null means examples (up to ~ 2 bln) from input will be used.</param>
        /// <returns>Array of per-feature 'contributions' to the score.</returns>
        public static ImmutableArray<RegressionEvaluator.Result>
            PermutationFeatureImportance(
                this RegressionContext ctx,
                IPredictionTransformer<IPredictor> model,
                IDataView data,
                string label = DefaultColumnNames.Label,
                string features = DefaultColumnNames.Features,
                bool useFeatureWeightFilter = false,
                int? topExamples = null)
        {
            return PermutationFeatureImportance<RegressionEvaluator.Result>.GetImportanceMetricsMatrix(
                            CatalogUtils.GetEnvironment(ctx),
                            model,
                            data,
                            idv => ctx.Evaluate(idv, label),
                            RegressionDelta,
                            features,
                            useFeatureWeightFilter,
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
