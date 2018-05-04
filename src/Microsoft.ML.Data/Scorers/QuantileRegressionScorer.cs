// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(typeof(IDataScorerTransform), typeof(QuantileRegressionScorerTransform), typeof(QuantileRegressionScorerTransform.Arguments),
    typeof(SignatureDataScorer), "Quantile Regression Scorer", "QuantileRegressionScorer", MetadataUtils.Const.ScoreColumnKind.QuantileRegression)]

[assembly: LoadableClass(typeof(ISchemaBindableMapper), typeof(QuantileRegressionScorerTransform), typeof(QuantileRegressionScorerTransform.Arguments),
    typeof(SignatureBindableMapper), "Quantile Regression Mapper", "QuantileRegressionScorer", MetadataUtils.Const.ScoreColumnKind.QuantileRegression)]

namespace Microsoft.ML.Runtime.Data
{
    public static class QuantileRegressionScorerTransform
    {
        public sealed class Arguments : ScorerArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "List of numbers between 0 and 1 (comma-separated) to get quantile statistics. The default value outputs Five point summary")]
            public string Quantiles = "0,0.25,0.5,0.75,1";
        }

        public static IDataScorerTransform Create(IHostEnvironment env, Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            return new GenericScorer(env, args, data, mapper, trainSchema);
        }

        public static ISchemaBindableMapper Create(IHostEnvironment env, Arguments args, IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(predictor, nameof(predictor));

            var pred = predictor as IQuantileRegressionPredictor;
            env.Check(pred != null, "Predictor doesn't support quantile regression");

            var quantiles = ParseQuantiles(args.Quantiles);
            return pred.CreateMapper(quantiles);
        }

        private static Double[] ParseQuantiles(string quantiles)
        {
            Contracts.CheckUserArg(quantiles != null, nameof(Arguments.Quantiles), "Quantiles are required");
            Double[] quantilesArray = quantiles.Split(',').Select(
                v =>
                {
                    Double q;
                    if (!Double.TryParse(v, out q))
                        throw Contracts.ExceptUserArg(nameof(Arguments.Quantiles), "Cannot parse quantile '{0}' as double.", v);
                    Contracts.CheckUserArg(0 <= q && q <= 1, nameof(Arguments.Quantiles), "Quantile must be between 0 and 1.");
                    return q;
                }).ToArray();
            Contracts.CheckUserArg(quantilesArray.Length > 0, nameof(Arguments.Quantiles), "There must be at least one quantile.");
            return quantilesArray;
        }
    }
}