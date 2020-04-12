using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.AutoML.AutoPipeline.Sweeper;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.AutoPipeline
{
    internal static class AutoEstimatorExtension
    {
        public static AutoEstimatorChain<TNewTrain>
            Append<TLastTrain, TNewTrain, TOption>(this EstimatorChain<TLastTrain> estimatorChain,
                                                   Func<TOption, IEstimator<TNewTrain>> estimatorBuilder,
                                                   OptionBuilder<TOption> parameters,
                                                   AutoML.AutoPipeline.Sweeper.ISweeper sweeper,
                                                   TransformerScope scope = TransformerScope.Everything)
            where TLastTrain: class, ITransformer
            where TNewTrain: class, ITransformer
            where TOption: class
        {
            var autoEstimator = new AutoEstimator<TNewTrain, TOption>(estimatorBuilder, parameters, sweeper);

            return new AutoEstimatorChain<TLastTrain>(estimatorChain.GetEstimators, estimatorChain.GetScopes)
                       .Append(autoEstimator, scope);
        }
    }
}
