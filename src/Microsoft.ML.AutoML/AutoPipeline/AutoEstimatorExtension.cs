using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.AutoPipeline
{
    internal static class AutoEstimatorExtension
    {
        public static AutoEstimatorChain<TTrain> Append<TTrain>(this IEstimator<ITransformer> start, IEstimator<TTrain> estimator, TransformerScope scope, ParameterSet parameters)
            where TTrain: class, ITransformer
        {
            throw new NotImplementedException();
        }
    }
}
