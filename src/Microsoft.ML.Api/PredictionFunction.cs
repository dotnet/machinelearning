// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;

namespace Microsoft.ML.Runtime.Data
{
    public sealed class PredictionFunction<TSrc, TDst>
                where TSrc : class
                where TDst : class, new()
    {
        private readonly PredictionEngine<TSrc, TDst> _engine;

        public PredictionFunction(IHostEnvironment env, ITransformer transformer)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));

            IDataView dv = env.CreateDataView(new TSrc[0]);
            _engine = env.CreatePredictionEngine<TSrc, TDst>(transformer.Transform(dv));
        }

        public TDst Predict(TSrc example) => _engine.Predict(example);
    }

    public static class PredictionFunctionExtensions
    {
        public static PredictionFunction<TSrc, TDst> MakePredictionFunction<TSrc, TDst>(this ITransformer transformer, IHostEnvironment env)
                where TSrc : class
                where TDst : class, new()
            => new PredictionFunction<TSrc, TDst>(env, transformer);
    }
}
