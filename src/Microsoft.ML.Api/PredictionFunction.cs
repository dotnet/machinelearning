// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A prediction engine class, that takes instances of <typeparamref name="TSrc"/> through
    /// the transformer pipeline and produces instances of <typeparamref name="TDst"/> as outputs.
    /// </summary>
    public sealed class PredictionFunction<TSrc, TDst>
                where TSrc : class
                where TDst : class, new()
    {
        private readonly PredictionEngine<TSrc, TDst> _engine;

        /// <summary>
        /// Create an instance of <see cref="PredictionFunction{TSrc, TDst}"/>.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="transformer">The model (transformer) to use for prediction.</param>
        public PredictionFunction(IHostEnvironment env, ITransformer transformer)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformer, nameof(transformer));

            IDataView dv = env.CreateDataView(new TSrc[0]);
            _engine = env.CreatePredictionEngine<TSrc, TDst>(transformer);
        }

        /// <summary>
        /// Perform one prediction using the model.
        /// </summary>
        /// <param name="example">The object that holds values to predict from.</param>
        /// <returns>The object populated with prediction results.</returns>
        public TDst Predict(TSrc example) => _engine.Predict(example);

        /// <summary>
        /// Perform one prediction using the model.
        /// Reuses the provided prediction object, which is more efficient in high-load scenarios.
        /// </summary>
        /// <param name="example">The object that holds values to predict from.</param>
        /// <param name="prediction">The object to store the predictions in. If it's <c>null</c>, a new object is created,
        /// otherwise the provided object is used.</param>
        public void Predict(TSrc example, ref TDst prediction) => _engine.Predict(example, ref prediction);
    }

    public static class PredictionFunctionExtensions
    {
        /// <summary>
        /// Create an instance of the 'prediction function', or 'prediction machine', from a model
        /// denoted by <paramref name="transformer"/>.
        /// It will be accepting instances of <typeparamref name="TSrc"/> as input, and produce
        /// instances of <typeparamref name="TDst"/> as output.
        /// </summary>
        public static PredictionFunction<TSrc, TDst> MakePredictionFunction<TSrc, TDst>(this ITransformer transformer, IHostEnvironment env)
                where TSrc : class
                where TDst : class, new()
            => new PredictionFunction<TSrc, TDst>(env, transformer);
    }
}
