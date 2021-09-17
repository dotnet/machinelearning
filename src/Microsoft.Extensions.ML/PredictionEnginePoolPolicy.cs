// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Extensions.ObjectPool;
using Microsoft.ML;

namespace Microsoft.Extensions.ML
{
    /// <summary>
    /// <see cref="PooledObjectPolicy{T}"/> for <see cref="PredictionEngine{TData, TPrediction}"/>
    /// which is responsible for creating pooled objects, and when to return objects to the pool.
    /// </summary>
    internal class PredictionEnginePoolPolicy<TData, TPrediction>
        : PooledObjectPolicy<PredictionEngine<TData, TPrediction>>
        where TData : class
        where TPrediction : class, new()
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;

        /// <summary>
        /// Initializes a new instance of <see cref="PredictionEnginePoolPolicy{TData, TPrediction}"/>.
        /// </summary>
        /// <param name="mlContext">
        /// <see cref="MLContext"/> used to load the model.
        /// </param>
        /// <param name="model">The transformer to use for prediction.</param>
        public PredictionEnginePoolPolicy(MLContext mlContext, ITransformer model)
        {
            _mlContext = mlContext;
            _model = model;
        }

        /// <inheritdoc />
        public override PredictionEngine<TData, TPrediction> Create() =>
            _mlContext.Model.CreatePredictionEngine<TData, TPrediction>(_model);

        /// <inheritdoc />
        public override bool Return(PredictionEngine<TData, TPrediction> obj) => true;
    }
}
