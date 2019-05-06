// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// An estimator class for composite data loader.
    /// It can be used to build a 'trainable smart data loader', although this pattern is not very common.
    /// </summary>
    public sealed class CompositeLoaderEstimator<TSource, TLastTransformer> : IDataLoaderEstimator<TSource, CompositeDataLoader<TSource, TLastTransformer>>
        where TLastTransformer : class, ITransformer
    {
        private readonly IDataLoaderEstimator<TSource, IDataLoader<TSource>> _start;
        private readonly EstimatorChain<TLastTransformer> _estimatorChain;

        public CompositeLoaderEstimator(IDataLoaderEstimator<TSource, IDataLoader<TSource>> start, EstimatorChain<TLastTransformer> estimatorChain = null)
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValueOrNull(estimatorChain);

            _start = start;
            _estimatorChain = estimatorChain ?? new EstimatorChain<TLastTransformer>();

            // REVIEW: enforce that estimator chain can read the loader's schema.
            // Right now it throws.
            // GetOutputSchema();
        }

        public CompositeDataLoader<TSource, TLastTransformer> Fit(TSource input)
        {
            var start = _start.Fit(input);
            var idv = start.Load(input);

            var xfChain = _estimatorChain.Fit(idv);
            return new CompositeDataLoader<TSource, TLastTransformer>(start, xfChain);
        }

        public SchemaShape GetOutputSchema()
        {
            var shape = _start.GetOutputSchema();
            return _estimatorChain.GetOutputSchema(shape);
        }

        /// <summary>
        /// Create a new loader estimator, by appending another estimator to the end of this loader estimator.
        /// </summary>
        public CompositeLoaderEstimator<TSource, TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator)
            where TNewTrans : class, ITransformer
        {
            Contracts.CheckValue(estimator, nameof(estimator));

            return new CompositeLoaderEstimator<TSource, TNewTrans>(_start, _estimatorChain.Append(estimator));
        }
    }
}
