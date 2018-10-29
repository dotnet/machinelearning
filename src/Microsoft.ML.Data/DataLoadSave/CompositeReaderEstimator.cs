// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// An estimator class for composite data reader.
    /// It can be used to build a 'trainable smart data reader', although this pattern is not very common.
    /// </summary>
    public sealed class CompositeReaderEstimator<TSource, TLastTransformer> : IDataReaderEstimator<TSource, CompositeDataReader<TSource, TLastTransformer>>
        where TLastTransformer : class, ITransformer
    {
        private readonly IDataReaderEstimator<TSource, IDataReader<TSource>> _start;
        private readonly EstimatorChain<TLastTransformer> _estimatorChain;

        public CompositeReaderEstimator(IDataReaderEstimator<TSource, IDataReader<TSource>> start, EstimatorChain<TLastTransformer> estimatorChain = null)
        {
            Contracts.CheckValue(start, nameof(start));
            Contracts.CheckValueOrNull(estimatorChain);

            _start = start;
            _estimatorChain = estimatorChain ?? new EstimatorChain<TLastTransformer>();

            // REVIEW: enforce that estimator chain can read the reader's schema.
            // Right now it throws.
            // GetOutputSchema();
        }

        public CompositeDataReader<TSource, TLastTransformer> Fit(TSource input)
        {
            var start = _start.Fit(input);
            var idv = start.Read(input);

            var xfChain = _estimatorChain.Fit(idv);
            return new CompositeDataReader<TSource, TLastTransformer>(start, xfChain);
        }

        public SchemaShape GetOutputSchema()
        {
            var shape = _start.GetOutputSchema();
            return _estimatorChain.GetOutputSchema(shape);
        }

        /// <summary>
        /// Create a new reader estimator, by appending another estimator to the end of this reader estimator.
        /// </summary>
        public CompositeReaderEstimator<TSource, TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator)
            where TNewTrans : class, ITransformer
        {
            Contracts.CheckValue(estimator, nameof(estimator));

            return new CompositeReaderEstimator<TSource, TNewTrans>(_start, _estimatorChain.Append(estimator));
        }
    }
}
