// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Linq;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Represents a chain (potentially empty) of estimators that end with a <typeparamref name="TLastTransformer"/>.
    /// If the chain is empty, <typeparamref name="TLastTransformer"/> is always <see cref="ITransformer"/>.
    /// </summary>
    public sealed class EstimatorChain<TLastTransformer> : IEstimator<TransformerChain<TLastTransformer>>
        where TLastTransformer : class, ITransformer
    {
        private readonly TransformerScope[] _scopes;
        private readonly IEstimator<ITransformer>[] _estimators;
        public readonly IEstimator<TLastTransformer> LastEstimator;

        private EstimatorChain(IEstimator<ITransformer>[] estimators, TransformerScope[] scopes)
        {
            Contracts.AssertValueOrNull(estimators);
            Contracts.AssertValueOrNull(scopes);
            Contracts.Assert(Utils.Size(estimators) == Utils.Size(scopes));

            _estimators = estimators ?? new IEstimator<ITransformer>[0];
            _scopes = scopes ?? new TransformerScope[0];
            LastEstimator = estimators.LastOrDefault() as IEstimator<TLastTransformer>;

            Contracts.Assert((_estimators.Length > 0) == (LastEstimator != null));
        }

        /// <summary>
        /// Create an empty estimator chain.
        /// </summary>
        public EstimatorChain()
        {
            _estimators = new IEstimator<ITransformer>[0];
            _scopes = new TransformerScope[0];
            LastEstimator = null;
        }

        public TransformerChain<TLastTransformer> Fit(IDataView input, IDataView validationData = null, IPredictor initialPredictor = null)
        {
            // REVIEW: before fitting, run schema propagation.
            // Currently, it throws.
            // GetOutputSchema(SchemaShape.Create(input.Schema);

            IDataView current = input;
            var xfs = new ITransformer[_estimators.Length];
            for (int i = 0; i < _estimators.Length; i++)
            {
                var est = _estimators[i];
                xfs[i] = est.Fit(current);
                current = xfs[i].Transform(current);
            }

            return new TransformerChain<TLastTransformer>(xfs, _scopes);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var s = inputSchema;
            foreach (var est in _estimators)
                s = est.GetOutputSchema(s);
            return s;
        }

        public EstimatorChain<TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans : class, ITransformer
        {
            Contracts.CheckValue(estimator, nameof(estimator));
            return new EstimatorChain<TNewTrans>(_estimators.Append(estimator).ToArray(), _scopes.Append(scope).ToArray());
        }
    }
}
