// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
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
        // Host is not null iff there is any 'true' values in _needCacheAfter (in this case, we need to create an instance of
        // CacheDataView.
        private readonly IHost _host;
        private readonly TransformerScope[] _scopes;
        private readonly IEstimator<ITransformer>[] _estimators;
        private readonly bool[] _needCacheAfter;
        public readonly IEstimator<TLastTransformer> LastEstimator;

        private EstimatorChain(IHostEnvironment env, IEstimator<ITransformer>[] estimators, TransformerScope[] scopes, bool[] needCacheAfter)
        {
            Contracts.AssertValueOrNull(env);
            Contracts.AssertValueOrNull(estimators);
            Contracts.AssertValueOrNull(scopes);
            Contracts.AssertValueOrNull(needCacheAfter);
            Contracts.Assert(Utils.Size(estimators) == Utils.Size(scopes));
            Contracts.Assert(Utils.Size(estimators) == Utils.Size(needCacheAfter));

            _host = env?.Register(nameof(EstimatorChain<TLastTransformer>));
            _estimators = estimators ?? new IEstimator<ITransformer>[0];
            _scopes = scopes ?? new TransformerScope[0];
            LastEstimator = estimators.LastOrDefault() as IEstimator<TLastTransformer>;
            _needCacheAfter = needCacheAfter ?? new bool[0];

            Contracts.Assert((_host != null) == _needCacheAfter.Any(x => x));
            Contracts.Assert((_estimators.Length > 0) == (LastEstimator != null));
        }

        /// <summary>
        /// Create an empty estimator chain.
        /// </summary>
        public EstimatorChain()
        {
            _host = null;
            _estimators = new IEstimator<ITransformer>[0];
            _scopes = new TransformerScope[0];
            _needCacheAfter = new bool[0];
            LastEstimator = null;
        }

        public TransformerChain<TLastTransformer> Fit(IDataView input)
        {
            // Before fitting, run schema propagation.
            GetOutputSchema(SchemaShape.Create(input.Schema));

            IDataView current = input;
            var xfs = new ITransformer[_estimators.Length];
            for (int i = 0; i < _estimators.Length; i++)
            {
                var est = _estimators[i];
                xfs[i] = est.Fit(current);
                current = xfs[i].Transform(current);
                if (_needCacheAfter[i] && i < _estimators.Length - 1)
                {
                    Contracts.AssertValue(_host);
                    current = new CacheDataView(_host, current, null);
                }
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
            return new EstimatorChain<TNewTrans>(_host, _estimators.AppendElement(estimator), _scopes.AppendElement(scope), _needCacheAfter.AppendElement(false));
        }

        /// <summary>
        /// Append a 'caching checkpoint' to the estimator chain. This will ensure that the downstream estimators will be trained against
        /// cached data. It is helpful to have a caching checkpoint before trainers that take multiple data passes.
        /// </summary>
        /// <param name="env">The host environment to use for caching.</param>
        public EstimatorChain<TLastTransformer> AppendCacheCheckpoint(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));

            if (_estimators.Length == 0 || _needCacheAfter.Last())
            {
                // If there are no estimators, or if we already need to cache after this, we don't need to do anything else.
                return this;
            }

            bool[] newNeedCache = _needCacheAfter.ToArray();
            newNeedCache[newNeedCache.Length - 1] = true;
            return new EstimatorChain<TLastTransformer>(env, _estimators, _scopes, newNeedCache);
        }
    }
}
