// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Linq;

namespace Microsoft.ML.Runtime.Data
{
    public sealed class EstimatorChain<TLastTransformer> : IEstimator<TransformerChain<TLastTransformer>>
        where TLastTransformer : class, ITransformer
    {
        private readonly TransformerScope[] _scopes;

        private readonly IEstimator<ITransformer>[] _estimators;
        public readonly IEstimator<TLastTransformer> LastEstimator;

        private EstimatorChain(IEstimator<ITransformer>[] estimators, TransformerScope[] scopes)
        {
            _estimators = estimators;
            _scopes = scopes;
            LastEstimator = estimators.Last() as IEstimator<TLastTransformer>;

            Contracts.Check(LastEstimator != null);
            Contracts.Check(Utils.Size(estimators) == Utils.Size(scopes));
        }

        public EstimatorChain()
        {
            _estimators = new IEstimator<ITransformer>[0];
            LastEstimator = null;
            _scopes = new TransformerScope[0];
        }

        public TransformerChain<TLastTransformer> Fit(IDataView input)
        {
            var dv = input;
            var xfs = new ITransformer[_estimators.Length];
            for (int i = 0; i < _estimators.Length; i++)
            {
                var est = _estimators[i];
                xfs[i] = est.Fit(dv);
                dv = xfs[i].Transform(dv);
            }

            return new TransformerChain<TLastTransformer>(xfs, _scopes);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var s = inputSchema;
            foreach (var est in _estimators)
            {
                s = est.GetOutputSchema(s);
                if (s == null)
                    return null;
            }
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
