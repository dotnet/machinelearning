using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.AutoML.AutoPipeline.Sweeper;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Tensorflow;

namespace Microsoft.ML.AutoML.AutoPipeline
{
    internal class AutoEstimatorChain<TLastTransformer> : IEstimator<TransformerChain<TLastTransformer>>
        where TLastTransformer : class, ITransformer
    {
        private readonly TransformerScope[] _scopes;
        private readonly IEstimator<ITransformer>[] _estimators;
        public readonly IEstimator<TLastTransformer> LastEstimator;

        public AutoEstimatorChain(IEstimator<ITransformer>[] estimators, TransformerScope[] scopes)
        {
            Contracts.AssertValueOrNull(estimators);
            Contracts.AssertValueOrNull(scopes);
            Contracts.Assert(Utils.Size(estimators) == Utils.Size(scopes));
            _estimators = estimators ?? new IEstimator<ITransformer>[0];
            _scopes = scopes ?? new TransformerScope[0];
            LastEstimator = estimators.LastOrDefault() as IEstimator<TLastTransformer>;
        }

        public AutoEstimatorChain()
        {
            _estimators = new IEstimator<ITransformer>[0];
            _scopes = new TransformerScope[0];
            LastEstimator = null;
        }

        public TransformerChain<TLastTransformer> Fit(IDataView input)
        {
            GetOutputSchema(SchemaShape.Create(input.Schema));

            while (true)
            {

            }
        }

        public IEnumerable<(TransformerChain<TLastTransformer>, AutoML.AutoPipeline.Sweeper.ISweeper)> Fits(IDataView input)
        {
            GetOutputSchema(SchemaShape.Create(input.Schema));

            // index of autoEstimator
            var autoEstimator = _estimators.Where(_est => _est is IAutoEstimator).FirstOrDefault() as IAutoEstimator;
            while (true)
            {
                // check sweeper
                if (autoEstimator.Sweeper.MoveNext() == false)
                {
                    yield break;
                }

                IDataView current = input;
                var xfs = new ITransformer[_estimators.Length];
                for (int i = 0; i < _estimators.Length; i++)
                {
                    var est = _estimators[i];
                    xfs[i] = est.Fit(current);
                    current = xfs[i].Transform(current);
                }

                yield return (new TransformerChain<TLastTransformer>(xfs, _scopes), autoEstimator.Sweeper);
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var s = inputSchema;
            foreach (var est in _estimators)
                s = est.GetOutputSchema(s);
            return s;
        }

        public AutoEstimatorChain<TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TNewTrans: class, ITransformer
        {
            Contracts.CheckValue(estimator, nameof(estimator));
            return new AutoEstimatorChain<TNewTrans>(_estimators.AppendElement(estimator), _scopes.AppendElement(scope));
        }
    }
}
