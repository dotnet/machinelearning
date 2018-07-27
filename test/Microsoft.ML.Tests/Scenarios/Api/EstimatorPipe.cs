using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public sealed class TransformerChain<TLastTransformer> : ITransformer
        where TLastTransformer : class, ITransformer
    {
        private readonly ITransformer[] _transformers;
        public readonly TLastTransformer LastTransformer;

        public TransformerChain(params ITransformer[] transformers)
        {
            if (Utils.Size(transformers) == 0)
            {
                _transformers = new ITransformer[0];
                LastTransformer = null;
            }
            else
            {
                _transformers = transformers.ToArray();
                LastTransformer = transformers.Last() as TLastTransformer;
                Contracts.Check(LastTransformer != null);
            }
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            var s = inputSchema;
            foreach (var xf in _transformers)
            {
                s = xf.GetOutputSchema(s);
                if (s == null)
                    return null;
            }
            return s;
        }

        public IDataView Transform(IDataView input)
        {
            var dv = input;
            foreach (var xf in _transformers)
            {
                dv = xf.Transform(dv);
            }
            return dv;
        }

        public IEnumerable<ITransformer> GetParts()
        {
            return _transformers;
        }

        public TransformerChain<TNewLast> Append<TNewLast>(TNewLast transformer)
            where TNewLast : class, ITransformer
        {
            Contracts.CheckValue(transformer, nameof(transformer));
            return new TransformerChain<TNewLast>(_transformers.Append(transformer).ToArray());
        }
    }

    public sealed class CompositeReader<TSource, TLastTransformer> : IDataReader<TSource>
        where TLastTransformer : class, ITransformer
    {
        private readonly IDataReader<TSource> _reader;
        private readonly TransformerChain<TLastTransformer> _transformerChain;

        public CompositeReader(IDataReader<TSource> reader, TransformerChain<TLastTransformer> transformerChain = null)
        {
            Contracts.CheckValue(reader, nameof(reader));
            Contracts.CheckValueOrNull(transformerChain);
            _reader = reader;
            _transformerChain = transformerChain ?? new TransformerChain<TLastTransformer>();
        }

        public IDataView Read(TSource input)
        {
            var idv = _reader.Read(input);
            idv = _transformerChain.Transform(idv);
            return idv;
        }

        public (IDataReader<TSource> reader, TransformerChain<TLastTransformer> transformerChain) GetParts()
        {
            return (reader: _reader, transformerChain: _transformerChain);
        }

        public ISchema GetOutputSchema()
        {
            var s = _reader.GetOutputSchema();
            s = _transformerChain.GetOutputSchema(s);
            return s;
        }

        public CompositeReader<TSource, TNewLastTransformer> Append<TNewLastTransformer>(TNewLastTransformer transformer)
            where TNewLastTransformer : class, ITransformer
        {
            return new CompositeReader<TSource, TNewLastTransformer>(_reader, _transformerChain.Append(transformer));
        }
    }

    public sealed class EstimatorChain<TLastTransformer> : IEstimator<TransformerChain<TLastTransformer>>
        where TLastTransformer : class, ITransformer
    {
        private readonly IEstimator<ITransformer>[] _estimators;
        public readonly IEstimator<TLastTransformer> LastEstimator;

        public EstimatorChain(params IEstimator<ITransformer>[] estimators)
        {
            Contracts.CheckValueOrNull(estimators);
            if (Utils.Size(estimators) == 0)
            {
                _estimators = new IEstimator<ITransformer>[0];
                LastEstimator = null;
            }
            else
            {
                _estimators = estimators;
                LastEstimator = estimators.Last() as IEstimator<TLastTransformer>;
                Contracts.Check(LastEstimator != null);
            }
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

            return new TransformerChain<TLastTransformer>(xfs);
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

        public EstimatorChain<TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator)
            where TNewTrans : class, ITransformer
        {
            Contracts.CheckValue(estimator, nameof(estimator));
            return new EstimatorChain<TNewTrans>(_estimators.Append(estimator).ToArray());
        }
    }

    public sealed class CompositeReaderEstimator<TSource, TLastTransformer> : IDataReaderEstimator<TSource, CompositeReader<TSource, TLastTransformer>>
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
        }

        public CompositeReader<TSource, TLastTransformer> Fit(TSource input)
        {
            var start = _start.Fit(input);
            var idv = start.Read(input);

            var xfChain = _estimatorChain.Fit(idv);
            return new CompositeReader<TSource, TLastTransformer>(start, xfChain);
        }

        public SchemaShape GetOutputSchema()
        {
            var shape = _start.GetOutputSchema();
            return _estimatorChain.GetOutputSchema(shape);
        }

        public (IDataReaderEstimator<TSource, IDataReader<TSource>>, EstimatorChain<TLastTransformer>) GetParts()
        {
            return (_start, _estimatorChain);
        }

        public CompositeReaderEstimator<TSource, TNewTrans> Append<TNewTrans>(IEstimator<TNewTrans> estimator)
            where TNewTrans : class, ITransformer
        {
            return new CompositeReaderEstimator<TSource, TNewTrans>(_start, _estimatorChain.Append(estimator));
        }
    }

    public static class LearningPipelineExtensions
    {
        public static CompositeReaderEstimator<TSource, ITransformer> StartPipe<TSource>(this IDataReaderEstimator<TSource, IDataReader<TSource>> start)
        {
            return new CompositeReaderEstimator<TSource, ITransformer>(start);
        }

        public static CompositeReaderEstimator<TSource, TTrans> Append<TSource, TTrans>(
            this IDataReaderEstimator<TSource, IDataReader<TSource>> start, IEstimator<TTrans> estimator)
            where TTrans: class, ITransformer
        {
            return new CompositeReaderEstimator<TSource, ITransformer>(start).Append(estimator);
        }
    }
}
