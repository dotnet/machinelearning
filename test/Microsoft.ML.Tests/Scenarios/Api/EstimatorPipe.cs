using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Tests.Scenarios.Api;
using System.Collections.Generic;
using System.IO;
using System.Linq;

[assembly: LoadableClass(typeof(ITransformer), typeof(TransformerChain), null, typeof(SignatureLoadModel),
    "Transformer chain", TransformerChain.LoaderSignature)]

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public sealed class TransformerChain<TLastTransformer> : ITransformer, ICanSaveModel
        where TLastTransformer : class, ITransformer
    {
        private readonly ITransformer[] _transformers;
        public readonly TLastTransformer LastTransformer;

        private const string TransformDirTemplate = "Transform_{0:000}";

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

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.Writer.Write(_transformers.Length);

            for (int i = 0; i < _transformers.Length; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.SaveModel(_transformers[i], dirName);
            }
        }

        internal TransformerChain(IHostEnvironment env, ModelLoadContext ctx)
        {
            int len = ctx.Reader.ReadInt32();
            _transformers = new ITransformer[len];
            for (int i = 0; i < len; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.LoadModel<ITransformer, SignatureLoadModel>(env, out _transformers[i], dirName);
            }
            if (len > 0)
                LastTransformer = _transformers[len - 1] as TLastTransformer;
            else
                LastTransformer = null;
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "XF  PIPE",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: TransformerChain.LoaderSignature);
        }

    }

    public static class TransformerChain
    {
        public const string LoaderSignature = "TransformerChain";

        public static ITransformer Create(IHostEnvironment env, ModelLoadContext ctx) => new TransformerChain<ITransformer>(env, ctx);
    }

    public sealed class CompositeReader<TSource, TLastTransformer> : IDataReader<TSource>
        where TLastTransformer : class, ITransformer
    {
        public readonly IDataReader<TSource> Reader;
        public readonly TransformerChain<TLastTransformer> Transformer;

        public CompositeReader(IDataReader<TSource> reader, TransformerChain<TLastTransformer> transformerChain = null)
        {
            Contracts.CheckValue(reader, nameof(reader));
            Contracts.CheckValueOrNull(transformerChain);
            Reader = reader;
            Transformer = transformerChain ?? new TransformerChain<TLastTransformer>();
        }

        public IDataView Read(TSource input)
        {
            var idv = Reader.Read(input);
            idv = Transformer.Transform(idv);
            return idv;
        }

        public ISchema GetOutputSchema()
        {
            var s = Reader.GetOutputSchema();
            s = Transformer.GetOutputSchema(s);
            return s;
        }

        public CompositeReader<TSource, TNewLastTransformer> Append<TNewLastTransformer>(TNewLastTransformer transformer)
            where TNewLastTransformer : class, ITransformer
        {
            return new CompositeReader<TSource, TNewLastTransformer>(Reader, Transformer.Append(transformer));
        }

        public void Save(IHostEnvironment env, Stream outputStream)
        {
            using (var ch = env.Start("Saving model"))
            {
                using (var rep = RepositoryWriter.CreateNew(outputStream, ch))
                {
                    ch.Trace("Saving data reader");
                    ModelSaveContext.SaveModel(rep, Reader, "Reader");

                    ch.Trace("Saving transformer chain");
                    ModelSaveContext.SaveModel(rep, Transformer, "TransformerChain");
                    rep.Commit();
                }
            }
        }
    }

    public static class CompositeReader
    {
        public static CompositeReader<IMultiStreamSource, ITransformer> LoadModel(IHostEnvironment env, Stream stream)
        {
            using (var rep = RepositoryReader.Open(stream, env))
            {
                ModelLoadContext.LoadModel<IDataReader<IMultiStreamSource>, SignatureLoadModel>(env, out var reader, rep, "Reader");
                ModelLoadContext.LoadModel<TransformerChain<ITransformer>, SignatureLoadModel>(env, out var transformerChain, rep, "TransformerChain");
                return new CompositeReader<IMultiStreamSource, ITransformer>(reader, transformerChain);
            }
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
            where TTrans : class, ITransformer
        {
            return new CompositeReaderEstimator<TSource, ITransformer>(start).Append(estimator);
        }
    }
}
