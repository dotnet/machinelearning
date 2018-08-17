// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

[assembly: LoadableClass(typeof(ITransformer), typeof(TransformerChain), null, typeof(SignatureLoadModel),
    "Transformer chain", TransformerChain.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class TransformerChain<TLastTransformer> : ITransformer, ICanSaveModel
        where TLastTransformer : class, ITransformer
    {
        private readonly ITransformer[] _transformers;
        private readonly TransformerScope[] _scopes;
        public readonly TLastTransformer LastTransformer;

        private const string TransformDirTemplate = "Transform_{0:000}";

        internal TransformerChain(ITransformer[] transformers, TransformerScope[] scopes)
        {
            _transformers = transformers;
            _scopes = scopes;
            LastTransformer = transformers.LastOrDefault() as TLastTransformer;
            Contracts.Check((transformers.Length > 0) == (LastTransformer != null));
            Contracts.Check(transformers.Length == scopes.Length);
        }

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
                _scopes = transformers.Select(x => TransformerScope.Everything).ToArray();
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

        public TransformerChain<ITransformer> GetModelFor(TransformerScope scopeFilter)
        {
            var xfs = new List<ITransformer>();
            var scopes = new List<TransformerScope>();
            for (int i=0; i<_transformers.Length; i++)
            {
                if ((_scopes[i] & scopeFilter) != TransformerScope.None)
                {
                    xfs.Add(_transformers[i]);
                    scopes.Add(_scopes[i]);
                }
            }
            return new TransformerChain<ITransformer>(xfs.ToArray(), scopes.ToArray());
        }

        public TransformerChain<TNewLast> Append<TNewLast>(TNewLast transformer, TransformerScope scope)
            where TNewLast : class, ITransformer
        {
            Contracts.CheckValue(transformer, nameof(transformer));
            return new TransformerChain<TNewLast>(_transformers.Append(transformer).ToArray(), _scopes.Append(scope).ToArray());
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.Writer.Write(_transformers.Length);

            for (int i = 0; i < _transformers.Length; i++)
            {
                ctx.Writer.Write((int)_scopes[i]);
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.SaveModel(_transformers[i], dirName);
            }
        }

        internal TransformerChain(IHostEnvironment env, ModelLoadContext ctx)
        {
            int len = ctx.Reader.ReadInt32();
            _transformers = new ITransformer[len];
            _scopes = new TransformerScope[len];
            for (int i = 0; i < len; i++)
            {
                _scopes[i] = (TransformerScope)(ctx.Reader.ReadInt32());
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

        public void SavePipeline(IHostEnvironment env, Stream outputStream)
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
        public static CompositeReader<IMultiStreamSource, ITransformer> LoadPipeline(IHostEnvironment env, Stream stream)
        {
            using (var rep = RepositoryReader.Open(stream, env))
            {
                ModelLoadContext.LoadModel<IDataReader<IMultiStreamSource>, SignatureLoadModel>(env, out var reader, rep, "Reader");
                ModelLoadContext.LoadModel<TransformerChain<ITransformer>, SignatureLoadModel>(env, out var transformerChain, rep, "TransformerChain");
                return new CompositeReader<IMultiStreamSource, ITransformer>(reader, transformerChain);
            }
        }
    }

    [Flags]
    public enum TransformerScope
    {
        None = 0,
        Training = 1 << 0,
        Testing = 1 << 1,
        Scoring = 1 << 2,
        TrainTest = Training | Testing,
        Everything = Training | Testing | Scoring
    }

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
        public static CompositeReaderEstimator<TSource, TTrans> Append<TSource, TTrans>(
            this IDataReaderEstimator<TSource, IDataReader<TSource>> start, IEstimator<TTrans> estimator)
            where TTrans : class, ITransformer
        {
            return new CompositeReaderEstimator<TSource, ITransformer>(start).Append(estimator);
        }

        public static EstimatorChain<TTrans> Append<TTrans>(
            this IEstimator<ITransformer> start, IEstimator<TTrans> estimator, TransformerScope scope = TransformerScope.Everything)
            where TTrans : class, ITransformer
        {
            return new EstimatorChain<ITransformer>().Append(start).Append(estimator, scope);
        }
    }
}
