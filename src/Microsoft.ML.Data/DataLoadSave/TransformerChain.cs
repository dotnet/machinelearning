// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

[assembly: LoadableClass(typeof(TransformerChain<ITransformer>), typeof(TransformerChain), null, typeof(SignatureLoadModel),
    "Transformer chain", TransformerChain.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This enum allows for 'tagging' the estimators (and subsequently transformers) in the chain to be used
    /// 'only for training', 'for training and evaluation' etc.
    /// Most notable example is, transformations over the label column should not be used for scoring, so the scope
    /// should be <see cref="Training"/> or <see cref="TrainTest"/>.
    /// </summary>
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

    /// <summary>
    /// A chain of transformers (possibly empty) that end with a <typeparamref name="TLastTransformer"/>.
    /// For an empty chain, <typeparamref name="TLastTransformer"/> is always <see cref="ITransformer"/>.
    /// </summary>
    public sealed class TransformerChain<TLastTransformer> : ITransformer, ICanSaveModel, IEnumerable<ITransformer>
    where TLastTransformer : class, ITransformer
    {
        private readonly ITransformer[] _transformers;
        private readonly TransformerScope[] _scopes;
        public readonly TLastTransformer LastTransformer;

        private const string TransformDirTemplate = "Transform_{0:000}";

        public bool IsRowToRowMapper => _transformers.All(t => t.IsRowToRowMapper);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "XF CHAIN",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: TransformerChain.LoaderSignature,
                loaderAssemblyName: typeof(TransformerChain<>).Assembly.FullName);
        }

        /// <summary>
        /// Create a transformer chain by specifying transformers and their scopes.
        /// </summary>
        /// <param name="transformers">Transformers to be chained.</param>
        /// <param name="scopes">Transformer scopes, parallel to <paramref name="transformers"/>.</param>
        public TransformerChain(IEnumerable<ITransformer> transformers, IEnumerable<TransformerScope> scopes)
        {
            Contracts.CheckValueOrNull(transformers);
            Contracts.CheckValueOrNull(scopes);

            _transformers = transformers?.ToArray() ?? new ITransformer[0];
            _scopes = scopes?.ToArray() ?? new TransformerScope[0];
            LastTransformer = transformers.LastOrDefault() as TLastTransformer;

            Contracts.Check((_transformers.Length > 0) == (LastTransformer != null));
            Contracts.Check(_transformers.Length == _scopes.Length);
        }

        /// <summary>
        /// Create a transformer chain by specifying all the transformers. The scopes are assumed to be
        /// <see cref="TransformerScope.Everything"/>.
        /// </summary>
        /// <param name="transformers"></param>
        public TransformerChain(params ITransformer[] transformers)
        {
            Contracts.CheckValueOrNull(transformers);

            if (Utils.Size(transformers) == 0)
            {
                _transformers = new ITransformer[0];
                _scopes = new TransformerScope[0];
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
            Contracts.CheckValue(inputSchema, nameof(inputSchema));

            var s = inputSchema;
            foreach (var xf in _transformers)
                s = xf.GetOutputSchema(s);
            return s;
        }

        public IDataView Transform(IDataView input)
        {
            Contracts.CheckValue(input, nameof(input));

            // Trigger schema propagation prior to transforming.
            // REVIEW: does this actually constitute 'early warning', given that Transform call is lazy anyway?
            GetOutputSchema(input.Schema);

            var dv = input;
            foreach (var xf in _transformers)
                dv = xf.Transform(dv);
            return dv;
        }

        public TransformerChain<ITransformer> GetModelFor(TransformerScope scopeFilter)
        {
            var xfs = new List<ITransformer>();
            var scopes = new List<TransformerScope>();
            for (int i = 0; i < _transformers.Length; i++)
            {
                if ((_scopes[i] & scopeFilter) != TransformerScope.None)
                {
                    xfs.Add(_transformers[i]);
                    scopes.Add(_scopes[i]);
                }
            }
            return new TransformerChain<ITransformer>(xfs.ToArray(), scopes.ToArray());
        }

        public TransformerChain<TNewLast> Append<TNewLast>(TNewLast transformer, TransformerScope scope = TransformerScope.Everything)
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

        /// <summary>
        /// The loading constructor of transformer chain. Reverse of <see cref="Save(ModelSaveContext)"/>.
        /// </summary>
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

        public void SaveTo(IHostEnvironment env, Stream outputStream)
        {
            using (var ch = env.Start("Saving pipeline"))
            {
                using (var rep = RepositoryWriter.CreateNew(outputStream, ch))
                {
                    ch.Trace("Saving transformer chain");
                    ModelSaveContext.SaveModel(rep, this, TransformerChain.LoaderSignature);
                    rep.Commit();
                }
            }
        }

        public IEnumerator<ITransformer> GetEnumerator() => ((IEnumerable<ITransformer>)_transformers).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public IRowToRowMapper GetRowToRowMapper(ISchema inputSchema)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.Check(IsRowToRowMapper, nameof(GetRowToRowMapper) + " method called despite " + nameof(IsRowToRowMapper) + " being false.");

            IRowToRowMapper[] mappers = new IRowToRowMapper[_transformers.Length];
            ISchema schema = inputSchema;
            for (int i = 0; i < mappers.Length; ++i)
            {
                mappers[i] = _transformers[i].GetRowToRowMapper(schema);
                schema = mappers[i].Schema;
            }
            return new CompositeRowToRowMapper(inputSchema, mappers);
        }
    }

    /// <summary>
    /// Saving/loading routines for transformer chains.
    /// </summary>
    public static class TransformerChain
    {
        public const string LoaderSignature = "TransformerChain";

        public static TransformerChain<ITransformer> Create(IHostEnvironment env, ModelLoadContext ctx)
            => new TransformerChain<ITransformer>(env, ctx);

        /// <summary>
        /// Save any transformer to a stream by wrapping it into a transformer chain.
        /// </summary>
        public static void SaveTo(this ITransformer transformer, IHostEnvironment env, Stream outputStream)
            => new TransformerChain<ITransformer>(transformer).SaveTo(env, outputStream);

        public static TransformerChain<ITransformer> LoadFrom(IHostEnvironment env, Stream stream)
        {
            using (var rep = RepositoryReader.Open(stream, env))
            {
                ModelLoadContext.LoadModel<TransformerChain<ITransformer>, SignatureLoadModel>(env, out var transformerChain, rep, LoaderSignature);
                return transformerChain;
            }
        }
    }
}
