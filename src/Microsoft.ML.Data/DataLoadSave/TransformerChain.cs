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

        public static void SaveTo(this ITransformer transformer, IHostEnvironment env, Stream outputStream)
        {
            using (var ch = env.Start("Saving pipeline"))
            {
                using (var rep = RepositoryWriter.CreateNew(outputStream, ch))
                {
                    ch.Trace("Saving transformer chain");
                    ModelSaveContext.SaveModel(rep, transformer, LoaderSignature);
                    rep.Commit();
                }
            }
        }

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
