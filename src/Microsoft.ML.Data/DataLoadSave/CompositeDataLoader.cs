// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(CompositeDataLoader<IMultiStreamSource, ITransformer>.Summary, typeof(CompositeDataLoader<IMultiStreamSource, ITransformer>), null, typeof(SignatureLoadModel),
    "Composite Loader", CompositeDataLoader<IMultiStreamSource, ITransformer>.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This class represents a data loader that applies a transformer chain after loading.
    /// It also has methods to save itself to a repository.
    /// </summary>
    public sealed class CompositeDataLoader<TSource, TLastTransformer> : IDataLoader<TSource>
        where TLastTransformer : class, ITransformer
    {
        internal const string TransformerDirectory = TransformerChain.LoaderSignature;
        private const string LoaderDirectory = "Loader";
        private const string LegacyLoaderDirectory = "Reader";

        /// <summary>
        /// The underlying data loader.
        /// </summary>
        public readonly IDataLoader<TSource> Loader;
        /// <summary>
        /// The chain of transformers (possibly empty) that are applied to data upon loading.
        /// </summary>
        public readonly TransformerChain<TLastTransformer> Transformer;

        public CompositeDataLoader(IDataLoader<TSource> loader, TransformerChain<TLastTransformer> transformerChain = null)
        {
            Contracts.CheckValue(loader, nameof(loader));
            Contracts.CheckValueOrNull(transformerChain);

            Loader = loader;
            Transformer = transformerChain ?? new TransformerChain<TLastTransformer>();
        }

        private CompositeDataLoader(IHost host, ModelLoadContext ctx)
        {
            if (!ctx.LoadModelOrNull<IDataLoader<TSource>, SignatureLoadModel>(host, out Loader, LegacyLoaderDirectory))
                ctx.LoadModel<IDataLoader<TSource>, SignatureLoadModel>(host, out Loader, LoaderDirectory);
            ctx.LoadModel<TransformerChain<TLastTransformer>, SignatureLoadModel>(host, out Transformer, TransformerDirectory);
        }

        private static CompositeDataLoader<TSource, TLastTransformer> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoaderSignature);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply("Loading Model", ch => new CompositeDataLoader<TSource, TLastTransformer>(h, ctx));
        }

        /// <summary>
        /// Produce the data view from the specified input.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        public IDataView Load(TSource input)
        {
            var idv = Loader.Load(input);
            idv = Transformer.Transform(idv);
            return idv;
        }

        public DataViewSchema GetOutputSchema()
        {
            var s = Loader.GetOutputSchema();
            return Transformer.GetOutputSchema(s);
        }

        /// <summary>
        /// Append a new transformer to the end.
        /// </summary>
        /// <returns>The new composite data loader</returns>
        public CompositeDataLoader<TSource, TNewLast> AppendTransformer<TNewLast>(TNewLast transformer)
            where TNewLast : class, ITransformer
        {
            Contracts.CheckValue(transformer, nameof(transformer));

            return new CompositeDataLoader<TSource, TNewLast>(Loader, Transformer.Append(transformer));
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.SaveModel(Loader, LoaderDirectory);
            ctx.SaveModel(Transformer, TransformerDirectory);
        }

        internal const string Summary = "A model loader that encapsulates a data loader and a transformer chain.";

        internal const string LoaderSignature = "CompositeLoader";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CMPSTLDR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CompositeDataLoader<,>).Assembly.FullName);
        }
    }
}
