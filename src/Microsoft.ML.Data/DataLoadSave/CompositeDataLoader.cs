// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This class represents a data loader that applies a transformer chain after loading.
    /// It also has methods to save itself to a repository.
    /// </summary>
    public sealed class CompositeDataLoader<TSource, TLastTransformer> : IDataLoader<TSource>
        where TLastTransformer : class, ITransformer
    {
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

        /// <summary>
        /// Save the contents to a stream, as a "model file".
        /// </summary>
        public void SaveTo(IHostEnvironment env, Stream outputStream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(outputStream, nameof(outputStream));

            env.Check(outputStream.CanWrite && outputStream.CanSeek, "Need a writable and seekable stream to save");
            using (var ch = env.Start("Saving pipeline"))
            {
                using (var rep = RepositoryWriter.CreateNew(outputStream, ch))
                {
                    ch.Trace("Saving data loader");
                    ModelSaveContext.SaveModel(rep, Loader, "Reader");

                    ch.Trace("Saving transformer chain");
                    ModelSaveContext.SaveModel(rep, Transformer, TransformerChain.LoaderSignature);
                    rep.Commit();
                }
            }
        }
    }

    /// <summary>
    /// Utility class to facilitate loading from a stream.
    /// </summary>
    [BestFriend]
    internal static class CompositeDataLoader
    {
        /// <summary>
        /// Save the contents to a stream, as a "model file".
        /// </summary>
        public static void SaveTo<TSource>(this IDataLoader<TSource> loader, IHostEnvironment env, Stream outputStream)
            => new CompositeDataLoader<TSource, ITransformer>(loader).SaveTo(env, outputStream);

        /// <summary>
        /// Load the pipeline from stream.
        /// </summary>
        public static CompositeDataLoader<IMultiStreamSource, ITransformer> LoadFrom(IHostEnvironment env, Stream stream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(stream, nameof(stream));

            env.Check(stream.CanRead && stream.CanSeek, "Need a readable and seekable stream to load");
            using (var rep = RepositoryReader.Open(stream, env))
            using (var ch = env.Start("Loading pipeline"))
            {
                ch.Trace("Loading data loader");
                ModelLoadContext.LoadModel<IDataLoader<IMultiStreamSource>, SignatureLoadModel>(env, out var loader, rep, "Reader");

                ch.Trace("Loader transformer chain");
                ModelLoadContext.LoadModel<TransformerChain<ITransformer>, SignatureLoadModel>(env, out var transformerChain, rep, TransformerChain.LoaderSignature);
                return new CompositeDataLoader<IMultiStreamSource, ITransformer>(loader, transformerChain);
            }
        }
    }
}
