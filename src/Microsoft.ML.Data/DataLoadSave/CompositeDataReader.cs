// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Model;
using System.IO;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This class represents a data reader that applies a transformer chain after reading.
    /// It also has methods to save itself to a repository.
    /// </summary>
    public sealed class CompositeDataReader<TSource, TLastTransformer> : IDataReader<TSource>
        where TLastTransformer : class, ITransformer
    {
        /// <summary>
        /// The underlying data reader.
        /// </summary>
        public readonly IDataReader<TSource> Reader;
        /// <summary>
        /// The chain of transformers (possibly empty) that are applied to data upon reading.
        /// </summary>
        public readonly TransformerChain<TLastTransformer> Transformer;

        public CompositeDataReader(IDataReader<TSource> reader, TransformerChain<TLastTransformer> transformerChain = null)
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
            return Transformer.GetOutputSchema(s);
        }

        /// <summary>
        /// Append a new transformer to the end.
        /// </summary>
        /// <returns>The new composite data reader</returns>
        public CompositeDataReader<TSource, TNewLast> AppendTransformer<TNewLast>(TNewLast transformer)
            where TNewLast : class, ITransformer
        {
            Contracts.CheckValue(transformer, nameof(transformer));

            return new CompositeDataReader<TSource, TNewLast>(Reader, Transformer.Append(transformer));
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
                    ch.Trace("Saving data reader");
                    ModelSaveContext.SaveModel(rep, Reader, "Reader");

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
    public static class CompositeDataReader
    {
        /// <summary>
        /// Load the pipeline from stream.
        /// </summary>
        public static CompositeDataReader<IMultiStreamSource, ITransformer> LoadFrom(IHostEnvironment env, Stream stream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(stream, nameof(stream));

            env.Check(stream.CanRead && stream.CanSeek, "Need a readable and seekable stream to load");
            using (var rep = RepositoryReader.Open(stream, env))
            using (var ch = env.Start("Loading pipeline"))
            {
                ch.Trace("Loading data reader");
                ModelLoadContext.LoadModel<IDataReader<IMultiStreamSource>, SignatureLoadModel>(env, out var reader, rep, "Reader");

                ch.Trace("Loader transformer chain");
                ModelLoadContext.LoadModel<TransformerChain<ITransformer>, SignatureLoadModel>(env, out var transformerChain, rep, TransformerChain.LoaderSignature);
                return new CompositeDataReader<IMultiStreamSource, ITransformer>(reader, transformerChain);
            }
        }
    }
}
