// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// An object serving as a 'catalog' of available model operations.
    /// </summary>
    public sealed class ModelOperationsCatalog : IInternalCatalog
    {
        private const string SchemaEntryName = "Schema";

        IHostEnvironment IInternalCatalog.Environment => _env;
        private readonly IHostEnvironment _env;

        internal ModelOperationsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            _env = env;
        }

        /// <summary>
        /// Save the model to the stream.
        /// </summary>
        /// <param name="model">The trained model to be saved.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public void Save<TSource>(IDataLoader<TSource> model, Stream stream)
        {
            _env.CheckValue(model, nameof(model));
            _env.CheckValue(stream, nameof(stream));

            using (var rep = RepositoryWriter.CreateNew(stream))
            {
                ModelSaveContext.SaveModel(rep, model, null);
                rep.Commit();
            }
        }

        /// <summary>
        /// Save the model to the file.
        /// </summary>
        /// <param name="model">The trained model to be saved.</param>
        /// <param name="filePath">Path where model should be saved.</param>
        public void Save<TSource>(IDataLoader<TSource> model, string filePath)
        {
            using (var stream = File.Create(filePath))
                Save(model, stream);
        }

        /// <summary>
        /// Save a transformer model and the loader used to create its input data to the stream.
        /// </summary>
        /// <param name="loader">The loader that was used to create data to train the model</param>
        /// <param name="model">The trained model to be saved</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public void Save<TSource>(IDataLoader<TSource> loader, ITransformer model, Stream stream) =>
            Save(new CompositeDataLoader<TSource, ITransformer>(loader, new TransformerChain<ITransformer>(model)), stream);

        /// <summary>
        /// Save a transformer model and the loader used to create its input data to the file.
        /// </summary>
        /// <param name="loader">The loader that was used to create data to train the model</param>
        /// <param name="model">The trained model to be saved</param>
        /// <param name="filePath">Path where model should be saved.</param>
        public void Save<TSource>(IDataLoader<TSource> loader, ITransformer model, string filePath)
        {
            using (var stream = File.Create(filePath))
                Save(loader, model, stream);
        }

        /// <summary>
        /// Save a transformer model and the schema of the data that was used to train it to the stream.
        /// </summary>
        /// <param name="model">The trained model to be saved.</param>
        /// <param name="inputSchema">The schema of the input to the transformer. This can be null.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public void Save(ITransformer model, DataViewSchema inputSchema, Stream stream)
        {
            _env.CheckValue(model, nameof(model));
            _env.CheckValueOrNull(inputSchema);
            _env.CheckValue(stream, nameof(stream));

            using (var rep = RepositoryWriter.CreateNew(stream))
            {
                ModelSaveContext.SaveModel(rep, model, CompositeDataLoader<object, ITransformer>.TransformerDirectory);
                SaveInputSchema(inputSchema, rep);
                rep.Commit();
            }
        }

        /// <summary>
        /// Save a transformer model and the schema of the data that was used to train it to the file.
        /// </summary>
        /// <param name="model">The trained model to be saved.</param>
        /// <param name="inputSchema">The schema of the input to the transformer. This can be null.</param>
        /// <param name="filePath">Path where model should be saved.</param>
        public void Save(ITransformer model, DataViewSchema inputSchema, string filePath)
        {
            using (var stream = File.Create(filePath))
                Save(model, inputSchema, stream);
        }

        private void SaveInputSchema(DataViewSchema inputSchema, RepositoryWriter rep)
        {
            _env.AssertValueOrNull(inputSchema);
            _env.AssertValue(rep);

            if (inputSchema == null)
                return;

            using (var ch = _env.Start("Saving Schema"))
            {
                var entry = rep.CreateEntry(SchemaEntryName);
                var saver = new BinarySaver(_env, new BinarySaver.Arguments { Silent = true });
                DataSaverUtils.SaveDataView(ch, saver, new EmptyDataView(_env, inputSchema), entry.Stream, keepHidden: true);
            }
        }

        /// <summary>
        /// Load the model and its input schema from the stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream to load from.</param>
        /// <param name="inputSchema">Will contain the input schema for the model. If the model was saved using older APIs
        /// it may not contain an input schema, in this case <paramref name="inputSchema"/> will be null.</param>
        /// <returns>The loaded model.</returns>
        public ITransformer Load(Stream stream, out DataViewSchema inputSchema)
        {
            _env.CheckValue(stream, nameof(stream));

            using (var rep = RepositoryReader.Open(stream, _env))
            {
                var entry = rep.OpenEntryOrNull(SchemaEntryName);
                if (entry != null)
                {
                    var loader = new BinaryLoader(_env, new BinaryLoader.Arguments(), entry.Stream);
                    inputSchema = loader.Schema;
                    ModelLoadContext.LoadModel<ITransformer, SignatureLoadModel>(_env, out var transformerChain, rep,
                        CompositeDataLoader<object, ITransformer>.TransformerDirectory);
                    return transformerChain;
                }

                ModelLoadContext.LoadModelOrNull<IDataLoader<IMultiStreamSource>, SignatureLoadModel>(_env, out var dataLoader, rep, null);
                if (dataLoader == null)
                {
                    // Try to see if the model was saved without a loader or a schema.
                    if (ModelLoadContext.LoadModelOrNull<ITransformer, SignatureLoadModel>(_env, out var transformerChain, rep,
                        CompositeDataLoader<object, ITransformer>.TransformerDirectory))
                    {
                        inputSchema = null;
                        return transformerChain;
                    }

                    // Try to load from legacy model format.
                    try
                    {
                        var loader = ModelFileUtils.LoadLoader(_env, rep, new MultiFileSource(null), false);
                        inputSchema = loader.Schema;
                        return TransformerChain.LoadFromLegacy(_env, stream);
                    }
                    catch (Exception ex)
                    {
                        throw _env.Except(ex, "Could not load legacy format model");
                    }
                }
                if (dataLoader is CompositeDataLoader<IMultiStreamSource, ITransformer> composite)
                {
                    inputSchema = composite.Loader.GetOutputSchema();
                    return composite.Transformer;
                }
                inputSchema = dataLoader.GetOutputSchema();
                return new TransformerChain<ITransformer>();
            }
        }

        /// <summary>
        /// Load the model and its input schema from the stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream to load from.</param>
        /// <returns>A model of type <see cref="CompositeDataLoader{IMultiStreamSource, ITransformer}"/> containing the loader
        /// and the transformer chain.</returns>
        public IDataLoader<IMultiStreamSource> Load(Stream stream)
        {
            _env.CheckValue(stream, nameof(stream));

            using (var rep = RepositoryReader.Open(stream))
            {
                try
                {
                    ModelLoadContext.LoadModel<IDataLoader<IMultiStreamSource>, SignatureLoadModel>(_env, out var model, rep, null);
                    return model;
                }
                catch (Exception ex)
                {
                    throw _env.Except(ex, "Model does not contain an IDataLoader");
                }
            }
        }

        /// <summary>
        /// Load a transformer model and a data loader model from the stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream to load from.</param>
        /// <param name="loader">The data loader from the model stream.</param>
        /// <returns>The transformer model from the model stream.</returns>
        public ITransformer LoadWithDataLoader(Stream stream, out IDataLoader<IMultiStreamSource> loader)
        {
            _env.CheckValue(stream, nameof(stream));

            loader = Load(stream);
            if (loader is CompositeDataLoader<IMultiStreamSource, ITransformer> composite)
            {
                loader = composite.Loader;
                return composite.Transformer;
            }
            return new TransformerChain<ITransformer>();
        }

        /// <summary>
        /// Create a prediction engine for one-time prediction.
        /// </summary>
        /// <typeparam name="TSrc">The class that defines the input data.</typeparam>
        /// <typeparam name="TDst">The class that defines the output data.</typeparam>
        /// <param name="transformer">The transformer to use for prediction.</param>
        /// <param name="ignoreMissingColumns">Whether to throw an exception if a column exists in
        /// <paramref name="outputSchemaDefinition"/> but the corresponding member doesn't exist in
        /// <typeparamref name="TDst"/>.</param>
        /// <param name="inputSchemaDefinition">Additional settings of the input schema.</param>
        /// <param name="outputSchemaDefinition">Additional settings of the output schema.</param>
        public PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(ITransformer transformer,
            bool ignoreMissingColumns = true, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            return transformer.CreatePredictionEngine<TSrc, TDst>(_env, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }

        public PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(ITransformer transformer, DataViewSchema inputSchema)
            where TSrc : class
            where TDst : class, new()
        {
            return transformer.CreatePredictionEngine<TSrc, TDst>(_env, false,
                DataViewConstructionUtils.GetSchemaDefinition<TSrc>(_env, inputSchema));
        }
    }
}
