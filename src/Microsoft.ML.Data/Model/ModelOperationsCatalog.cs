// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.Data.DataView;
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

        public ExplainabilityTransforms Explainability { get; }

        internal ModelOperationsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            _env = env;

            Explainability = new ExplainabilityTransforms(this);
        }

        /// <summary>
        /// Save the model to the stream.
        /// </summary>
        /// <param name="model">The trained model to be saved.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public void Save<TSource>(IDataLoader<TSource> model, Stream stream)
        {
            using (var rep = RepositoryWriter.CreateNew(stream))
            {
                ModelSaveContext.SaveModel(rep, model, null);
                SaveInputSchema(model.GetOutputSchema(), rep);
                rep.Commit();
            }
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
        /// Save a transformer model and the schema of the data that was used to train it to the stream.
        /// </summary>
        /// <param name="inputSchema">The schema of the input to the transformer.</param>
        /// <param name="model">The trained model to be saved.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public void Save(DataViewSchema inputSchema, ITransformer model, Stream stream)
        {
            using (var rep = RepositoryWriter.CreateNew(stream))
            {
                ModelSaveContext.SaveModel(rep, model, TransformerDirectory);
                SaveInputSchema(inputSchema, rep);
                rep.Commit();
            }
        }

        private void SaveInputSchema(DataViewSchema inputSchema, RepositoryWriter rep)
        {
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
            using (var rep = RepositoryReader.Open(stream, _env))
            {
                var entry = rep.OpenEntryOrNull(SchemaEntryName);
                if (entry != null)
                {
                    var loader = new BinaryLoader(_env, new BinaryLoader.Arguments(), entry.Stream);
                    inputSchema = loader.Schema;
                }
                else
                {
                    // Try to load from legacy model format.
                    try
                    {
                        var loader = ModelFileUtils.LoadLoader(_env, rep, new MultiFileSource(null), false);
                        inputSchema = loader.Schema;
                    }
                    catch (Exception ex)
                    {
                        if (!ex.IsMarked())
                            throw;
                        inputSchema = null;
                    }
                }
                return TransformerChain.LoadFrom(_env, stream);
            }
        }

        /// <summary>
        /// Load the model and its input schema from the stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream to load from.</param>
        /// <returns>A model of type <see cref="CompositeDataLoader{IMultiStreamSource, ITransformer}"/> containing the loader
        /// and the transformer chain.</returns>
        public CompositeDataLoader<IMultiStreamSource, ITransformer> Load(Stream stream)
        {
            using (var rep = RepositoryReader.Open(stream))
            {
                ModelLoadContext.LoadModel<CompositeDataLoader<IMultiStreamSource, ITransformer>, SignatureLoadModel>(_env, out var model, rep, null);
                return model;
            }
        }

        /// <summary>
        /// Load a transformer model and a data loader model from the stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream to load from.</param>
        /// <param name="loader">The data loader from the model stream.</param>
        /// <returns>The transformer model from the model stream.</returns>
        public ITransformer Load(Stream stream, out IDataLoader<IMultiStreamSource> loader)
        {
            loader = Load(stream);
            if (loader is CompositeDataLoader<IMultiStreamSource, ITransformer> composite)
            {
                loader = composite.Loader;
                return composite.Transformer;
            }
            return new TransformerChain<ITransformer>();
        }

        /// <summary>
        /// Load the model from a file path.
        /// </summary>
        /// <param name="modelPath">Path to model.</param>
        /// <returns>The loaded model.</returns>
        public ITransformer Load(string modelPath)
        {
            using (var stream = File.OpenRead(modelPath))
                return Load(stream);
        }

        /// <summary>
        /// The catalog of model explainability operations.
        /// </summary>
        public sealed class ExplainabilityTransforms : IInternalCatalog
        {
            IHostEnvironment IInternalCatalog.Environment => _env;
            private readonly IHostEnvironment _env;

            internal ExplainabilityTransforms(ModelOperationsCatalog owner)
            {
                _env = owner._env;
            }
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
