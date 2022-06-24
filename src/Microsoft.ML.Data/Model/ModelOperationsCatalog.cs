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
    /// Class used by <see cref="MLContext"/> to save and load trained models.
    /// </summary>
    public sealed class ModelOperationsCatalog : IInternalCatalog
    {
        internal const string SchemaEntryName = "Schema";

        IHostEnvironment IInternalCatalog.Environment => _env;
        private readonly IHostEnvironment _env;

        internal ModelOperationsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            _env = env;
        }

        /// <summary>
        /// Save a transformer model and the loader used to create its input data to the stream.
        /// </summary>
        /// <param name="model">The trained model to be saved. Note that this can be <see langword="null"/>, as a shorthand
        /// for an empty transformer chain. Upon loading with <see cref="LoadWithDataLoader(Stream, out IDataLoader{IMultiStreamSource})"/>
        /// the returned value will be an empty <see cref="TransformerChain{TLastTransformer}"/>.</param>
        /// <param name="loader">The loader that was used to create data to train the model.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public void Save<TSource>(ITransformer model, IDataLoader<TSource> loader, Stream stream)
        {
            _env.CheckValue(loader, nameof(loader));
            _env.CheckValueOrNull(model);
            _env.CheckValue(stream, nameof(stream));

            // For the sake of consistency of this API specifically, when called upon we save any transformer
            // in a single element transformer chain.
            var chainedModel = model == null ? null : new TransformerChain<ITransformer>(model);
            var compositeLoader = new CompositeDataLoader<TSource, ITransformer>(loader, chainedModel);

            using (var rep = RepositoryWriter.CreateNew(stream, _env))
            {
                ModelSaveContext.SaveModel(rep, compositeLoader, null);
                rep.Commit();
            }
        }

        /// <summary>
        /// Save a transformer model and the loader used to create its input data to the file.
        /// </summary>
        /// <param name="model">The trained model to be saved. Note that this can be <see langword="null"/>, as a shorthand
        /// for an empty transformer chain. Upon loading with <see cref="LoadWithDataLoader(Stream, out IDataLoader{IMultiStreamSource})"/>
        /// the returned value will be an empty <see cref="TransformerChain{TLastTransformer}"/>.</param>
        /// <param name="loader">The loader that was used to create data to train the model.</param>
        /// <param name="filePath">Path where model should be saved.</param>
        public void Save<TSource>(ITransformer model, IDataLoader<TSource> loader, string filePath)
        {
            _env.CheckValueOrNull(model);
            _env.CheckValue(loader, nameof(loader));
            _env.CheckNonEmpty(filePath, nameof(filePath));

            using (var stream = File.Create(filePath))
                Save(model, loader, stream);
        }

        /// <summary>
        /// Save a transformer model and the schema of the data that was used to train it to the stream.
        /// </summary>
        /// <param name="model">The trained model to be saved. Note that this can be <see langword="null"/>, as a shorthand
        /// for an empty transformer chain. Upon loading with <see cref="Load(Stream, out DataViewSchema)"/> the returned value will
        /// be an empty <see cref="TransformerChain{TLastTransformer}"/>.</param>
        /// <param name="inputSchema">The schema of the input to the transformer. This can be <see langword="null"/>.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public void Save(ITransformer model, DataViewSchema inputSchema, Stream stream)
        {
            _env.CheckValueOrNull(model);
            _env.CheckValueOrNull(inputSchema);
            _env.CheckValue(stream, nameof(stream));

            using (var rep = RepositoryWriter.CreateNew(stream, _env))
            {
                ModelSaveContext.SaveModel(rep, model, CompositeDataLoader<object, ITransformer>.TransformerDirectory);
                SaveInputSchema(inputSchema, rep);
                rep.Commit();
            }
        }

        /// <summary>
        /// Save a transformer model and the schema of the data that was used to train it to the file.
        /// </summary>
        /// <param name="model">The trained model to be saved. Note that this can be <see langword="null"/>, as a shorthand
        /// for an empty transformer chain. Upon loading with <see cref="Load(Stream, out DataViewSchema)"/> the returned value will
        /// be an empty <see cref="TransformerChain{TLastTransformer}"/>.</param>
        /// <param name="inputSchema">The schema of the input to the transformer. This can be <see langword="null"/>.</param>
        /// <param name="filePath">Path where model should be saved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Save](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ModelOperations/SaveLoadModel.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public void Save(ITransformer model, DataViewSchema inputSchema, string filePath)
        {
            _env.CheckValueOrNull(model);
            _env.CheckValueOrNull(inputSchema);
            _env.CheckNonEmpty(filePath, nameof(filePath));

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
        /// Load the model and its input schema from a stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream to load from.</param>
        /// <param name="inputSchema">Will contain the input schema for the model. If the model was saved without
        /// any description of the input, there will be no input schema. In this case this can be <see langword="null"/>.</param>
        /// <returns>The loaded model.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Save](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ModelOperations/SaveLoadModel.cs)]
        /// ]]>
        /// </format>
        /// </example>
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
                var transformer = DecomposeLoader(ref dataLoader);
                inputSchema = dataLoader.GetOutputSchema();
                return transformer;
            }
        }

        /// <summary>
        /// Load the model and its input schema from a file.
        /// </summary>
        /// <param name="filePath">Path to a file where the model should be read from.</param>
        /// <param name="inputSchema">Will contain the input schema for the model. If the model was saved without
        /// any description of the input, there will be no input schema. In this case this can be <see langword="null"/>.</param>
        /// <returns>The loaded model.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Save](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ModelOperations/SaveLoadModelFile.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public ITransformer Load(string filePath, out DataViewSchema inputSchema)
        {
            _env.CheckNonEmpty(filePath, nameof(filePath));

            using (var stream = File.OpenRead(filePath))
                return Load(stream, out inputSchema);
        }

        /// <summary>
        /// Given a loader, test try to "decompose" it into a source loader, and its transform if any.
        /// If necessary an empty chain will be created to stand in for the trivial transformation; it
        /// should never return <see langword="null"/>.
        /// </summary>
        private ITransformer DecomposeLoader(ref IDataLoader<IMultiStreamSource> loader)
        {
            _env.AssertValue(loader);

            if (loader is CompositeDataLoader<IMultiStreamSource, ITransformer> composite)
            {
                loader = composite.Loader;
                var chain = composite.Transformer;
                // The save method corresponding to this load method encapsulates the input ITransformer
                // into a single-element transformer chain. If it is that sort, we guess that it is in fact
                // that sort, and so return it.
                var accessor = (ITransformerChainAccessor)chain;
                if (accessor.Transformers.Length == 1)
                    return accessor.Transformers[0];
                // If it is some other length than 1 due to, say, some legacy model saving, just return that
                // chain. Using the above API this is not possible, since the chain saved will always be of length
                // one, but older APIs behaved differently so we should retain flexibility with those schemes.
                // (Those schemes are BTW by no means incorrect, they just aren't what the API in this particular
                // class will specifically do.)
                return chain;
            }
            // Maybe we have no transformer stored. Rather than return null, we prefer to return the
            // empty "trivial" transformer chain.
            return new TransformerChain<ITransformer>();
        }

        /// <summary>
        /// Load a transformer model and a data loader model from a stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream to load from.</param>
        /// <param name="loader">The data loader from the model stream. Note that if there is no data loader,
        /// this method will throw an exception. The scenario where no loader is stored in the stream should
        /// be handled instead using the <see cref="Load(Stream, out DataViewSchema)"/> method.</param>
        /// <returns>The transformer model from the model stream.</returns>
        public ITransformer LoadWithDataLoader(Stream stream, out IDataLoader<IMultiStreamSource> loader)
        {
            _env.CheckValue(stream, nameof(stream));

            using (var rep = RepositoryReader.Open(stream))
            {
                try
                {
                    ModelLoadContext.LoadModel<IDataLoader<IMultiStreamSource>, SignatureLoadModel>(_env, out loader, rep, null);
                    return DecomposeLoader(ref loader);
                }
                catch (Exception ex)
                {
                    throw _env.Except(ex, "Model does not contain an " + nameof(IDataLoader<IMultiStreamSource>) +
                        ". Perhaps this was saved with an " + nameof(DataViewSchema) + ", or even no information on its input at all. " +
                        "Consider using the " + nameof(Load) + " method instead.");
                }
            }
        }

        /// <summary>
        /// Load a transformer model and a data loader model from a file.
        /// </summary>
        /// <param name="filePath">Path to a file where the model should be read from.</param>
        /// <param name="loader">The data loader from the model stream. Note that if there is no data loader,
        /// this method will throw an exception. The scenario where no loader is stored in the stream should
        /// be handled instead using the <see cref="Load(Stream, out DataViewSchema)"/> method.</param>
        /// <returns>The transformer model from the model file.</returns>
        public ITransformer LoadWithDataLoader(string filePath, out IDataLoader<IMultiStreamSource> loader)
        {
            _env.CheckNonEmpty(filePath, nameof(filePath));

            using (var stream = File.OpenRead(filePath))
                return LoadWithDataLoader(stream, out loader);
        }

        /// <summary>
        /// Create a prediction engine for one-time prediction (default usage).
        /// </summary>
        /// <typeparam name="TSrc">The class that defines the input data.</typeparam>
        /// <typeparam name="TDst">The class that defines the output data.</typeparam>
        /// <param name="transformer">The transformer to use for prediction.</param>
        /// <param name="ignoreMissingColumns">Whether to throw an exception if a column exists in
        /// <paramref name="outputSchemaDefinition"/> but the corresponding member doesn't exist in
        /// <typeparamref name="TDst"/>.</param>
        /// <param name="inputSchemaDefinition">Additional settings of the input schema.</param>
        /// <param name="outputSchemaDefinition">Additional settings of the output schema.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Save](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/ModelOperations/SaveLoadModel.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(ITransformer transformer,
            bool ignoreMissingColumns = true, SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class
            where TDst : class, new()
        {
            return transformer.CreatePredictionEngine<TSrc, TDst>(_env, ignoreMissingColumns, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// Create a prediction engine for one-time prediction.
        /// It's mainly used in conjunction with <see cref="Load(Stream, out DataViewSchema)"/>,
        /// where input schema is extracted during loading the model.
        /// </summary>
        /// <typeparam name="TSrc">The class that defines the input data.</typeparam>
        /// <typeparam name="TDst">The class that defines the output data.</typeparam>
        /// <param name="transformer">The transformer to use for prediction.</param>
        /// <param name="inputSchema">Input schema.</param>
        public PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(ITransformer transformer, DataViewSchema inputSchema)
            where TSrc : class
            where TDst : class, new()
        {
            return transformer.CreatePredictionEngine<TSrc, TDst>(_env, false,
                DataViewConstructionUtils.GetSchemaDefinition<TSrc>(_env, inputSchema));
        }

        /// <summary>
        /// Create a prediction engine for one-time prediction.
        /// It's mainly used in conjunction with <see cref="Load(Stream, out DataViewSchema)"/>,
        /// where input schema is extracted during loading the model.
        /// </summary>
        /// <typeparam name="TSrc">The class that defines the input data.</typeparam>
        /// <typeparam name="TDst">The class that defines the output data.</typeparam>
        /// <param name="transformer">The transformer to use for prediction.</param>
        /// <param name="options">Advanced configuration options.</param>
        public PredictionEngine<TSrc, TDst> CreatePredictionEngine<TSrc, TDst>(ITransformer transformer, PredictionEngineOptions options)
            where TSrc : class
            where TDst : class, new()
        {
            return transformer.CreatePredictionEngine<TSrc, TDst>(_env, options.IgnoreMissingColumns,
                options.InputSchemaDefinition, options.OutputSchemaDefinition, options.OwnsTransformer);
        }
    }
}
