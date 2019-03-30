// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;

namespace Microsoft.ML.StaticPipe
{
    public static class ModelOperationsCatalogExtensions
    {
        /// <summary>
        /// Save statically typed model to the stream.
        /// </summary>
        /// <param name="catalog">The model explainability operations catalog.</param>
        /// <param name="model">The trained model to be saved. Note that this can be <see langword="null"/>, as a shorthand
        /// for an empty transformer chain. Upon loading with <see cref="ML.ModelOperationsCatalog.Load(Stream, out DataViewSchema)"/> the returned value will
        /// be an empty <see cref="TransformerChain{TLastTransformer}"/>.</param>
        /// <param name="dataView">The data view with the schema of the input to the transformer. This can be <see langword="null"/>.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public static void Save<TInShape, TOutShape, TTransformer>(this ML.ModelOperationsCatalog catalog, Transformer<TInShape, TOutShape, TTransformer> model, DataView<TInShape> dataView, Stream stream)
            where TTransformer : class, ITransformer
        {
            catalog.Save(model?.AsDynamic, dataView?.AsDynamic.Schema, stream);
        }

        /// <summary>
        /// Save statically typed model to the stream.
        /// </summary>
        /// <param name="catalog">The model explainability operations catalog.</param>
        /// <param name="model">The trained model to be saved. Note that this can be <see langword="null"/>, as a shorthand
        /// for an empty transformer chain. Upon loading with <see cref="ML.ModelOperationsCatalog.Load(Stream, out DataViewSchema)"/> the returned value will
        /// be an empty <see cref="TransformerChain{TLastTransformer}"/>.</param>
        /// <param name="dataView">The data view with the schema of the input to the transformer. This can be <see langword="null"/>.</param>
        /// <param name="filePath">Path where model should be saved.</param>
        public static void Save<TInShape, TOutShape, TTransformer>(this ML.ModelOperationsCatalog catalog, Transformer<TInShape, TOutShape, TTransformer> model, DataView<TInShape> dataView, string filePath)
            where TTransformer : class, ITransformer
        {
            catalog.Save(model?.AsDynamic, dataView?.AsDynamic.Schema, filePath);
        }
    }
}
