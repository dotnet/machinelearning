// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;

namespace Microsoft.ML.StaticPipe
{
    public static class ModelOperationsCatalog
    {
        /// <summary>
        /// Save statically typed model to the stream.
        /// </summary>
        /// <param name="catalog">The model explainability operations catalog.</param>
        /// <param name="model">The trained model to be saved.</param>
        /// <param name="stream">A writeable, seekable stream to save to.</param>
        public static void Save<TInShape, TOutShape, TTransformer>(this ML.ModelOperationsCatalog catalog, Transformer<TInShape, TOutShape, TTransformer> model, Stream stream)
            where TTransformer : class, ITransformer
        {
            catalog.Save(model.AsDynamic, stream);
        }
    }
}
