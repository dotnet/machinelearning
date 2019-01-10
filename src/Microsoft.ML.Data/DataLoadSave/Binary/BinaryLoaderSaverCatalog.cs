// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;

namespace Microsoft.ML
{
    public static class BinaryLoaderSaverCatalog
    {
        /// <summary>
        /// Read a data view from an <see cref="IMultiStreamSource"/> on a binary file using <see cref="BinaryLoader"/>.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="fileSource">The file source to read from. This can be a <see cref="MultiFileSource"/>, for example.</param>
        public static IDataView ReadFromBinary(this DataOperations catalog, IMultiStreamSource fileSource)
        {
            Contracts.CheckValue(fileSource, nameof(fileSource));

            var env = catalog.GetEnvironment();

            var reader = new BinaryLoader(env, new BinaryLoader.Arguments(), fileSource);
            return reader;
        }

        /// <summary>
        /// Read a data view from a binary file using <see cref="BinaryLoader"/>.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="path">The path to the file to read from.</param>
        public static IDataView ReadFromBinary(this DataOperations catalog, string path)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var env = catalog.GetEnvironment();

            var reader = new BinaryLoader(env, new BinaryLoader.Arguments(), path);
            return reader;
        }

        /// <summary>
        /// Save the data view into a binary stream.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="data">The data view to save.</param>
        /// <param name="stream">The stream to write to.</param>
        /// <param name="keepHidden">Whether to keep hidden columns in the dataset.</param>
        public static void SaveAsBinary(this DataOperations catalog, IDataView data, Stream stream,
            bool keepHidden = false)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckValue(stream, nameof(stream));

            var env = catalog.GetEnvironment();
            var saver = new BinarySaver(env, new BinarySaver.Arguments());

            using (var ch = env.Start("Saving data"))
                DataSaverUtils.SaveDataView(ch, saver, data, stream, keepHidden);
        }
    }
}
