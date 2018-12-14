// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using System;
using System.IO;
using static Microsoft.ML.Runtime.Data.TextLoader;

namespace Microsoft.ML
{
    public static class TextLoaderSaverCatalog
    {
        /// <summary>
        /// Create a text reader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="dataSample">The optional location of a data sample.</param>
        public static TextLoader CreateTextReader(this DataOperations catalog,
            Column[] columns, bool hasHeader = false, char separatorChar = '\t', IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), columns, hasHeader, separatorChar, dataSample);

        /// <summary>
        /// Create a text reader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="args">Defines the settings of the load operation.</param>
        /// <param name="dataSample">Allows to expose items that can be used for reading.</param>
        public static TextLoader CreateTextReader(this DataOperations catalog, Arguments args, IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), args, dataSample);

        /// <summary>
        /// Read a data view from a text file using <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="path">The path to the file.</param>
        /// <returns>The data view.</returns>
        public static IDataView ReadFromTextFile(this DataOperations catalog,
            string path, Column[] columns, bool hasHeader = false, char separatorChar = '\t')
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var env = catalog.GetEnvironment();

            // REVIEW: it is almost always a mistake to have a 'trainable' text loader here.
            // Therefore, we are going to disallow data sample.
            var reader = new TextLoader(env, columns, hasHeader, separatorChar, dataSample: null);
            return reader.Read(new MultiFileSource(path));
        }

        /// <summary>
        /// Read a data view from a text file using <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="path">Specifies a file from which to read.</param>
        /// <param name="args">Defines the settings of the load operation.</param>
        public static IDataView ReadFromTextFile(this DataOperations catalog, string path, Arguments args = null)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var env = catalog.GetEnvironment();
            var source = new MultiFileSource(path);

            return new TextLoader(env, args, source).Read(source);
        }

        /// <summary>
        /// Save the data view as text.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="data">The data view to save.</param>
        /// <param name="stream">The stream to write to.</param>
        /// <param name="separator">The column separator.</param>
        /// <param name="headerRow">Whether to write the header row.</param>
        /// <param name="schema">Whether to write the header comment with the schema.</param>
        /// <param name="keepHidden">Whether to keep hidden columns in the dataset.</param>
        public static void SaveAsText(this DataOperations catalog, IDataView data, Stream stream,
            char separator = '\t', bool headerRow = true, bool schema = true, bool keepHidden = false)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckValue(stream, nameof(stream));

            var env = catalog.GetEnvironment();
            var saver = new TextSaver(env, new TextSaver.Arguments { Separator = separator.ToString(), OutputHeader = headerRow, OutputSchema = schema });

            using (var ch = env.Start("Saving data"))
                DataSaverUtils.SaveDataView(ch, saver, data, stream, keepHidden);
        }
    }
}
