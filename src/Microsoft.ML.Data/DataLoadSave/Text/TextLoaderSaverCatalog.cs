// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.StaticPipe;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using static Microsoft.ML.Runtime.Data.TextLoader;

namespace Microsoft.ML
{
    public static class TextLoaderSaverCatalog
    {
        /// <summary>
        /// Configures a reader for text files.
        /// </summary>
        /// <typeparam name="TShape">The type shape parameter, which must be a valid-schema shape. As a practical
        /// matter this is generally not explicitly defined from the user, but is instead inferred from the return
        /// type of the <paramref name="func"/> where one takes an input <see cref="Context"/> and uses it to compose
        /// a shape-type instance describing what the columns are and how to load them from the file.</typeparam>
        /// <param name="catalog">The catalog.</param>
        /// <param name="func">The delegate that describes what fields to read from the text file, as well as
        /// describing their input type. The way in which it works is that the delegate is fed a <see cref="Context"/>,
        /// and the user composes a shape type with <see cref="PipelineColumn"/> instances out of that <see cref="Context"/>.
        /// The resulting data will have columns with the names corresponding to their names in the shape type.</param>
        /// <param name="files">Input files. If <c>null</c> then no files are read, but this means that options or
        /// configurations that require input data for initialization (for example, <paramref name="hasHeader"/> or
        /// <see cref="Context.LoadFloat(int, int?)"/>) with a <c>null</c> second argument.</param>
        /// <param name="hasHeader">Data file has header with feature names.</param>
        /// <param name="separator">Text field separator.</param>
        /// <param name="allowQuoting">Whether the input -may include quoted values, which can contain separator
        /// characters, colons, and distinguish empty values from missing values. When true, consecutive separators
        /// denote a missing value and an empty value is denoted by <c>""</c>. When false, consecutive separators
        /// denote an empty value.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations.</param>
        /// <param name="trimWhitspace">Remove trailing whitespace from lines.</param>
        /// <returns>A configured statically-typed reader for text files.</returns>
        public static DataReader<IMultiStreamSource, TShape> Reader<[IsShape] TShape>(
            this DataLoadSaveOperations catalog, Func<Context, TShape> func, IMultiStreamSource files = null,
            bool hasHeader = false, char separator = '\t', bool allowQuoting = true, bool allowSparse = true,
            bool trimWhitspace = false)
         => TextLoader.CreateReader(catalog.Environment, func, files, hasHeader, separator, allowQuoting, allowSparse, trimWhitspace);

        /// <summary>
        /// Create a text reader.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="args">The arguments to text reader, describing the data schema.</param>
        /// <param name="dataSample">The optional location of a data sample.</param>
        public static TextLoader TextReader(this DataLoadSaveOperations catalog,
            TextLoader.Arguments args, IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), args, dataSample);

        /// <summary>
        /// Create a text reader.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="advancedSettings">The delegate to set additional settings.</param>
        /// <param name="dataSample">The optional location of a data sample.</param>
        public static TextLoader TextReader(this DataLoadSaveOperations catalog,
            TextLoader.Column[] columns, Action<TextLoader.Arguments> advancedSettings = null, IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), columns, advancedSettings, dataSample);

        /// <summary>
        /// Read a data view from a text file using <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="advancedSettings">The delegate to set additional settings</param>
        /// <param name="path">The path to the file</param>
        /// <returns>The data view.</returns>
        public static IDataView ReadFromTextFile(this DataLoadSaveOperations catalog,
            TextLoader.Column[] columns, string path, Action<TextLoader.Arguments> advancedSettings = null)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var env = catalog.GetEnvironment();

            // REVIEW: it is almost always a mistake to have a 'trainable' text loader here.
            // Therefore, we are going to disallow data sample.
            var reader = new TextLoader(env, columns, advancedSettings, dataSample: null);
            return reader.Read(new MultiFileSource(path));
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
        public static void SaveAsText(this DataLoadSaveOperations catalog, IDataView data, Stream stream,
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
