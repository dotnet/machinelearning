// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;

namespace Microsoft.ML
{
    public static class TextLoaderSaverCatalog
    {
        /// <summary>
        /// Create a text loader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="columns">Array of columns <see cref="TextLoader.Column"/> defining the schema.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer column names and number of slots in each column.</param>
        public static TextLoader CreateTextLoader(this DataOperationsCatalog catalog,
            TextLoader.Column[] columns,
            bool hasHeader = TextLoader.DefaultArguments.HasHeader,
            char separatorChar = TextLoader.DefaultArguments.Separator,
            IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), columns, hasHeader, separatorChar, dataSample);

        /// <summary>
        /// Create a text loader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="args">Defines the settings of the load operation.</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer column names and number of slots in each column.</param>
        public static TextLoader CreateTextLoader(this DataOperationsCatalog catalog,
            TextLoader.Arguments args,
            IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), args, dataSample);

        /// <summary>
        /// Create a text loader <see cref="TextLoader"/> by inferencing the dataset schema from a data model type.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="hasHeader">Does the file contains header?</param>
        /// <param name="separatorChar">Column separator character. Default is '\t'</param>
        /// <param name="allowQuotedStrings">Whether the input may include quoted values,
        /// which can contain separator characters, colons,
        /// and distinguish empty values from missing values. When true, consecutive separators
        /// denote a missing value and an empty value is denoted by \"\".
        /// When false, consecutive separators denote an empty value.</param>
        /// <param name="supportSparse">Whether the input may include sparse representations for example,
        /// if one of the row contains "5 2:6 4:3" that's mean there are 5 columns all zero
        /// except for 3rd and 5th columns which have values 6 and 3</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        public static TextLoader CreateTextLoader<TInput>(this DataOperationsCatalog catalog,
            bool hasHeader = TextLoader.DefaultArguments.HasHeader,
            char separatorChar = TextLoader.DefaultArguments.Separator,
            bool allowQuotedStrings = TextLoader.DefaultArguments.AllowQuoting,
            bool supportSparse = TextLoader.DefaultArguments.AllowSparse,
            bool trimWhitespace = TextLoader.DefaultArguments.TrimWhitespace)
            => TextLoader.CreateTextReader<TInput>(CatalogUtils.GetEnvironment(catalog), hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace);

        /// <summary>
        /// Read a data view from a text file using <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="path">The path to the file.</param>
        /// <returns>The data view.</returns>
        public static IDataView ReadFromTextFile(this DataOperationsCatalog catalog,
            string path,
            TextLoader.Column[] columns,
            bool hasHeader = TextLoader.DefaultArguments.HasHeader,
            char separatorChar = TextLoader.DefaultArguments.Separator)
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
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="hasHeader">Does the file contains header?</param>
        /// <param name="separatorChar">Column separator character. Default is '\t'</param>
        /// <param name="allowQuotedStrings">Whether the input may include quoted values,
        /// which can contain separator characters, colons,
        /// and distinguish empty values from missing values. When true, consecutive separators
        /// denote a missing value and an empty value is denoted by \"\".
        /// When false, consecutive separators denote an empty value.</param>
        /// <param name="supportSparse">Whether the input may include sparse representations for example,
        /// if one of the row contains "5 2:6 4:3" that's mean there are 5 columns all zero
        /// except for 3rd and 5th columns which have values 6 and 3</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        /// <param name="path">The path to the file.</param>
        /// <returns>The data view.</returns>
        public static IDataView ReadFromTextFile<TInput>(this DataOperationsCatalog catalog,
            string path,
            bool hasHeader = TextLoader.DefaultArguments.HasHeader,
            char separatorChar = TextLoader.DefaultArguments.Separator,
            bool allowQuotedStrings = TextLoader.DefaultArguments.AllowQuoting,
            bool supportSparse = TextLoader.DefaultArguments.AllowSparse,
            bool trimWhitespace = TextLoader.DefaultArguments.TrimWhitespace)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            // REVIEW: it is almost always a mistake to have a 'trainable' text loader here.
            // Therefore, we are going to disallow data sample.
            return TextLoader.CreateTextReader<TInput>(CatalogUtils.GetEnvironment(catalog), hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace)
                             .Read(new MultiFileSource(path));
        }

        /// <summary>
        /// Read a data view from a text file using <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="path">Specifies a file from which to read.</param>
        /// <param name="args">Defines the settings of the load operation.</param>
        public static IDataView ReadFromTextFile(this DataOperationsCatalog catalog, string path, TextLoader.Arguments args = null)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var env = catalog.GetEnvironment();
            var source = new MultiFileSource(path);

            return new TextLoader(env, args, source).Read(source);
        }

        /// <summary>
        /// Save the data view as text.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="data">The data view to save.</param>
        /// <param name="stream">The stream to write to.</param>
        /// <param name="separatorChar">The column separator.</param>
        /// <param name="headerRow">Whether to write the header row.</param>
        /// <param name="schema">Whether to write the header comment with the schema.</param>
        /// <param name="keepHidden">Whether to keep hidden columns in the dataset.</param>
        public static void SaveAsText(this DataOperationsCatalog catalog,
            IDataView data,
            Stream stream,
            char separatorChar = TextLoader.DefaultArguments.Separator,
            bool headerRow = TextLoader.DefaultArguments.HasHeader,
            bool schema = true,
            bool keepHidden = false)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckValue(stream, nameof(stream));

            var env = catalog.GetEnvironment();
            var saver = new TextSaver(env, new TextSaver.Arguments { Separator = separatorChar.ToString(), OutputHeader = headerRow, OutputSchema = schema });

            using (var ch = env.Start("Saving data"))
                DataSaverUtils.SaveDataView(ch, saver, data, stream, keepHidden);
        }
    }
}
