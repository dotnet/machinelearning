// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    public static class TextLoaderSaverCatalog
    {
        /// <summary>
        /// Create a text loader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="columns">Array of columns <see cref="TextLoader.Column"/> defining the schema.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer column names and number of slots in each column.</param>
        /// <param name="allowQuoting">Whether the file can contain column defined by a quoted string.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        /// <param name="allowSparse">Whether the file can contain numerical vectors in sparse format.</param>
        public static TextLoader CreateTextLoader(this DataOperationsCatalog catalog,
            TextLoader.Column[] columns,
            char separatorChar = TextLoader.Defaults.Separator,
            bool hasHeader = TextLoader.Defaults.HasHeader,
            IMultiStreamSource dataSample = null,
            bool allowQuoting = TextLoader.Defaults.AllowQuoting,
            bool trimWhitespace = TextLoader.Defaults.TrimWhitespace,
            bool allowSparse = TextLoader.Defaults.AllowSparse)
        {
            var options = new TextLoader.Options
            {
                Columns = columns,
                Separators = new[] { separatorChar },
                HasHeader = hasHeader,
                AllowQuoting = allowQuoting,
                TrimWhitespace = trimWhitespace,
                AllowSparse = allowSparse
            };

            return new TextLoader(CatalogUtils.GetEnvironment(catalog), options: options, dataSample: dataSample);
        }

        /// <summary>
        /// Create a text loader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="options">Defines the settings of the load operation.</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer column names and number of slots in each column.</param>
        public static TextLoader CreateTextLoader(this DataOperationsCatalog catalog,
            TextLoader.Options options,
            IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), options, dataSample);

        /// <summary>
        /// Create a text loader <see cref="TextLoader"/> by inferencing the dataset schema from a data model type.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="separatorChar">Column separator character. Default is '\t'</param>
        /// <param name="hasHeader">Does the file contains header?</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer column names and number of slots in each column.</param>
        /// <param name="allowQuoting">Whether the input may include quoted values,
        /// which can contain separator characters, colons,
        /// and distinguish empty values from missing values. When true, consecutive separators
        /// denote a missing value and an empty value is denoted by \"\".
        /// When false, consecutive separators denote an empty value.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        /// <param name="allowSparse">Whether the input may include sparse representations for example,
        /// if one of the row contains "5 2:6 4:3" that's mean there are 5 columns all zero
        /// except for 3rd and 5th columns which have values 6 and 3</param>
        public static TextLoader CreateTextLoader<TInput>(this DataOperationsCatalog catalog,
            char separatorChar = TextLoader.Defaults.Separator,
            bool hasHeader = TextLoader.Defaults.HasHeader,
            IMultiStreamSource dataSample = null,
            bool allowQuoting = TextLoader.Defaults.AllowQuoting,
            bool trimWhitespace = TextLoader.Defaults.TrimWhitespace,
            bool allowSparse = TextLoader.Defaults.AllowSparse)
            => TextLoader.CreateTextLoader<TInput>(CatalogUtils.GetEnvironment(catalog), hasHeader, separatorChar, allowQuoting,
                allowSparse, trimWhitespace, dataSample: dataSample);

        /// <summary>
        /// Load a <see cref="IDataView"/> from a text file using <see cref="TextLoader"/>.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="path">The path to the file.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="allowQuoting">Whether the file can contain column defined by a quoted string.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        /// <param name="allowSparse">Whether the file can contain numerical vectors in sparse format.</param>
        /// <returns>The data view.</returns>
        public static IDataView LoadFromTextFile(this DataOperationsCatalog catalog,
            string path,
            TextLoader.Column[] columns,
            char separatorChar = TextLoader.Defaults.Separator,
            bool hasHeader = TextLoader.Defaults.HasHeader,
            bool allowQuoting = TextLoader.Defaults.AllowQuoting,
            bool trimWhitespace = TextLoader.Defaults.TrimWhitespace,
            bool allowSparse = TextLoader.Defaults.AllowSparse)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var options = new TextLoader.Options
            {
                Columns = columns,
                Separators = new[] { separatorChar },
                HasHeader = hasHeader,
                AllowQuoting = allowQuoting,
                TrimWhitespace = trimWhitespace,
                AllowSparse = allowSparse
            };

            var loader = new TextLoader(CatalogUtils.GetEnvironment(catalog), options: options);
            return loader.Load(new MultiFileSource(path));
        }

        /// <summary>
        /// Load a <see cref="IDataView"/> from a text file using <see cref="TextLoader"/>.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="path">The path to the file.</param>
        /// <param name="separatorChar">Column separator character. Default is '\t'</param>
        /// <param name="hasHeader">Does the file contains header?</param>
        /// <param name="allowQuoting">Whether the input may include quoted values,
        /// which can contain separator characters, colons,
        /// and distinguish empty values from missing values. When true, consecutive separators
        /// denote a missing value and an empty value is denoted by \"\".
        /// When false, consecutive separators denote an empty value.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        /// <param name="allowSparse">Whether the input may include sparse representations for example,
        /// if one of the row contains "5 2:6 4:3" that's mean there are 5 columns all zero
        /// except for 3rd and 5th columns which have values 6 and 3</param>
        /// <returns>The data view.</returns>
        public static IDataView LoadFromTextFile<TInput>(this DataOperationsCatalog catalog,
            string path,
            char separatorChar = TextLoader.Defaults.Separator,
            bool hasHeader = TextLoader.Defaults.HasHeader,
            bool allowQuoting = TextLoader.Defaults.AllowQuoting,
            bool trimWhitespace = TextLoader.Defaults.TrimWhitespace,
            bool allowSparse = TextLoader.Defaults.AllowSparse)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            // REVIEW: it is almost always a mistake to have a 'trainable' text loader here.
            // Therefore, we are going to disallow data sample.
            return TextLoader.CreateTextLoader<TInput>(CatalogUtils.GetEnvironment(catalog), hasHeader, separatorChar,
                allowQuoting, allowSparse, trimWhitespace).Load(new MultiFileSource(path));
        }

        /// <summary>
        /// Load a <see cref="IDataView"/> from a text file using <see cref="TextLoader"/>.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="path">Specifies a file from which to load.</param>
        /// <param name="options">Defines the settings of the load operation.</param>
        public static IDataView LoadFromTextFile(this DataOperationsCatalog catalog, string path,
            TextLoader.Options options = null)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var env = catalog.GetEnvironment();
            var source = new MultiFileSource(path);

            return new TextLoader(env, options, dataSample: source).Load(source);
        }

        /// <summary>
        /// Save the <see cref="IDataView"/> as text.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="data">The data view to save.</param>
        /// <param name="stream">The stream to write to.</param>
        /// <param name="separatorChar">The column separator.</param>
        /// <param name="headerRow">Whether to write the header row.</param>
        /// <param name="schema">Whether to write the header comment with the schema.</param>
        /// <param name="keepHidden">Whether to keep hidden columns in the dataset.</param>
        /// <param name="forceDense">Whether to save columns in dense format even if they are sparse vectors.</param>
        public static void SaveAsText(this DataOperationsCatalog catalog,
            IDataView data,
            Stream stream,
            char separatorChar = TextSaver.Defaults.Separator,
            bool headerRow = TextSaver.Defaults.OutputHeader,
            bool schema = TextSaver.Defaults.OutputSchema,
            bool keepHidden = false,
            bool forceDense = TextSaver.Defaults.ForceDense)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckValue(stream, nameof(stream));

            var env = catalog.GetEnvironment();
            var saver = new TextSaver(env, new TextSaver.Arguments { Dense = forceDense, Separator = separatorChar.ToString(), OutputHeader = headerRow, OutputSchema = schema });

            using (var ch = env.Start("Saving data"))
                DataSaverUtils.SaveDataView(ch, saver, data, stream, keepHidden);
        }
    }
}
