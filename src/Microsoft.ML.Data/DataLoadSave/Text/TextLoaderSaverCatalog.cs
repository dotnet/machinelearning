// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for the <see cref="DataOperationsCatalog"/> to read from delimited text
    /// files such as csv and tsv.
    /// </summary>
    public static class TextLoaderSaverCatalog
    {
        /// <summary>
        /// Create a text loader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="columns">Array of columns <see cref="TextLoader.Column"/> defining the schema.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="hasHeader">Whether the data file has a header. If true, it will cause the header to be skipped,
        /// but will not help with automatically detecting column names or types, which must be specified with <paramref name="columns"/> or in <paramref name="dataSample"/>.</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer column names and number of slots in each column.
        /// The sample must be a text file previously saved with <see cref="SaveAsText(DataOperationsCatalog, IDataView, Stream, char, bool, bool, bool, bool)"/>,
        /// so that it contains the schema information in the header that the loader can use to infer columns.</param>
        /// <param name="allowQuoting">Whether the input may include double-quoted values. This parameter is used to distinguish separator characters
        /// in an input value from actual separators. When true, separators within double quotes are treated as part of the
        /// input value. When false, all separators, even those whitin quotes, are treated as delimiting a new column.
        /// It is also used to distinguish empty values from missing values. When true, missing value are denoted by consecutive
        /// separators and empty values by \"\". When false, empty values are denoted by consecutive separators and missing
        /// values by the default missing value for each type documented in <see cref="DataKind"/>.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations. For example, a row containing
        /// "5 2:6 4:3" means that there are 5 columns, and the only non-zero are columns 2 and 4, which have values 6 and 3,
        /// respectively. Column indices are zero-based, so columns 2 and 4 represent the 3rd and 5th columns.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CreateTextLoader](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/LoadingText.cs)]
        /// ]]>
        /// </format>
        /// </example>
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
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer column names and number of slots in each column.
        /// The sample must be a text file previously saved with <see cref="SaveAsText(DataOperationsCatalog, IDataView, Stream, char, bool, bool, bool, bool)"/>,
        /// so that it contains the schema information in the header that the loader can use to infer columns.</param>
        public static TextLoader CreateTextLoader(this DataOperationsCatalog catalog,
            TextLoader.Options options,
            IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), options, dataSample);

        /// <summary>
        /// Create a text loader <see cref="TextLoader"/> by inferencing the dataset schema from a data model type.
        /// </summary>
        /// <typeparam name="TInput">Defines the schema of the data to be loaded. Use public fields or properties
        /// decorated with <see cref="LoadColumnAttribute"/> (and possibly other attributes) to specify the column
        /// names and their data types in the schema of the loaded data.</typeparam>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="separatorChar">Column separator character. Default is '\t'</param>
        /// <param name="hasHeader">Whether the data file has a header. If true, it will cause the header to be skipped,
        /// but will not help with automatically detecting column names or types, which must be specified in the input
        /// type <typeparamref name="TInput"/>.</param>
        /// <param name="dataSample">The optional location of a data sample. Since <typeparamref name="TInput"/> defines
        /// the schema of the data to be loaded, the data sample is ignored.</param>
        /// <param name="allowQuoting">Whether the input may include double-quoted values. This parameter is used to distinguish separator characters
        /// in an input value from actual separators. When true, separators within double quotes are treated as part of the
        /// input value. When false, all separators, even those whitin quotes, are treated as delimiting a new column.
        /// It is also used to distinguish empty values from missing values. When true, missing value are denoted by consecutive
        /// separators and empty values by \"\". When false, empty values are denoted by consecutive separators and missing
        /// values by the default missing value for each type documented in <see cref="DataKind"/>.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations. For example, a row containing
        /// "5 2:6 4:3" means that there are 5 columns, and the only non-zero are columns 2 and 4, which have values 6 and 3,
        /// respectively. Column indices are zero-based, so columns 2 and 4 represent the 3rd and 5th columns.</param>
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
        /// <param name="hasHeader">Whether the data file has a header. If true, it will cause the header to be skipped,
        /// but will not help with automatically detecting column names or types, which must be specified with <paramref name="columns"/>.</param>
        /// <param name="allowQuoting">Whether the input may include double-quoted values. This parameter is used to distinguish separator characters
        /// in an input value from actual separators. When true, separators within double quotes are treated as part of the
        /// input value. When false, all separators, even those whitin quotes, are treated as delimiting a new column.
        /// It is also used to distinguish empty values from missing values. When true, missing value are denoted by consecutive
        /// separators and empty values by \"\". When false, empty values are denoted by consecutive separators and missing
        /// values by the default missing value for each type documented in <see cref="DataKind"/>.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines.</param>
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
            if (!File.Exists(path))
            {
                throw Contracts.ExceptParam(nameof(path), "File does not exist at path: {0}", path);
            }

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
        /// <param name="hasHeader">Whether the data file has a header. If true, it will cause the header to be skipped,
        /// but will not help with automatically detecting column names or types, which must be specified in the input
        /// type <typeparamref name="TInput"/>.</param>
        /// <param name="allowQuoting">Whether the input may include double-quoted values. This parameter is used to distinguish separator characters
        /// in an input value from actual separators. When true, separators within double quotes are treated as part of the
        /// input value. When false, all separators, even those whitin quotes, are treated as delimiting a new column.
        /// It is also used to distinguish empty values from missing values. When true, missing value are denoted by consecutive
        /// separators and empty values by \"\". When false, empty values are denoted by consecutive separators and missing
        /// values by the default missing value for each type documented in <see cref="DataKind"/>.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations. For example, a row containing
        /// "5 2:6 4:3" means that there are 5 columns, and the only non-zero are columns 2 and 4, which have values 6 and 3,
        /// respectively. Column indices are zero-based, so columns 2 and 4 represent the 3rd and 5th columns.</param>
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
            if (!File.Exists(path))
            {
                throw Contracts.ExceptParam(nameof(path), "File does not exist at path: {0}", path);
            }

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
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LoadFromTextFile](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/SaveAndLoadFromText.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static IDataView LoadFromTextFile(this DataOperationsCatalog catalog, string path,
            TextLoader.Options options = null)
        {
            Contracts.CheckNonEmpty(path, nameof(path));
            if (!File.Exists(path))
            {
                throw Contracts.ExceptParam(nameof(path), "File does not exist at path: {0}", path);
            }

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
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SaveAsText](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/SaveAndLoadFromText.cs)]
        /// ]]>
        /// </format>
        /// </example>
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
