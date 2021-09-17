// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
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
        /// <param name="hasHeader">Whether the file has a header with feature names. When a <see paramref="dataSample"/> is provided, <see langword="true"/>
        /// indicates that the first line in the <see paramref="dataSample"/> will be used for feature names, and that when <see cref="TextLoader.Load(IMultiStreamSource)"/>
        /// is called, the first line will be skipped. When there is no <see paramref="dataSample"/> provided, <see langword="true"/> just indicates that the loader should
        /// skip the first line when <see cref="TextLoader.Load(IMultiStreamSource)"/> is called, but columns will not have slot names annotations. This is
        /// because the output schema is made when the loader is created, and not when <see cref="TextLoader.Load(IMultiStreamSource)"/> is called.</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer slot name annotations if present, and also the number
        /// of slots in a column defined with <see cref="TextLoader.Range"/> with <see langword="null"/> maximum index.
        /// If the sample has been saved with ML.NET's <see cref="SaveAsText(DataOperationsCatalog, IDataView, Stream, char, bool, bool, bool, bool)"/>,
        /// it will also contain the schema information in the header that the loader can read even if <paramref name="columns"/> is <see langword="null"/>.
        /// In order to use the schema defined in the file, all other arguments sould be left with their default values.</param>
        /// <param name="allowQuoting">Whether the input may include double-quoted values. This parameter is used to distinguish separator characters
        /// in an input value from actual separators. When <see langword="true"/>, separators within double quotes are treated as part of the
        /// input value. When <see langword="false"/>, all separators, even those within quotes, are treated as delimiting a new column.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations. For example, a row containing
        /// "5 2:6 4:3" means that there are 5 columns, and the only non-zero are columns 2 and 4, which have values 6 and 3,
        /// respectively. Column indices are zero-based, so columns 2 and 4 represent the 3rd and 5th columns.
        /// A column may also have dense values followed by sparse values represented in this fashion. For example,
        /// a row containing "1 2 5 2:6 4:3" represents two dense columns with values 1 and 2, followed by 5 sparsely represented
        /// columns with values 0, 0, 6, 0, and 3. The indices of the sparse columns start from 0, even though 0 represents the third column.</param>
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
                AllowSparse = allowSparse,
            };

            return new TextLoader(CatalogUtils.GetEnvironment(catalog), options: options, dataSample: dataSample);
        }

        /// <summary>
        /// Create a text loader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="options">Defines the settings of the load operation.</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer slot name annotations if present, and also the number
        /// of slots in <see cref="TextLoader.Options.Columns"/> defined with <see cref="TextLoader.Range"/> with <see langword="null"/> maximum index.
        /// If the sample has been saved with ML.NET's <see cref="SaveAsText(DataOperationsCatalog, IDataView, Stream, char, bool, bool, bool, bool)"/>,
        /// it will also contain the schema information in the header that the loader can read even if <see cref="TextLoader.Options.Columns"/> are not specified.
        /// In order to use the schema defined in the file, all other <see cref="TextLoader.Options"/> sould be left with their default values.</param>
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
        /// <param name="hasHeader">Whether the file has a header with feature names. When a <see paramref="dataSample"/> is provided, <see langword="true"/>
        /// indicates that the first line in the <see paramref="dataSample"/> will be used for feature names, and that when <see cref="TextLoader.Load(IMultiStreamSource)"/>
        /// is called, the first line will be skipped. When there is no <see paramref="dataSample"/> provided, <see langword="true"/> just indicates that the loader should
        /// skip the first line when <see cref="TextLoader.Load(IMultiStreamSource)"/> is called, but columns will not have slot names annotations. This is
        /// because the output schema is made when the loader is created, and not when <see cref="TextLoader.Load(IMultiStreamSource)"/> is called.</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer slot name annotations if present.</param>
        /// <param name="allowQuoting">Whether the input may include double-quoted values. This parameter is used to distinguish separator characters
        /// in an input value from actual separators. When <see langword="true"/>, separators within double quotes are treated as part of the
        /// input value. When <see langword="false"/>, all separators, even those whitin quotes, are treated as delimiting a new column.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations. For example, a row containing
        /// "5 2:6 4:3" means that there are 5 columns, and the only non-zero are columns 2 and 4, which have values 6 and 3,
        /// respectively. Column indices are zero-based, so columns 2 and 4 represent the 3rd and 5th columns.
        /// A column may also have dense values followed by sparse values represented in this fashion. For example,
        /// a row containing "1 2 5 2:6 4:3" represents two dense columns with values 1 and 2, followed by 5 sparsely represented
        /// columns with values 0, 0, 6, 0, and 3. The indices of the sparse columns start from 0, even though 0 represents the third column.</param>
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
        /// Create a text loader <see cref="TextLoader"/> by inferencing the dataset schema from a data model type.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="options">Defines the settings of the load operation. Defines the settings of the load operation. No need to specify a Columns field,
        /// as columns will be infered by this method.</param>
        /// <param name="dataSample">The optional location of a data sample. The sample can be used to infer information
        /// about the columns, such as slot names.</param>
        public static TextLoader CreateTextLoader<TInput>(this DataOperationsCatalog catalog,
            TextLoader.Options options,
            IMultiStreamSource dataSample = null)
            => TextLoader.CreateTextLoader<TInput>(CatalogUtils.GetEnvironment(catalog), options, dataSample);

        /// <summary>
        /// Load a <see cref="IDataView"/> from a text file using <see cref="TextLoader"/>.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="path">The path to the file(s).</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="hasHeader">Whether the file has a header. When <see langword="true"/>, the loader will skip the first line when
        /// <see cref="TextLoader.Load(IMultiStreamSource)"/> is called.</param>
        /// <param name="allowQuoting">Whether the input may include double-quoted values. This parameter is used to distinguish separator characters
        /// in an input value from actual separators. When <see langword="true"/>, separators within double quotes are treated as part of the
        /// input value. When <see langword="false"/>, all separators, even those whitin quotes, are treated as delimiting a new column.
        /// It is also used to distinguish empty values from missing values. When <see langword="true"/>, missing value are denoted by consecutive
        /// separators and empty values by \"\". When <see langword="false"/>, empty values are denoted by consecutive separators and missing
        /// values by the default missing value for each type documented in <see cref="DataKind"/>.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations. For example, a row containing
        /// "5 2:6 4:3" means that there are 5 columns, and the only non-zero are columns 2 and 4, which have values 6 and 3,
        /// respectively. Column indices are zero-based, so columns 2 and 4 represent the 3rd and 5th columns.
        /// A column may also have dense values followed by sparse values represented in this fashion. For example,
        /// a row containing "1 2 5 2:6 4:3" represents two dense columns with values 1 and 2, followed by 5 sparsely represented
        /// columns with values 0, 0, 6, 0, and 3. The indices of the sparse columns start from 0, even though 0 represents the third column.</param>
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
            CheckValidPathContents(path);

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
        /// <param name="path">Specifies a file or path of files from which to load.</param>
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
            CheckValidPathContents(path);

            var env = catalog.GetEnvironment();
            var source = new MultiFileSource(path);

            return new TextLoader(env, options, dataSample: source).Load(source);
        }

        /// <summary>
        /// Load a <see cref="IDataView"/> from a text file using <see cref="TextLoader"/>.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="path">The path to the file(s).</param>
        /// <param name="separatorChar">Column separator character. Default is '\t'</param>
        /// <param name="hasHeader">Whether the file has a header. When <see langword="true"/>, the loader will skip the first line when
        /// <see cref="TextLoader.Load(IMultiStreamSource)"/> is called.</param>
        /// <param name="allowQuoting">Whether the input may include double-quoted values. This parameter is used to distinguish separator characters
        /// in an input value from actual separators. When <see langword="true"/>, separators within double quotes are treated as part of the
        /// input value. When <see langword="false"/>, all separators, even those whitin quotes, are treated as delimiting a new column.
        /// It is also used to distinguish empty values from missing values. When <see langword="true"/>, missing value are denoted by consecutive
        /// separators and empty values by \"\". When <see langword="false"/>, empty values are denoted by consecutive separators and missing
        /// values by the default missing value for each type documented in <see cref="DataKind"/>.</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations. For example, a row containing
        /// "5 2:6 4:3" means that there are 5 columns, and the only non-zero are columns 2 and 4, which have values 6 and 3,
        /// respectively. Column indices are zero-based, so columns 2 and 4 represent the 3rd and 5th columns.
        /// A column may also have dense values followed by sparse values represented in this fashion. For example,
        /// a row containing "1 2 5 2:6 4:3" represents two dense columns with values 1 and 2, followed by 5 sparsely represented
        /// columns with values 0, 0, 6, 0, and 3. The indices of the sparse columns start from 0, even though 0 represents the third column.</param>
        /// <returns>The data view.</returns>
        public static IDataView LoadFromTextFile<TInput>(this DataOperationsCatalog catalog,
            string path,
            char separatorChar = TextLoader.Defaults.Separator,
            bool hasHeader = TextLoader.Defaults.HasHeader,
            bool allowQuoting = TextLoader.Defaults.AllowQuoting,
            bool trimWhitespace = TextLoader.Defaults.TrimWhitespace,
            bool allowSparse = TextLoader.Defaults.AllowSparse)
        {
            CheckValidPathContents(path);

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
        /// <param name="path">Specifies a file or path of files from which to load.</param>
        /// <param name="options">Defines the settings of the load operation. No need to specify a Columns field,
        /// as columns will be infered by this method.</param>
        /// <returns>The data view.</returns>
        public static IDataView LoadFromTextFile<TInput>(this DataOperationsCatalog catalog, string path,
            TextLoader.Options options)
        {
            CheckValidPathContents(path);

            return TextLoader.CreateTextLoader<TInput>(CatalogUtils.GetEnvironment(catalog), options)
                .Load(new MultiFileSource(path));
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

        /// <summary>
        /// Checks the validity of a given path, and whether or not it is a
        /// valid path to a data file, or a path to a directory of files.
        /// </summary>
        /// <param name="path">Specifies a file or path of files from which to load.</param>
        private static void CheckValidPathContents(string path)
        {
            Contracts.CheckNonEmpty(path, nameof(path));
            if (!File.Exists(path) && StreamUtils.ExpandWildCards(path).Length < 1)
                throw Contracts.ExceptParam(nameof(path), "File or directory does not exist at path: {0}", path);
        }
    }
}
