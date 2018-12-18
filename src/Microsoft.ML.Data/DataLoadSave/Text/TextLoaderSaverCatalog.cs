// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;

namespace Microsoft.ML
{
    public static class TextLoaderSaverCatalog
    {
        /// <summary>
        /// Create a text reader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperations"/> catalog.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="dataSample">The optional location of a data sample.</param>
        public static TextLoader CreateTextReader(this DataOperations catalog,
            TextLoader.Column[] columns,
            bool hasHeader = false,
            char separatorChar = '\t',
            IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), columns, hasHeader, separatorChar, dataSample);

        /// <summary>
        /// Create a text reader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperations"/> catalog.</param>
        /// <param name="args">Defines the settings of the load operation.</param>
        /// <param name="dataSample">Allows to expose items that can be used for reading.</param>
        public static TextLoader CreateTextReader(this DataOperations catalog,
            TextLoader.Arguments args,
            IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), args, dataSample);

        /// <summary>
        /// Create a text reader <see cref="TextLoader"/> by inferencing the dataset schema from a data model type.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperations"/> catalog.</param>
        /// <param name="hasHeader">Does the file contains header?</param>
        /// <param name="separator">Column separator character. Default is '\t'</param>
        /// <param name="allowQuotedStrings">Whether the input may include quoted values,
        /// which can contain separator characters, colons,
        /// and distinguish empty values from missing values. When true, consecutive separators
        /// denote a missing value and an empty value is denoted by \"\".
        /// When false, consecutive separators denote an empty value.</param>
        /// <param name="supportSparse">Whether the input may include sparse representations for example,
        /// if one of the row contains "5 2:6 4:3" that's mean there are 5 columns all zero
        /// except for 3rd and 5th columns which have values 6 and 3</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        public static TextLoader CreateTextReader<TInput>(this DataOperations catalog,
            bool hasHeader = TextLoader.DefaultArguments.HasHeader,
            char separator = TextLoader.DefaultArguments.Separator,
            bool allowQuotedStrings = TextLoader.DefaultArguments.AllowQuoting,
            bool supportSparse = TextLoader.DefaultArguments.AllowSparse,
            bool trimWhitespace = TextLoader.DefaultArguments.TrimWhitespace)
        {
            var userType = typeof(TInput);

            var fieldInfos = userType.GetFields(BindingFlags.Public | BindingFlags.Instance);

            var propertyInfos =
                userType
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(x => x.CanRead && x.CanWrite && x.GetGetMethod() != null && x.GetSetMethod() != null && x.GetIndexParameters().Length == 0);

            var memberInfos = (fieldInfos as IEnumerable<MemberInfo>).Concat(propertyInfos).ToArray();

            var columns = new TextLoader.Column[memberInfos.Length];

            for (int index = 0; index < memberInfos.Length; index++)
            {
                var memberInfo = memberInfos[index];
                var mappingAttr = memberInfo.GetCustomAttribute<LoadColumnAttribute>();
                var mptr = memberInfo.GetCustomAttributes();

                Contracts.Assert(mappingAttr != null, $"Field or property {memberInfo.Name} is missing the LoadColumn attribute");

                var column = new TextLoader.Column();
                column.Name = mappingAttr.Name ?? memberInfo.Name;
                column.Source = mappingAttr.Sources.ToArray();
                DataKind dk;
                switch (memberInfo)
                {
                    case FieldInfo field:
                        if (!DataKindExtensions.TryGetDataKind(field.FieldType.IsArray ? field.FieldType.GetElementType() : field.FieldType, out dk))
                            throw Contracts.Except($"Field {memberInfo.Name} is of unsupported type.");

                        break;

                    case PropertyInfo property:
                        if (!DataKindExtensions.TryGetDataKind(property.PropertyType.IsArray ? property.PropertyType.GetElementType() : property.PropertyType, out dk))
                            throw Contracts.Except($"Property {memberInfo.Name} is of unsupported type.");
                        break;

                    default:
                        Contracts.Assert(false);
                        throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");
                }

                column.Type = dk;

                columns[index] = column;
            }

            TextLoader.Arguments args = new TextLoader.Arguments
            {
                HasHeader = hasHeader,
                SeparatorChars = new[] { separator },
                AllowQuoting = allowQuotedStrings,
                AllowSparse = supportSparse,
                TrimWhitespace = trimWhitespace,
                Column = columns
            };

            return new TextLoader(CatalogUtils.GetEnvironment(catalog), args);
        }

        /// <summary>
        /// Read a data view from a text file using <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperations"/> catalog.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="path">The path to the file.</param>
        /// <returns>The data view.</returns>
        public static IDataView ReadFromTextFile(this DataOperations catalog,
            string path,
            TextLoader.Column[] columns,
            bool hasHeader = false,
            char separatorChar = '\t')
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
        /// <param name="catalog">The <see cref="DataOperations"/> catalog.</param>
        /// <param name="path">Specifies a file from which to read.</param>
        /// <param name="args">Defines the settings of the load operation.</param>
        public static IDataView ReadFromTextFile(this DataOperations catalog, string path, TextLoader.Arguments args = null)
        {
            Contracts.CheckNonEmpty(path, nameof(path));

            var env = catalog.GetEnvironment();
            var source = new MultiFileSource(path);

            return new TextLoader(env, args, source).Read(source);
        }

        /// <summary>
        /// Save the data view as text.
        /// </summary>
        /// <param name="catalog">The <see cref="DataOperations"/> catalog.</param>
        /// <param name="data">The data view to save.</param>
        /// <param name="stream">The stream to write to.</param>
        /// <param name="separator">The column separator.</param>
        /// <param name="headerRow">Whether to write the header row.</param>
        /// <param name="schema">Whether to write the header comment with the schema.</param>
        /// <param name="keepHidden">Whether to keep hidden columns in the dataset.</param>
        public static void SaveAsText(this DataOperations catalog,
            IDataView data,
            Stream stream,
            char separator = '\t',
            bool headerRow = true,
            bool schema = true,
            bool keepHidden = false)
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
