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
        /// <param name="catalog">The catalog.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="dataSample">The optional location of a data sample.</param>
        public static TextLoader CreateTextReader(this DataOperations catalog,
            TextLoader.Column[] columns, bool hasHeader = false, char separatorChar = '\t', IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), columns, hasHeader, separatorChar, dataSample);

        /// <summary>
        /// Create a text reader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="args">Defines the settings of the load operation.</param>
        /// <param name="dataSample">Allows to expose items that can be used for reading.</param>
        public static TextLoader CreateTextReader(this DataOperations catalog, TextLoader.Arguments args, IMultiStreamSource dataSample = null)
            => new TextLoader(CatalogUtils.GetEnvironment(catalog), args, dataSample);

        /// <summary>
        /// Create a text reader <see cref="TextLoader"/>.
        /// </summary>
        /// <param name="catalog">The catalog.</param>
        /// <param name="hasHeader"></param>
        /// <param name="separator"></param>
        /// <param name="allowQuotedStrings"></param>
        /// <param name="supportSparse"></param>
        /// <param name="trimWhitespace"></param>
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

                if (mappingAttr == null)
                    throw Contracts.Except($"Field or property {memberInfo.Name} is missing ColumnAttribute");

                var mappingNameAttr = memberInfo.GetCustomAttribute<ColumnNameAttribute>();
                var name = mappingAttr.Name ?? mappingNameAttr?.Name ?? memberInfo.Name;

                TextLoader.Range[] sources;
                if (!TextLoader.Column.TryParseSourceEx(mappingAttr.Range, out sources))
                    throw Contracts.Except($"{mappingAttr.Range} could not be parsed.");

                Contracts.Assert(sources != null);

                var column = new TextLoader.Column();
                column.Name = name;
                column.Source = new TextLoader.Range[sources.Length];
                DataKind dk;
                switch (memberInfo)
                {
                    case FieldInfo field:
                        if (!DataKindExtensions.TryGetDataKind(field.FieldType.IsArray ? field.FieldType.GetElementType() : field.FieldType, out dk))
                            throw Contracts.Except($"Field {name} is of unsupported type.");

                        break;

                    case PropertyInfo property:
                        if (!DataKindExtensions.TryGetDataKind(property.PropertyType.IsArray ? property.PropertyType.GetElementType() : property.PropertyType, out dk))
                            throw Contracts.Except($"Property {name} is of unsupported type.");
                        break;

                    default:
                        Contracts.Assert(false);
                        throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");
                }

                column.Type = dk;

                for (int indexLocal = 0; indexLocal < column.Source.Length; indexLocal++)
                {
                    column.Source[indexLocal] = new TextLoader.Range
                    {
                        AllOther = sources[indexLocal].AllOther,
                        AutoEnd = sources[indexLocal].AutoEnd,
                        ForceVector = sources[indexLocal].ForceVector,
                        VariableEnd = sources[indexLocal].VariableEnd,
                        Max = sources[indexLocal].Max,
                        Min = sources[indexLocal].Min
                    };
                }

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
        /// <param name="catalog">The catalog.</param>
        /// <param name="columns">The columns of the schema.</param>
        /// <param name="hasHeader">Whether the file has a header.</param>
        /// <param name="separatorChar">The character used as separator between data points in a row. By default the tab character is used as separator.</param>
        /// <param name="path">The path to the file.</param>
        /// <returns>The data view.</returns>
        public static IDataView ReadFromTextFile(this DataOperations catalog,
            string path, TextLoader.Column[] columns, bool hasHeader = false, char separatorChar = '\t')
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
