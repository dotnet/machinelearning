// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Data
{
    public sealed partial class TextLoaderRange
    {
        [JsonIgnore]
        public int Ordinal { get { return Ordinal; } set { Min = value; Max = value; } }
    }

    public sealed partial class TextLoader
    {

        /// <summary>
        /// Construct a TextLoader object by inferencing the dataset schema from a type.
        /// </summary>
        /// <param name="useHeader">Does the file contains header?</param>
        /// <param name="separator">Column separator character. Default is '\t'</param>
        /// <param name="allowQuotedStrings">Whether the input may include quoted values, 
        /// which can contain separator characters, colons,
        /// and distinguish empty values from missing values. When true, consecutive separators 
        /// denote a missing value and an empty value is denoted by \"\". 
        /// When false, consecutive separators denote an empty value.</param>
        /// <param name="supportSparse">Whether the input may include sparse representations e.g. 
        /// if one of the row contains "5 2:6 4:3" that's mean there are 5 columns all zero 
        /// except for 3rd and 5th columns which have values 6 and 3</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        public TextLoader CreateFrom<TInput>(bool useHeader = false,
            char separator = '\t', bool allowQuotedStrings = true,
            bool supportSparse = true, bool trimWhitespace = false)
        {
            var fields = typeof(TInput).GetFields();
            Arguments.Column = new TextLoaderColumn[fields.Length];
            for (int index = 0; index < fields.Length; index++)
            {
                var field = fields[index];
                var mappingAttr = field.GetCustomAttribute<ColumnAttribute>();
                if (mappingAttr == null)
                    throw Contracts.Except($"{field.Name} is missing ColumnAttribute");

                if (Regex.Match(mappingAttr.Ordinal, @"[^(0-9,\*\-~)]+").Success)
                    throw Contracts.Except($"{mappingAttr.Ordinal} contains invalid characters. " +
                        $"Valid characters are 0-9, *, - and ~");

                var name = mappingAttr.Name ?? field.Name;
                if (name.Any(c => !Char.IsLetterOrDigit(c)))
                    throw Contracts.Except($"{name} is not alphanumeric.");

                DataKind dk;
                Utils.TryGetDataKind(field.FieldType.IsArray ? field.FieldType.GetElementType() : field.FieldType, out dk);
                Runtime.Data.TextLoader.Range[] sources;
                if (!Runtime.Data.TextLoader.Column.TryParseSourceEx(mappingAttr.Ordinal, out sources))
                    throw Contracts.Except($"{mappingAttr.Ordinal} could not be parsed.");

                Contracts.Assert(sources != null);

                TextLoaderColumn tlc = new TextLoaderColumn();
                tlc.Name = name;
                tlc.Source = new TextLoaderRange[sources.Length];
                for (int indexLocal = 0; indexLocal < tlc.Source.Length; indexLocal++)
                {
                    tlc.Source[indexLocal] = new TextLoaderRange
                    {
                        AllOther = sources[indexLocal].AllOther,
                        AutoEnd = sources[indexLocal].AutoEnd,
                        ForceVector = sources[indexLocal].ForceVector,
                        VariableEnd = sources[indexLocal].VariableEnd,
                        Max = sources[indexLocal].Max,
                        Min = sources[indexLocal].Min
                    };
                }

                tlc.Type = dk;
                Arguments.Column[index] = tlc;
            }

            Arguments.HasHeader = useHeader;
            Arguments.Separator = new[] { separator };
            Arguments.AllowQuoting = allowQuotedStrings;
            Arguments.AllowSparse = supportSparse;
            Arguments.TrimWhitespace = trimWhitespace;

            return this;
        }
    }
}
