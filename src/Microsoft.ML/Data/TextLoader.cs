// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Microsoft.ML.Data
{
    public sealed class TextLoader<TInput> : TextLoader
    {

        /// <summary>
        /// Construct a TextLoader object
        /// </summary>
        /// <param name="inputFilePath">Data file path</param>
        /// <param name="useHeader">Does the file contains header?</param>
        /// <param name="separator">How the columns are seperated? 
        /// Options: separator="tab", separator="space", separator="comma" or separator=[single character]. 
        /// By default separator=null means "tab"</param>
        /// <param name="allowQuotedStrings">Whether the input may include quoted values, 
        /// which can contain separator characters, colons,
        /// and distinguish empty values from missing values. When true, consecutive separators 
        /// denote a missing value and an empty value is denoted by \"\". 
        /// When false, consecutive separators denote an empty value.</param>
        /// <param name="supportSparse">Whether the input may include sparse representations e.g. 
        /// if one of the row contains "5 2:6 4:3" that's mean there are 5 columns all zero 
        /// except for 3rd and 5th columns which have values 6 and 3</param>
        /// <param name="trimWhitespace">Remove trailing whitespace from lines</param>
        public TextLoader(string inputFilePath, bool useHeader = false,
            string separator = null, bool allowQuotedStrings = true,
            bool supportSparse = true, bool trimWhitespace = false) : base(inputFilePath)
        {
            var fields = typeof(TInput).GetFields();
            Arguments.Column = new TextLoaderColumn[fields.Length];
            for (int index = 0; index < fields.Length; index++)
            {
                var field = fields[index];
                var mappingAttr = field.GetCustomAttribute<ColumnAttribute>();
                if (mappingAttr == null)
                    throw Contracts.ExceptParam(nameof(field.Name), " is missing ColumnAttribute");

                var col = Runtime.Data.TextLoader.Column.Parse(
                    $"{mappingAttr.Name ?? field.Name}:" +
                    $"{TypeToName(field.FieldType.IsArray ? field.FieldType.GetElementType() : field.FieldType)}:" +
                    $"{mappingAttr.Ordinal}"
                    );

                TextLoaderColumn tlc = new TextLoaderColumn();
                if (col.KeyRange != null)
                {
                    tlc.KeyRange = new KeyRange();
                    tlc.KeyRange.Min = col.KeyRange.Min;
                    tlc.KeyRange.Max = col.KeyRange.Max;
                }

                tlc.Name = col.Name;
                tlc.Source = new TextLoaderRange[col.Source.Length];
                for (int indexLocal = 0; indexLocal < tlc.Source.Length; indexLocal++)
                {
                    tlc.Source[indexLocal] = new TextLoaderRange
                    {
                        AllOther = col.Source[indexLocal].AllOther,
                        AutoEnd = col.Source[indexLocal].AutoEnd,
                        ForceVector = col.Source[indexLocal].ForceVector,
                        VariableEnd = col.Source[indexLocal].VariableEnd,
                        Max = col.Source[indexLocal].Max,
                        Min = col.Source[indexLocal].Min
                    };
                }

                tlc.Type = col.Type;
                Arguments.Column[index] = tlc;
            }

            Arguments.HasHeader = useHeader;
            Arguments.Separator = separator;
            Arguments.AllowQuoting = allowQuotedStrings;
            Arguments.AllowSparse = supportSparse;
            Arguments.TrimWhitespace = trimWhitespace;
        }

        private string TypeToName(Type type)
        {
            if (type == typeof(string))
                return "TX";
            else if (type == typeof(float) || type == typeof(double))
                return "R4";
            else if (type == typeof(bool))
                return "BL";
            else
                throw new Exception("Type not implemented or supported."); //Add more types.
        }
    }
}
