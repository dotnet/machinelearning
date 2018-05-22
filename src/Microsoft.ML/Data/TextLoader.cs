// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Data
{
    public sealed partial class TextLoaderRange
    {
        public TextLoaderRange()
        {
        }

        /// <summary>
        /// Convenience constructor for the scalar case, when a given column 
        /// in the schema spans only a single column in the dataset.
        /// <see cref="Min"/> and <see cref="Max"/> are set to the single value <paramref name="ordinal"/>.
        /// </summary>
        /// <param name="ordinal">Column index in the dataset.</param>
        public TextLoaderRange(int ordinal)
        {

            Contracts.CheckParam(ordinal >= 0, nameof(ordinal), "Cannot be a negative number");

            Min = ordinal;
            Max = ordinal;
        }

        /// <summary>
        /// Convenience constructor for the vector case, when a given column 
        /// in the schema spans contiguous columns in the dataset.
        /// </summary>
        /// <param name="min">Starting column index in the dataset.</param>
        /// <param name="max">Ending column index in the dataset.</param>
        public TextLoaderRange(int min, int max)
        {

            Contracts.CheckParam(min >= 0, nameof(min), "Cannot be a negative number.");
            Contracts.CheckParam(max >= min, nameof(max), "Cannot be less than " + nameof(min) +".");

            Min = min;
            Max = max;
        }
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

                Runtime.Data.TextLoader.Range[] sources;
                if (!Runtime.Data.TextLoader.Column.TryParseSourceEx(mappingAttr.Ordinal, out sources))
                    throw Contracts.Except($"{mappingAttr.Ordinal} could not be parsed.");

                Contracts.Assert(sources != null);

                TextLoaderColumn tlc = new TextLoaderColumn();
                tlc.Name = name;
                tlc.Source = new TextLoaderRange[sources.Length];
                DataKind dk;
                if (!TryGetDataKind(field.FieldType.IsArray ? field.FieldType.GetElementType() : field.FieldType, out dk))
                    throw Contracts.Except($"{name} is of unsupported type.");

                tlc.Type = dk;

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

                Arguments.Column[index] = tlc;
            }

            Arguments.HasHeader = useHeader;
            Arguments.Separator = new[] { separator };
            Arguments.AllowQuoting = allowQuotedStrings;
            Arguments.AllowSparse = supportSparse;
            Arguments.TrimWhitespace = trimWhitespace;

            return this;
        }

        /// <summary>
        /// Try to map a System.Type to a corresponding DataKind value.
        /// </summary>
        private static bool TryGetDataKind(Type type, out DataKind kind)
        {
            Contracts.AssertValue(type);

            // REVIEW: Make this more efficient. Should we have a global dictionary?
            if (type == typeof(DvInt1) || type == typeof(sbyte))
                kind = DataKind.I1;
            else if (type == typeof(byte) || type == typeof(char))
                kind = DataKind.U1;
            else if (type == typeof(DvInt2) || type == typeof(short))
                kind = DataKind.I2;
            else if (type == typeof(ushort))
                kind = DataKind.U2;
            else if (type == typeof(DvInt4) || type == typeof(int))
                kind = DataKind.I4;
            else if (type == typeof(uint))
                kind = DataKind.U4;
            else if (type == typeof(DvInt8) || type == typeof(long))
                kind = DataKind.I8;
            else if (type == typeof(ulong))
                kind = DataKind.U8;
            else if (type == typeof(Single))
                kind = DataKind.R4;
            else if (type == typeof(Double))
                kind = DataKind.R8;
            else if (type == typeof(DvText) || type == typeof(string))
                kind = DataKind.TX;
            else if (type == typeof(DvBool) || type == typeof(bool))
                kind = DataKind.BL;
            else if (type == typeof(DvTimeSpan) || type == typeof(TimeSpan))
                kind = DataKind.TS;
            else if (type == typeof(DvDateTime) || type == typeof(DateTime))
                kind = DataKind.DT;
            else if (type == typeof(DvDateTimeZone) || type == typeof(TimeZoneInfo))
                kind = DataKind.DZ;
            else if (type == typeof(UInt128))
                kind = DataKind.UG;
            else
            {
                kind = default(DataKind);
                return false;
            }

            return true;
        }
    }
}
