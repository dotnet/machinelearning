// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;

[assembly: LoadableClass(NAHandleTransform.Summary, typeof(IDataTransform), typeof(NAHandleTransform), typeof(NAHandleTransform.Arguments), typeof(SignatureDataTransform),
    NAHandleTransform.FriendlyName, "NAHandleTransform", NAHandleTransform.ShortName, "NA", DocName = "transform/NAHandle.md")]

namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="NAHandle"]'/>
    public static class NAHandleTransform
    {
        public enum ReplacementKind : byte
        {
            /// <summary>
            /// Replace with the default value of the column based on it's type. For example, 'zero' for numeric and 'empty' for string/text columns.
            /// </summary>
            [EnumValueDisplay("Zero/empty")]
            DefaultValue = 0,

            /// <summary>
            /// Replace with the mean value of the column. Supports only numeric/time span/ DateTime columns.
            /// </summary>
            Mean = 1,

            /// <summary>
            /// Replace with the minimum value of the column. Supports only numeric/time span/ DateTime columns.
            /// </summary>
            Minimum = 2,

            /// <summary>
            /// Replace with the maximum value of the column. Supports only numeric/time span/ DateTime columns.
            /// </summary>
            Maximum = 3,

            [HideEnumValue]
            Def = DefaultValue,
            [HideEnumValue]
            Default = DefaultValue,
            [HideEnumValue]
            Min = Minimum,
            [HideEnumValue]
            Max = Maximum,
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:rep:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The replacement method to utilize", ShortName = "kind", SortOrder = 2)]
            public ReplacementKind ReplaceWith = ReplacementKind.DefaultValue;

            // Leaving this value null indicates that the default will be used, with the default being imputation by slot for most vectors and
            // imputation across all columns for vectors of unknown size. Specifying by-slot imputation for vectors of unknown size will cause
            // an error to be thrown.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to impute values by slot", ShortName = "slot")]
            public bool ImputeBySlot = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not to concatenate an indicator vector column to the value column", ShortName = "ind")]
            public bool Concat = true;
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The replacement method to utilize")]
            public ReplacementKind? Kind;

            // REVIEW: The default is to perform imputation by slot. If the input column is an unknown size vector type, then imputation
            // will be performed across columns. Should the default be changed/an imputation method required?
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to impute values by slot", ShortName = "slot")]
            public bool? ImputeBySlot;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether or not to concatenate an indicator vector column to the value column", ShortName = "ind")]
            public bool? ConcatIndicator;

            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal const string Summary = "Handle missing values by replacing them with either the default value or the "
            + "mean/min/max value (for non-text columns only). An indicator column can optionally be concatenated, if the" +
            "input column type is numeric.";

        internal const string FriendlyName = "NA Handle Transform";
        internal const string ShortName = "NAHandle";

        /// <summary>
        /// A helper method to create <see cref="NAHandleTransform"/> for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="replaceWith">The replacement method to utilize.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string name, string source = null, ReplacementKind replaceWith = ReplacementKind.DefaultValue)
        {
            var args = new Arguments()
            {
                Column = new[]
                {
                    new Column() { Source = source ?? name, Name = name }
                },
                ReplaceWith = replaceWith
            };
            return Create(env, args, input);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Categorical");
            h.CheckValue(args, nameof(args));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));

            var replaceCols = new List<NAReplaceTransform.ColumnInfo>();
            var naIndicatorCols = new List<NAIndicatorTransform.Column>();
            var naConvCols = new List<ConvertTransform.Column>();
            var concatCols = new List<ConcatTransform.TaggedColumn>();
            var dropCols = new List<string>();
            var tmpIsMissingColNames = input.Schema.GetTempColumnNames(args.Column.Length, "IsMissing");
            var tmpReplaceColNames = input.Schema.GetTempColumnNames(args.Column.Length, "Replace");
            for (int i = 0; i < args.Column.Length; i++)
            {
                var column = args.Column[i];

                var addInd = column.ConcatIndicator ?? args.Concat;
                if (!addInd)
                {
                    replaceCols.Add(new NAReplaceTransform.ColumnInfo(column.Source, column.Name, (NAReplaceTransform.ColumnInfo.ReplacementMode)(column.Kind ?? args.ReplaceWith), column.ImputeBySlot ?? args.ImputeBySlot));
                    continue;
                }

                // Check that the indicator column has a type that can be converted to the NAReplaceTransform output type,
                // so that they can be concatenated.
                if (!input.Schema.TryGetColumnIndex(column.Source, out int inputCol))
                    throw h.Except("Column '{0}' does not exist", column.Source);
                var replaceType = input.Schema.GetColumnType(inputCol);
                if (!Conversions.Instance.TryGetStandardConversion(BoolType.Instance, replaceType.ItemType, out Delegate conv, out bool identity))
                {
                    throw h.Except("Cannot concatenate indicator column of type '{0}' to input column of type '{1}'",
                        BoolType.Instance, replaceType.ItemType);
                }

                // Find a temporary name for the NAReplaceTransform and NAIndicatorTransform output columns.
                var tmpIsMissingColName = tmpIsMissingColNames[i];
                var tmpReplacementColName = tmpReplaceColNames[i];

                // Add an NAHandleTransform column.
                naIndicatorCols.Add(new NAIndicatorTransform.Column() { Name = tmpIsMissingColName, Source = column.Source });

                // Add a ConvertTransform column if necessary.
                if (!identity)
                    naConvCols.Add(new ConvertTransform.Column() { Name = tmpIsMissingColName, Source = tmpIsMissingColName, ResultType = replaceType.ItemType.RawKind });

                // Add the NAReplaceTransform column.
                replaceCols.Add(new NAReplaceTransform.ColumnInfo(column.Source, tmpReplacementColName, (NAReplaceTransform.ColumnInfo.ReplacementMode)(column.Kind ?? args.ReplaceWith), column.ImputeBySlot ?? args.ImputeBySlot));

                // Add the ConcatTransform column.
                if (replaceType.IsVector)
                {
                    concatCols.Add(new ConcatTransform.TaggedColumn()
                    {
                        Name = column.Name,
                        Source = new[] {
                            new KeyValuePair<string, string>(tmpReplacementColName, tmpReplacementColName),
                            new KeyValuePair<string, string>("IsMissing", tmpIsMissingColName)
                        }
                    });
                }
                else
                {
                    concatCols.Add(new ConcatTransform.TaggedColumn()
                    {
                        Name = column.Name,
                        Source = new[]
                        {
                            new KeyValuePair<string, string>(column.Source, tmpReplacementColName),
                            new KeyValuePair<string, string>(string.Format("IsMissing.{0}", column.Source), tmpIsMissingColName),
                        }
                    });
                }

                // Add the temp column to the list of columns to drop at the end.
                dropCols.Add(tmpIsMissingColName);
                dropCols.Add(tmpReplacementColName);
            }

            IDataTransform output = null;

            // Create the indicator columns.
            if (naIndicatorCols.Count > 0)
                output = NAIndicatorTransform.Create(h, new NAIndicatorTransform.Arguments() { Column = naIndicatorCols.ToArray() }, input);

            // Convert the indicator columns to the correct type so that they can be concatenated to the NAReplace outputs.
            if (naConvCols.Count > 0)
            {
                h.AssertValue(output);
                output = new ConvertTransform(h, new ConvertTransform.Arguments() { Column = naConvCols.ToArray() }, output);
            }
            // Create the NAReplace transform.
            output = NAReplaceTransform.Create(env, output ?? input, replaceCols.ToArray());

            // Concat the NAReplaceTransform output and the NAIndicatorTransform output.
            if (naIndicatorCols.Count > 0)
                output = ConcatTransform.Create(h, new ConcatTransform.TaggedArguments() { Column = concatCols.ToArray() }, output);

            // Finally, drop the temporary indicator columns.
            if (dropCols.Count > 0)
                output = SelectColumnsTransform.CreateDrop(h, output, dropCols.ToArray());

            return output;
        }
    }
}
