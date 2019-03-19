// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(MissingValueHandlingTransformer.Summary, typeof(IDataTransform), typeof(MissingValueHandlingTransformer),
    typeof(MissingValueHandlingTransformer.Options), typeof(SignatureDataTransform),
    MissingValueHandlingTransformer.FriendlyName, "NAHandleTransform", MissingValueHandlingTransformer.ShortName, "NA", DocName = "transform/NAHandle.md")]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="NAHandle"]'/>
    internal static class MissingValueHandlingTransformer
    {
        public enum ReplacementKind : byte
        {
            /// <summary>
            /// Replace with the default value of the column based on its type. For example, 'zero' for numeric and 'empty' for string/text columns.
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

        public sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:rep:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

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

            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
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
        /// A helper method to create <see cref="MissingValueHandlingTransformer"/> for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="outputColumnName">Name of the output column.</param>
        /// <param name="inputColumnName">Name of the column to be transformed. If this is null '<paramref name="outputColumnName"/>' will be used.</param>
        /// <param name="replaceWith">The replacement method to utilize.</param>
        private static IDataView Create(IHostEnvironment env, IDataView input, string outputColumnName, string inputColumnName = null,
            ReplacementKind replaceWith = ReplacementKind.DefaultValue)
        {
            var args = new Options()
            {
                Columns = new[]
                {
                    new Column() { Name = outputColumnName, Source = inputColumnName ?? outputColumnName }
                },
                ReplaceWith = replaceWith
            };
            return Create(env, args, input);
        }

        /// Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Categorical");
            h.CheckValue(options, nameof(options));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns));

            var replaceCols = new List<MissingValueReplacingEstimator.ColumnOptions>();
            var naIndicatorCols = new List<MissingValueIndicatorTransformer.Column>();
            var naConvCols = new List<TypeConvertingEstimator.ColumnOptions>();
            var concatCols = new List<ColumnConcatenatingTransformer.TaggedColumn>();
            var dropCols = new List<string>();
            var tmpIsMissingColNames = input.Schema.GetTempColumnNames(options.Columns.Length, "IsMissing");
            var tmpReplaceColNames = input.Schema.GetTempColumnNames(options.Columns.Length, "Replace");
            for (int i = 0; i < options.Columns.Length; i++)
            {
                var column = options.Columns[i];

                var addInd = column.ConcatIndicator ?? options.Concat;
                if (!addInd)
                {
                    replaceCols.Add(new MissingValueReplacingEstimator.ColumnOptions(column.Name, column.Source,
                        (MissingValueReplacingEstimator.ReplacementMode)(column.Kind ?? options.ReplaceWith), column.ImputeBySlot ?? options.ImputeBySlot));
                    continue;
                }

                // Check that the indicator column has a type that can be converted to the NAReplaceTransform output type,
                // so that they can be concatenated.
                if (!input.Schema.TryGetColumnIndex(column.Source, out int inputCol))
                    throw h.Except("Column '{0}' does not exist", column.Source);
                var replaceType = input.Schema[inputCol].Type;
                var replaceItemType = replaceType.GetItemType();
                if (!Data.Conversion.Conversions.Instance.TryGetStandardConversion(BooleanDataViewType.Instance, replaceItemType, out Delegate conv, out bool identity))
                {
                    throw h.Except("Cannot concatenate indicator column of type '{0}' to input column of type '{1}'",
                        BooleanDataViewType.Instance, replaceItemType);
                }

                // Find a temporary name for the NAReplaceTransform and NAIndicatorTransform output columns.
                var tmpIsMissingColName = tmpIsMissingColNames[i];
                var tmpReplacementColName = tmpReplaceColNames[i];

                // Add an NAHandleTransform column.
                naIndicatorCols.Add(new MissingValueIndicatorTransformer.Column() { Name = tmpIsMissingColName, Source = column.Source });

                // Add a ConvertTransform column if necessary.
                if (!identity)
                {
                    if (!replaceItemType.RawType.TryGetDataKind(out InternalDataKind replaceItemTypeKind))
                    {
                        throw h.Except("Cannot get a DataKind for type '{0}'", replaceItemType.RawType);
                    }
                    naConvCols.Add(new TypeConvertingEstimator.ColumnOptions(tmpIsMissingColName, replaceItemTypeKind.ToDataKind(), tmpIsMissingColName));
                }

                // Add the NAReplaceTransform column.
                replaceCols.Add(new MissingValueReplacingEstimator.ColumnOptions(tmpReplacementColName, column.Source,
                    (MissingValueReplacingEstimator.ReplacementMode)(column.Kind ?? options.ReplaceWith), column.ImputeBySlot ?? options.ImputeBySlot));

                // Add the ConcatTransform column.
                if (replaceType is VectorType)
                {
                    concatCols.Add(new ColumnConcatenatingTransformer.TaggedColumn()
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
                    concatCols.Add(new ColumnConcatenatingTransformer.TaggedColumn()
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
                output = MissingValueIndicatorTransformer.Create(h, new MissingValueIndicatorTransformer.Options() { Columns = naIndicatorCols.ToArray() }, input);

            // Convert the indicator columns to the correct type so that they can be concatenated to the NAReplace outputs.
            if (naConvCols.Count > 0)
            {
                h.AssertValue(output);
                //REVIEW: all this need to be converted to estimatorChain as soon as we done with dropcolumns.
                output = new TypeConvertingTransformer(h, naConvCols.ToArray()).Transform(output) as IDataTransform;
            }
            // Create the NAReplace transform.
            output = MissingValueReplacingTransformer.Create(env, output ?? input, replaceCols.ToArray());

            // Concat the NAReplaceTransform output and the NAIndicatorTransform output.
            if (naIndicatorCols.Count > 0)
                output = ColumnConcatenatingTransformer.Create(h, new ColumnConcatenatingTransformer.TaggedOptions() { Columns = concatCols.ToArray() }, output);

            // Finally, drop the temporary indicator columns.
            if (dropCols.Count > 0)
                output = ColumnSelectingTransformer.CreateDrop(h, output, dropCols.ToArray()) as IDataTransform;

            return output;
        }
    }
}
