// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(CountFeatureSelectionTransform.Summary, typeof(IDataTransform), typeof(CountFeatureSelectionTransform), typeof(CountFeatureSelectionTransform.Arguments), typeof(SignatureDataTransform),
    CountFeatureSelectionTransform.UserName, "CountFeatureSelectionTransform", "CountFeatureSelection")]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Selects the slots for which the count of non-default values is greater than a threshold.
    /// Uses a set of aggregators to count the number of non-default values for each slot and
    /// instantiates a DropSlots transform to actually drop the slots.
    /// </summary>
    public static class CountFeatureSelectionTransform
    {
        public const string Summary = "Selects the slots for which the count of non-default values is greater than or equal to a threshold.";
        public const string UserName = "Count Feature Selection Transform";

        private static class Defaults
        {
            public const long Count = 1;
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to use for feature selection", ShortName = "col", SortOrder = 1)]
            public string[] Column;

            [Argument(ArgumentType.Required, HelpText = "If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved", ShortName = "c", SortOrder = 1)]
            public long Count = Defaults.Count;
        }

        internal static string RegistrationName = "CountFeatureSelectionTransform";

        /// <summary>
        /// A helper method to create CountFeatureSelection transform for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        /// <param name="columns">Columns to use for feature selection.</param>
        /// <returns></returns>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, long count = Defaults.Count, params string[] columns)
        {
            var args = new Arguments()
            {
                Column = columns,
                Count = count
            };
            return Create(env, args, input);
        }

        /// <summary>
        /// Create method corresponding to SignatureDataTransform.
        /// </summary>
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(args, nameof(args));
            host.CheckValue(input, nameof(input));
            host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));
            host.CheckUserArg(args.Count > 0, nameof(args.Count));

            int[] colSizes;
            var scores = CountFeatureSelectionUtils.Train(host, input, args.Column, out colSizes);
            var size = args.Column.Length;

            using (var ch = host.Start("Dropping Slots"))
            {
                int[] selectedCount;
                var columns = CreateDropSlotsColumns(args, size, scores, out selectedCount);

                if (columns.Count <= 0)
                {
                    ch.Info("No features are being dropped.");
                    return NopTransform.CreateIfNeeded(host, input);
                }

                for (int i = 0; i < selectedCount.Length; i++)
                    ch.Info(MessageSensitivity.Schema, "Selected {0} slots out of {1} in column '{2}'", selectedCount[i], colSizes[i], args.Column[i]);
                ch.Info("Total number of slots selected: {0}", selectedCount.Sum());

                var dsArgs = new DropSlotsTransform.Arguments();
                dsArgs.Column = columns.ToArray();
                ch.Done();
                return new DropSlotsTransform(host, dsArgs, input);
            }
        }

        private static List<DropSlotsTransform.Column> CreateDropSlotsColumns(Arguments args, int size, long[][] scores, out int[] selectedCount)
        {
            Contracts.Assert(size > 0);
            Contracts.Assert(Utils.Size(scores) == size);
            Contracts.AssertValue(args);
            Contracts.Assert(Utils.Size(args.Column) == size);

            selectedCount = new int[scores.Length];
            var columns = new List<DropSlotsTransform.Column>();
            for (int i = 0; i < size; i++)
            {
                var col = new DropSlotsTransform.Column();
                col.Source = args.Column[i];
                var slots = new List<DropSlotsTransform.Range>();
                var score = scores[i];
                selectedCount[i] = 0;
                for (int j = 0; j < score.Length; j++)
                {
                    if (score[j] < args.Count)
                    {
                        // Adjacent slots are combined into a single range.
                        var range = new DropSlotsTransform.Range();
                        range.Min = j;
                        while (j < score.Length && score[j] < args.Count)
                            j++;
                        range.Max = j - 1;
                        slots.Add(range);
                        if (j < score.Length)
                            selectedCount[i]++;
                    }
                    else
                        selectedCount[i]++;
                }
                if (slots.Count > 0)
                {
                    col.Slots = slots.ToArray();
                    columns.Add(col);
                }
            }
            return columns;
        }
    }

    public static class CountFeatureSelectionUtils
    {
        /// <summary>
        /// Returns the feature selection scores for each slot of each column.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="input">The input dataview.</param>
        /// <param name="columns">The columns for which to compute the feature selection scores.</param>
        /// <param name="colSizes">Outputs an array containing the vector sizes of the input columns</param>
        /// <returns>A list of scores.</returns>
        public static long[][] Train(IHostEnvironment env, IDataView input, string[] columns, out int[] colSizes)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));
            env.CheckParam(Utils.Size(columns) > 0, nameof(columns));

            var schema = input.Schema;
            var size = columns.Length;
            var activeInput = new bool[schema.ColumnCount];
            var colSrcs = new int[size];
            var colTypes = new ColumnType[size];
            colSizes = new int[size];
            for (int i = 0; i < size; i++)
            {
                int colSrc;
                var colName = columns[i];
                if (!schema.TryGetColumnIndex(colName, out colSrc))
                    throw env.ExceptUserArg(nameof(CountFeatureSelectionTransform.Arguments.Column), "Source column '{0}' not found", colName);

                var colType = schema.GetColumnType(colSrc);
                if (colType.IsVector && !colType.IsKnownSizeVector)
                    throw env.ExceptUserArg(nameof(CountFeatureSelectionTransform.Arguments.Column), "Variable length column '{0}' is not allowed", colName);

                activeInput[colSrc] = true;
                colSrcs[i] = colSrc;
                colTypes[i] = colType;
                colSizes[i] = colType.ValueCount;
            }

            var aggregators = new CountAggregator[size];
            long rowCur = 0;
            double rowCount = input.GetRowCount(true) ?? double.NaN;
            using (var pch = env.StartProgressChannel("Aggregating counts"))
            using (var cursor = input.GetRowCursor(col => activeInput[col]))
            {
                var header = new ProgressHeader(new[] { "rows" });
                pch.SetHeader(header, e => { e.SetProgress(0, rowCur, rowCount); });
                for (int i = 0; i < size; i++)
                {
                    if (colTypes[i].IsVector)
                        aggregators[i] = GetVecAggregator(cursor, colTypes[i], colSrcs[i]);
                    else
                        aggregators[i] = GetOneAggregator(cursor, colTypes[i], colSrcs[i]);
                }

                while (cursor.MoveNext())
                {
                    for (int i = 0; i < size; i++)
                        aggregators[i].ProcessValue();
                    rowCur++;
                }
                pch.Checkpoint(rowCur);
            }
            return aggregators.Select(a => a.Count).ToArray();
        }

        private static CountAggregator GetOneAggregator(IRow row, ColumnType colType, int colSrc)
        {
            Func<IRow, ColumnType, int, CountAggregator> del = GetOneAggregator<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.RawType);
            return (CountAggregator)methodInfo.Invoke(null, new object[] { row, colType, colSrc });
        }

        private static CountAggregator GetOneAggregator<T>(IRow row, ColumnType colType, int colSrc)
            where T : IEquatable<T>
        {
            return new CountAggregator<T>(colType, row.GetGetter<T>(colSrc));
        }

        private static CountAggregator GetVecAggregator(IRow row, ColumnType colType, int colSrc)
        {
            Func<IRow, ColumnType, int, CountAggregator> del = GetVecAggregator<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.ItemType.RawType);
            return (CountAggregator)methodInfo.Invoke(null, new object[] { row, colType, colSrc });
        }

        private static CountAggregator GetVecAggregator<T>(IRow row, ColumnType colType, int colSrc)
            where T : IEquatable<T>
        {
            return new CountAggregator<T>(colType, row.GetGetter<VBuffer<T>>(colSrc));
        }

        private abstract class CountAggregator
        {
            public abstract long[] Count { get; }
            public abstract void ProcessValue();
        }

        private sealed class CountAggregator<T> : CountAggregator, IColumnAggregator<VBuffer<T>>
            where T : IEquatable<T>
        {
            private readonly long[] _count;
            private readonly Action _fillBuffer;
            private readonly RefPredicate<T> _isDefault;
            private readonly RefPredicate<T> _isMissing;
            private VBuffer<T> _buffer;

            public CountAggregator(ColumnType type, ValueGetter<T> getter)
            {
                Contracts.Assert(type.IsPrimitive);
                _count = new long[1];
                _buffer = new VBuffer<T>(1, new T[1]);
                var t = default(T);
                _fillBuffer =
                    () =>
                    {
                        getter(ref t);
                        _buffer.Values[0] = t;
                    };
                _isDefault = Conversions.Instance.GetIsDefaultPredicate<T>(type);
                _isMissing = Conversions.Instance.GetIsNAPredicate<T>(type);
            }

            public CountAggregator(ColumnType type, ValueGetter<VBuffer<T>> getter)
            {
                Contracts.Assert(type.IsKnownSizeVector);
                var size = type.ValueCount;
                _count = new long[size];
                _fillBuffer = () => getter(ref _buffer);
                _isDefault = Conversions.Instance.GetIsDefaultPredicate<T>(type.ItemType);
                _isMissing = Conversions.Instance.GetIsNAPredicate<T>(type.ItemType);
            }

            public override long[] Count
            {
                get { return _count; }
            }

            public override void ProcessValue()
            {
                _fillBuffer();
                ProcessValue(ref _buffer);
            }

            public void ProcessValue(ref VBuffer<T> value)
            {
                var size = _count.Length;
                Contracts.Check(value.Length == size);

                foreach (var kvp in value.Items())
                {
                    var val = kvp.Value;
                    if (!_isDefault(ref val) && !_isMissing(ref val))
                        _count[kvp.Key]++;
                }
            }

            public void Finish()
            {
            }
        }
    }
}
