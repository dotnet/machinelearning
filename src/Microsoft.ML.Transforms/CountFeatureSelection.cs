// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms.FeatureSelection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

[assembly: LoadableClass(CountFeatureSelectingEstimator.Summary, typeof(IDataTransform), typeof(CountFeatureSelectingEstimator), typeof(CountFeatureSelectingEstimator.Arguments), typeof(SignatureDataTransform),
    CountFeatureSelectingEstimator.UserName, "CountFeatureSelectionTransform", "CountFeatureSelection")]

namespace Microsoft.ML.Transforms.FeatureSelection
{
    /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
    public sealed class CountFeatureSelectingEstimator : IEstimator<ITransformer>
    {
        internal const string Summary = "Selects the slots for which the count of non-default values is greater than or equal to a threshold.";
        internal const string UserName = "Count Feature Selection Transform";

        private readonly IHost _host;
        private readonly ColumnInfo[] _columns;

        internal static class Defaults
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

        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly long MinCount;

            /// <summary>
            /// Describes the parameters of the feature selection process for a column pair.
            /// </summary>
            /// <param name="input">Name of the input column.</param>
            /// <param name="output">Name of the column resulting from the transformation of <paramref name="input"/>. Null means <paramref name="input"/> is replaced.</param>
            /// <param name="minCount">If the count of non-default values for a slot is greater than or equal to this threshold in the training data, the slot is preserved.</param>
            public ColumnInfo(string input, string output = null, long minCount = Defaults.Count)
            {
                Input = input;
                Contracts.CheckValue(Input, nameof(Input));
                Output = output ?? input;
                Contracts.CheckValue(Output, nameof(Output));
                MinCount = minCount;
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="env">The environment to use.</param>
        /// <param name="columns">Describes the parameters of the feature selection process for each column pair.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CountFeatureSelectingEstimator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        public CountFeatureSelectingEstimator(IHostEnvironment env, params ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckUserArg(Utils.Size(columns) > 0, nameof(columns));

            _columns = columns;
        }

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="env">The environment to use.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the column resulting from the transformation of <paramref name="inputColumn"/>. Null means <paramref name="inputColumn"/> is replaced. </param>
        /// <param name="minCount">If the count of non-default values for a slot is greater than or equal to this threshold in the training data, the slot is preserved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CountFeatureSelectingEstimator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        public CountFeatureSelectingEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null, long minCount = Defaults.Count)
            : this(env, new ColumnInfo(inputColumn, outputColumn ?? inputColumn, minCount))
        {
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colPair in _columns)
            {
                if (!inputSchema.TryFindColumn(colPair.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.Input);
                if (!CountFeatureSelectionUtils.IsValidColumnType(col.ItemType))
                    throw _host.ExceptUserArg(nameof(inputSchema), "Column '{0}' does not have compatible type. Expected types are float, double or string.", colPair.Input);
                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.CategoricalSlotRanges, out var categoricalSlotMeta))
                    metadata.Add(categoricalSlotMeta);
                metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));
                result[colPair.Output] = new SchemaShape.Column(colPair.Output, col.Kind, col.ItemType, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }

        public ITransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            int[] colSizes;
            var scores = CountFeatureSelectionUtils.Train(_host, input, _columns.Select(column => column.Input).ToArray(), out colSizes);
            var size = _columns.Length;

            using (var ch = _host.Start("Dropping Slots"))
            {
                // If no slots should be dropped from a column, use copy column to generate the corresponding output column.
                SlotsDroppingTransformer.ColumnInfo[] dropSlotsColumns;
                (string input, string output)[] copyColumnsPairs;
                CreateDropAndCopyColumns(_columns, size, scores, out int[] selectedCount, out dropSlotsColumns, out copyColumnsPairs);

                for (int i = 0; i < selectedCount.Length; i++)
                    ch.Info(MessageSensitivity.Schema, "Selected {0} slots out of {1} in column '{2}'", selectedCount[i], colSizes[i], _columns[i].Input);
                ch.Info("Total number of slots selected: {0}", selectedCount.Sum());

                if (dropSlotsColumns.Length <= 0)
                    return new ColumnCopyingTransformer(_host, copyColumnsPairs);
                else if (copyColumnsPairs.Length <= 0)
                    return new SlotsDroppingTransformer(_host, dropSlotsColumns);

                var transformerChain = new TransformerChain<SlotsDroppingTransformer>(
                    new ITransformer[] {
                        new ColumnCopyingTransformer(_host, copyColumnsPairs),
                        new SlotsDroppingTransformer(_host, dropSlotsColumns)
                    });
                return transformerChain;
            }
        }

        /// <summary>
        /// Create method corresponding to SignatureDataTransform.
        /// </summary>
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(args, nameof(args));
            host.CheckValue(input, nameof(input));
            host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));
            host.CheckUserArg(args.Count > 0, nameof(args.Count));

            var columnInfos = args.Column.Select(inColName => new ColumnInfo(inColName, minCount: args.Count)).ToArray();

            return new CountFeatureSelectingEstimator(env, columnInfos).Fit(input).Transform(input) as IDataTransform;
        }

        private static void CreateDropAndCopyColumns(ColumnInfo[] columnInfos, int size, long[][] scores,
            out int[] selectedCount, out SlotsDroppingTransformer.ColumnInfo[] dropSlotsColumns, out (string input, string output)[] copyColumnsPairs)
        {
            Contracts.Assert(size > 0);
            Contracts.Assert(Utils.Size(scores) == size);
            Contracts.AssertValue(columnInfos);
            Contracts.Assert(Utils.Size(columnInfos) == size);

            selectedCount = new int[scores.Length];
            var dropSlotsCols = new List<SlotsDroppingTransformer.ColumnInfo>();
            var copyCols = new List<(string input, string output)>();
            for (int i = 0; i < size; i++)
            {
                var slots = new List<(int min, int? max)>();
                var score = scores[i];
                selectedCount[i] = 0;
                for (int j = 0; j < score.Length; j++)
                {
                    if (score[j] < columnInfos[i].MinCount)
                    {
                        // Adjacent slots are combined into a single range.
                        int min = j;
                        while (j < score.Length && score[j] < columnInfos[i].MinCount)
                            j++;
                        int max = j - 1;
                        slots.Add((min, max));
                        if (j < score.Length)
                            selectedCount[i]++;
                    }
                    else
                        selectedCount[i]++;
                }
                if (slots.Count <= 0)
                    copyCols.Add((columnInfos[i].Input, columnInfos[i].Output));
                else
                    dropSlotsCols.Add(new SlotsDroppingTransformer.ColumnInfo(columnInfos[i].Input, columnInfos[i].Output, slots.ToArray()));
            }
            dropSlotsColumns = dropSlotsCols.ToArray();
            copyColumnsPairs = copyCols.ToArray();
        }
    }

    internal static class CountFeatureSelectionUtils
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
                    throw env.ExceptUserArg(nameof(CountFeatureSelectingEstimator.Arguments.Column), "Source column '{0}' not found", colName);

                var colType = schema.GetColumnType(colSrc);
                if (colType.IsVector && !colType.IsKnownSizeVector)
                    throw env.ExceptUserArg(nameof(CountFeatureSelectingEstimator.Arguments.Column), "Variable length column '{0}' is not allowed", colName);

                activeInput[colSrc] = true;
                colSrcs[i] = colSrc;
                colTypes[i] = colType;
                colSizes[i] = colType.ValueCount;
            }

            var aggregators = new CountAggregator[size];
            long rowCur = 0;
            double rowCount = input.GetRowCount() ?? double.NaN;
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

        public static bool IsValidColumnType(ColumnType type)
            => type == NumberType.R4 || type == NumberType.R8 || type.IsText;

        private static CountAggregator GetOneAggregator(Row row, ColumnType colType, int colSrc)
        {
            Func<Row, ColumnType, int, CountAggregator> del = GetOneAggregator<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.RawType);
            return (CountAggregator)methodInfo.Invoke(null, new object[] { row, colType, colSrc });
        }

        private static CountAggregator GetOneAggregator<T>(Row row, ColumnType colType, int colSrc)
        {
            return new CountAggregator<T>(colType, row.GetGetter<T>(colSrc));
        }

        private static CountAggregator GetVecAggregator(Row row, ColumnType colType, int colSrc)
        {
            Func<Row, ColumnType, int, CountAggregator> del = GetVecAggregator<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.ItemType.RawType);
            return (CountAggregator)methodInfo.Invoke(null, new object[] { row, colType, colSrc });
        }

        private static CountAggregator GetVecAggregator<T>(Row row, ColumnType colType, int colSrc)
        {
            return new CountAggregator<T>(colType, row.GetGetter<VBuffer<T>>(colSrc));
        }

        private abstract class CountAggregator
        {
            public abstract long[] Count { get; }
            public abstract void ProcessValue();
        }

        private sealed class CountAggregator<T> : CountAggregator, IColumnAggregator<VBuffer<T>>
        {
            private readonly long[] _count;
            private readonly Action _fillBuffer;
            private readonly InPredicate<T> _isDefault;
            private readonly InPredicate<T> _isMissing;
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
                        VBufferEditor.CreateFromBuffer(ref _buffer).Values[0] = t;
                    };
                _isDefault = Runtime.Data.Conversion.Conversions.Instance.GetIsDefaultPredicate<T>(type);
                if (!Runtime.Data.Conversion.Conversions.Instance.TryGetIsNAPredicate<T>(type, out _isMissing))
                    _isMissing = (in T value) => false;
            }

            public CountAggregator(ColumnType type, ValueGetter<VBuffer<T>> getter)
            {
                Contracts.Assert(type.IsKnownSizeVector);
                var size = type.ValueCount;
                _count = new long[size];
                _fillBuffer = () => getter(ref _buffer);
                _isDefault = Runtime.Data.Conversion.Conversions.Instance.GetIsDefaultPredicate<T>(type.ItemType);
                if (!Runtime.Data.Conversion.Conversions.Instance.TryGetIsNAPredicate<T>(type.ItemType, out _isMissing))
                    _isMissing = (in T value) => false;
            }

            public override long[] Count
            {
                get { return _count; }
            }

            public override void ProcessValue()
            {
                _fillBuffer();
                ProcessValue(in _buffer);
            }

            public void ProcessValue(in VBuffer<T> value)
            {
                var size = _count.Length;
                Contracts.Check(value.Length == size);

                foreach (var kvp in value.Items())
                {
                    var val = kvp.Value;
                    if (!_isDefault(in val) && !_isMissing(in val))
                        _count[kvp.Key]++;
                }
            }

            public void Finish()
            {
            }
        }
    }
}
