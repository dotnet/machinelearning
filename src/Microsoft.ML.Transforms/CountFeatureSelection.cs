// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(CountFeatureSelectingEstimator.Summary, typeof(IDataTransform), typeof(CountFeatureSelectingEstimator), typeof(CountFeatureSelectingEstimator.Options), typeof(SignatureDataTransform),
    CountFeatureSelectingEstimator.UserName, "CountFeatureSelectionTransform", "CountFeatureSelection")]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
    public sealed class CountFeatureSelectingEstimator : IEstimator<ITransformer>
    {
        internal const string Summary = "Selects the slots for which the count of non-default values is greater than or equal to a threshold.";
        internal const string UserName = "Count Feature Selection Transform";

        private readonly IHost _host;
        private readonly ColumnOptions[] _columns;

        [BestFriend]
        internal static class Defaults
        {
            public const long Count = 1;
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to use for feature selection", Name = "Column", ShortName = "col", SortOrder = 1)]
            public string[] Columns;

            [Argument(ArgumentType.Required, HelpText = "If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved", ShortName = "c", SortOrder = 1)]
            public long Count = Defaults.Count;
        }

        internal static string RegistrationName = "CountFeatureSelectionTransform";

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary> Name of the column resulting from the transformation of <see cref="InputColumnName"/>.</summary>
            public readonly string Name;
            /// <summary> Name of the column to transform.</summary>
            public readonly string InputColumnName;
            /// <summary>If the count of non-default values for a slot is greater than or equal to this threshold in the training data, the slot is preserved.</summary>
            public readonly long Count;

            /// <summary>
            /// Describes the parameters of the feature selection process for a column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold in the training data, the slot is preserved.</param>

            public ColumnOptions(string name, string inputColumnName = null, long count = Defaults.Count)
            {
                Name = name;
                Contracts.CheckValue(Name, nameof(Name));

                InputColumnName = inputColumnName ?? name;
                Contracts.CheckValue(InputColumnName, nameof(InputColumnName));
                Contracts.CheckParam(count >= 0, nameof(count), "Must be non-negative.");
                Count = count;
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
        internal CountFeatureSelectingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckUserArg(Utils.Size(columns) > 0, nameof(columns));

            _columns = columns;
        }

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="env">The environment to use.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="minCount">If the count of non-default values for a slot is greater than or equal to this threshold in the training data, the slot is preserved.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[CountFeatureSelectingEstimator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        internal CountFeatureSelectingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, long minCount = Defaults.Count)
            : this(env, new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName, minCount))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colPair in _columns)
            {
                if (!inputSchema.TryFindColumn(colPair.InputColumnName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.InputColumnName);
                if (!CountFeatureSelectionUtils.IsValidColumnType(col.ItemType))
                    throw _host.ExceptUserArg(nameof(inputSchema), "Column '{0}' does not have compatible type. Expected types are float, double or string.", colPair.InputColumnName);
                var metadata = new List<SchemaShape.Column>();
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.CategoricalSlotRanges, out var categoricalSlotMeta))
                    metadata.Add(categoricalSlotMeta);
                metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
                result[colPair.Name] = new SchemaShape.Column(colPair.Name, col.Kind, col.ItemType, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }

        /// <summary>
        /// Trains and returns a <see cref="ITransformer"/>.
        /// </summary>
        public ITransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            int[] colSizes;
            var scores = CountFeatureSelectionUtils.Train(_host, input, _columns.Select(column => column.InputColumnName).ToArray(), out colSizes);
            var size = _columns.Length;

            using (var ch = _host.Start("Dropping Slots"))
            {
                // If no slots should be dropped from a column, use copy column to generate the corresponding output column.
                SlotsDroppingTransformer.ColumnOptions[] dropSlotsColumns;
                (string outputColumnName, string inputColumnName)[] copyColumnsPairs;
                CreateDropAndCopyColumns(_columns, size, scores, out int[] selectedCount, out dropSlotsColumns, out copyColumnsPairs);

                for (int i = 0; i < selectedCount.Length; i++)
                    ch.Info(MessageSensitivity.Schema, "Selected {0} slots out of {1} in column '{2}'", selectedCount[i], colSizes[i], _columns[i].InputColumnName);
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
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(options, nameof(options));
            host.CheckValue(input, nameof(input));
            host.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns));
            host.CheckUserArg(options.Count > 0, nameof(options.Count));

            var columnOptions = options.Columns.Select(inColName => new ColumnOptions(inColName, count: options.Count)).ToArray();

            return new CountFeatureSelectingEstimator(env, columnOptions).Fit(input).Transform(input) as IDataTransform;
        }

        private static void CreateDropAndCopyColumns(ColumnOptions[] columnOptions, int size, long[][] scores,
            out int[] selectedCount, out SlotsDroppingTransformer.ColumnOptions[] dropSlotsColumns, out (string outputColumnName, string inputColumnName)[] copyColumnsPairs)
        {
            Contracts.Assert(size > 0);
            Contracts.Assert(Utils.Size(scores) == size);
            Contracts.AssertValue(columnOptions);
            Contracts.Assert(Utils.Size(columnOptions) == size);

            selectedCount = new int[scores.Length];
            var dropSlotsCols = new List<SlotsDroppingTransformer.ColumnOptions>();
            var copyCols = new List<(string outputColumnName, string inputColumnName)>();
            for (int i = 0; i < size; i++)
            {
                var slots = new List<(int min, int? max)>();
                var score = scores[i];
                selectedCount[i] = 0;
                for (int j = 0; j < score.Length; j++)
                {
                    if (score[j] < columnOptions[i].Count)
                    {
                        // Adjacent slots are combined into a single range.
                        int min = j;
                        while (j < score.Length && score[j] < columnOptions[i].Count)
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
                    copyCols.Add((columnOptions[i].Name, columnOptions[i].InputColumnName));
                else
                    dropSlotsCols.Add(new SlotsDroppingTransformer.ColumnOptions(columnOptions[i].Name, columnOptions[i].InputColumnName, slots.ToArray()));
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
            var activeCols = new List<DataViewSchema.Column>();
            var colSrcs = new int[size];
            var colTypes = new DataViewType[size];
            colSizes = new int[size];
            for (int i = 0; i < size; i++)
            {
                int colSrc;
                var colName = columns[i];
                if (!schema.TryGetColumnIndex(colName, out colSrc))
                    throw env.ExceptUserArg(nameof(CountFeatureSelectingEstimator.Options.Columns), "Source column '{0}' not found", colName);

                var colType = schema[colSrc].Type;
                if (colType is VectorType vectorType && !vectorType.IsKnownSize)
                    throw env.ExceptUserArg(nameof(CountFeatureSelectingEstimator.Options.Columns), "Variable length column '{0}' is not allowed", colName);

                activeCols.Add(schema[colSrc]);
                colSrcs[i] = colSrc;
                colTypes[i] = colType;
                colSizes[i] = colType.GetValueCount();
            }

            var aggregators = new CountAggregator[size];
            long rowCur = 0;
            double rowCount = input.GetRowCount() ?? double.NaN;
            using (var pch = env.StartProgressChannel("Aggregating counts"))
            using (var cursor = input.GetRowCursor(activeCols))
            {
                var header = new ProgressHeader(new[] { "rows" });
                pch.SetHeader(header, e => { e.SetProgress(0, rowCur, rowCount); });
                for (int i = 0; i < size; i++)
                {
                    if (colTypes[i] is VectorType vectorType)
                        aggregators[i] = GetVecAggregator(cursor, vectorType, colSrcs[i]);
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

        public static bool IsValidColumnType(DataViewType type)
            => type == NumberDataViewType.Single || type == NumberDataViewType.Double || type is TextDataViewType;

        private static CountAggregator GetOneAggregator(DataViewRow row, DataViewType colType, int colSrc)
        {
            Func<DataViewRow, DataViewType, int, CountAggregator> del = GetOneAggregator<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.RawType);
            return (CountAggregator)methodInfo.Invoke(null, new object[] { row, colType, colSrc });
        }

        private static CountAggregator GetOneAggregator<T>(DataViewRow row, DataViewType colType, int colSrc)
        {
            return new CountAggregator<T>(colType, row.GetGetter<T>(row.Schema[colSrc]));
        }

        private static CountAggregator GetVecAggregator(DataViewRow row, VectorType colType, int colSrc)
        {
            Func<DataViewRow, VectorType, int, CountAggregator> del = GetVecAggregator<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.ItemType.RawType);
            return (CountAggregator)methodInfo.Invoke(null, new object[] { row, colType, colSrc });
        }

        private static CountAggregator GetVecAggregator<T>(DataViewRow row, VectorType colType, int colSrc)
        {
            return new CountAggregator<T>(colType, row.GetGetter<VBuffer<T>>(row.Schema[colSrc]));
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

            public CountAggregator(DataViewType type, ValueGetter<T> getter)
            {
                Contracts.Assert(type is PrimitiveDataViewType);
                _count = new long[1];
                _buffer = new VBuffer<T>(1, new T[1]);
                var t = default(T);
                _fillBuffer =
                    () =>
                    {
                        getter(ref t);
                        VBufferEditor.CreateFromBuffer(ref _buffer).Values[0] = t;
                    };
                _isDefault = Data.Conversion.Conversions.Instance.GetIsDefaultPredicate<T>(type);
                if (!Data.Conversion.Conversions.Instance.TryGetIsNAPredicate<T>(type, out _isMissing))
                    _isMissing = (in T value) => false;
            }

            public CountAggregator(VectorType type, ValueGetter<VBuffer<T>> getter)
            {
                Contracts.Assert(type.IsKnownSize);
                var size = type.Size;
                _count = new long[size];
                _fillBuffer = () => getter(ref _buffer);
                _isDefault = Data.Conversion.Conversions.Instance.GetIsDefaultPredicate<T>(type.ItemType);
                if (!Data.Conversion.Conversions.Instance.TryGetIsNAPredicate<T>(type.ItemType, out _isMissing))
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
