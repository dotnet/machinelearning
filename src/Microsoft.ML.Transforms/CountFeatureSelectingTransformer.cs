// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
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

        // TODO commenting
        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly long Count;

            public ColumnInfo(string input, string output, long count = Defaults.Count)
            {
                // TODO check values
                Input = input;
                Output = output;
                Count = count;
            }
        }

        // TODO commenting
        public CountFeatureSelectingEstimator(IHostEnvironment env, params ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _columns = columns;
        }

        // TODO commenting
        public CountFeatureSelectingEstimator(IHostEnvironment env, string input, string output = null, long count = Defaults.Count)
            : this(env, new ColumnInfo(input, output ?? input, count))
        {
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colPair in _columns)
            {
                if (!inputSchema.TryFindColumn(colPair.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.Input);
                // TODO check types work (right input)
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
                DropSlotsTransform.ColumnInfo[] dropSlotsColumns;
                (string input, string output)[] copyColumnsPairs;
                CreateDropAndCopyColumns(_columns, size, scores, out int[] selectedCount, out dropSlotsColumns, out copyColumnsPairs);

                for (int i = 0; i < selectedCount.Length; i++)
                    ch.Info(MessageSensitivity.Schema, "Selected {0} slots out of {1} in column '{2}'", selectedCount[i], colSizes[i], _columns[i].Input);
                ch.Info("Total number of slots selected: {0}", selectedCount.Sum());

                if (dropSlotsColumns.Length <= 0)
                    return new ColumnsCopyingTransformer(_host, copyColumnsPairs);
                else if (copyColumnsPairs.Length <= 0)
                    return new DropSlotsTransform(_host, dropSlotsColumns);

                var transformerChain = new TransformerChain<DropSlotsTransform>(
                    new ITransformer[] {
                        new ColumnsCopyingTransformer(_host, copyColumnsPairs),
                        new DropSlotsTransform(_host, dropSlotsColumns)
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

            var columnInfos = args.Column.Select(column => new ColumnInfo(column, column, args.Count)).ToArray();

            return new CountFeatureSelectingEstimator(env, columnInfos).Fit(input).Transform(input) as IDataTransform;
        }

        private static void CreateDropAndCopyColumns(ColumnInfo[] columnInfos, int size, long[][] scores,
            out int[] selectedCount, out DropSlotsTransform.ColumnInfo[] dropSlotsColumns, out (string input, string output)[] copyColumnsPairs)
        {
            Contracts.Assert(size > 0);
            Contracts.Assert(Utils.Size(scores) == size);
            Contracts.AssertValue(columnInfos);
            Contracts.Assert(Utils.Size(columnInfos) == size);

            selectedCount = new int[scores.Length];
            var dropSlotsCols = new List<DropSlotsTransform.ColumnInfo>();
            var copyCols = new List<(string input, string output)>();
            for (int i = 0; i < size; i++)
            {
                var slots = new List<(int min, int? max)>();
                var score = scores[i];
                selectedCount[i] = 0;
                for (int j = 0; j < score.Length; j++)
                {
                    if (score[j] < columnInfos[i].Count)
                    {
                        // Adjacent slots are combined into a single range.
                        int min = j;
                        while (j < score.Length && score[j] < columnInfos[i].Count)
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
                    dropSlotsCols.Add(new DropSlotsTransform.ColumnInfo(columnInfos[i].Input, columnInfos[i].Output, slots.ToArray()));
            }
            dropSlotsColumns = dropSlotsCols.ToArray();
            copyColumnsPairs = copyCols.ToArray();
        }
    }

    public static class CountFeatureSelectorExtensions
    {
        private sealed class OutPipelineColumn<T> : Vector<T>
        {
            public readonly Vector<T> Input;

            public OutPipelineColumn(Vector<T> input, long count)
                : base(new Reconciler<T>(count), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler<T> : EstimatorReconciler
        {
            private readonly long _count;

            public Reconciler(long count)
            {
                _count = count;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                env.Assert(toOutput.Length == 1);

                var infos = new CountFeatureSelectingEstimator.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; i++)
                    infos[i] = new CountFeatureSelectingEstimator.ColumnInfo(inputNames[((OutPipelineColumn<T>)toOutput[i]).Input], outputNames[toOutput[i]], _count);

                return new CountFeatureSelectingEstimator(env, infos);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static Vector<float> SelectFeaturesBasedOnCount(this Vector<float> input,
            long count = CountFeatureSelectingEstimator.Defaults.Count) => new OutPipelineColumn<float>(input, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static Vector<double> SelectFeaturesBasedOnCount(this Vector<double> input,
            long count = CountFeatureSelectingEstimator.Defaults.Count) => new OutPipelineColumn<double>(input, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static Vector<string> SelectFeaturesBasedOnCount(this Vector<string> input,
            long count = CountFeatureSelectingEstimator.Defaults.Count) => new OutPipelineColumn<string>(input, count);
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

        private static CountAggregator GetOneAggregator(IRow row, ColumnType colType, int colSrc)
        {
            Func<IRow, ColumnType, int, CountAggregator> del = GetOneAggregator<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(colType.RawType);
            return (CountAggregator)methodInfo.Invoke(null, new object[] { row, colType, colSrc });
        }

        private static CountAggregator GetOneAggregator<T>(IRow row, ColumnType colType, int colSrc)
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
                        _buffer.Values[0] = t;
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
