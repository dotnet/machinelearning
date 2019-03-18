// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(MutualInformationFeatureSelectingEstimator.Summary, typeof(IDataTransform), typeof(MutualInformationFeatureSelectingEstimator), typeof(MutualInformationFeatureSelectingEstimator.Options), typeof(SignatureDataTransform),
    MutualInformationFeatureSelectingEstimator.UserName, "MutualInformationFeatureSelection", "MutualInformationFeatureSelectionTransform", MutualInformationFeatureSelectingEstimator.ShortName)]

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
    public sealed class MutualInformationFeatureSelectingEstimator : IEstimator<ITransformer>
    {
        internal const string Summary =
            "Selects the top k slots across all specified columns ordered by their mutual information with the label column.";

        internal const string UserName = "Mutual Information Feature Selection Transform";
        internal const string ShortName = "MIFeatureSelection";
        internal static string RegistrationName = "MutualInformationFeatureSelectionTransform";

        [BestFriend]
        internal static class Defaults
        {
            public const string LabelColumn = DefaultColumnNames.Label;
            public const int SlotsInOutput = 1000;
            public const int NumBins = 256;
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to use for feature selection", Name = "Column", ShortName = "col", SortOrder = 1)]
            public string[] Columns;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for labels", ShortName = "lab",
                SortOrder = 4, Purpose = SpecialPurpose.ColumnName)]
            public string LabelColumn = Defaults.LabelColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The maximum number of slots to preserve in output", ShortName = "topk,numSlotsToKeep",
                SortOrder = 1)]
            public int SlotsInOutput = Defaults.SlotsInOutput;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of bins for R4/R8 columns, power of 2 recommended",
                ShortName = "bins")]
            public int NumBins = Defaults.NumBins;
        }

        private IHost _host;
        private readonly (string outputColumnName, string inputColumnName)[] _columns;
        private readonly string _labelColumn;
        private readonly int _slotsInOutput;
        private readonly int _numBins;

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numberOfBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <param name="columns">Specifies the names of the input columns for the transformation, and their respective output column names.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MutualInformationFeatureSelectingEstimator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        internal MutualInformationFeatureSelectingEstimator(IHostEnvironment env,
            string labelColumn = Defaults.LabelColumn,
            int slotsInOutput = Defaults.SlotsInOutput,
            int numberOfBins = Defaults.NumBins,
            params (string outputColumnName, string inputColumnName)[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);

            _host.CheckUserArg(Utils.Size(columns) > 0, nameof(columns));
            _host.CheckUserArg(slotsInOutput > 0, nameof(slotsInOutput));
            _host.CheckNonWhiteSpace(labelColumn, nameof(labelColumn));
            _host.Check(numberOfBins > 1, "numBins must be greater than 1.");

            _columns = columns;
            _labelColumn = labelColumn;
            _slotsInOutput = slotsInOutput;
            _numBins = numberOfBins;
        }

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="env">The environment to use.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="labelColumn">Name of the column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in the output. The number of slots to preserve is taken across all input columns.</param>
        /// <param name="numBins">Max number of bins used to approximate mutual information between each input column and the label column. Power of 2 recommended.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[MutualInformationFeatureSelectingEstimator](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/FeatureSelectionTransform.cs?range=1-4,10-121)]
        /// ]]>
        /// </format>
        /// </example>
        internal MutualInformationFeatureSelectingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            string labelColumn = Defaults.LabelColumn, int slotsInOutput = Defaults.SlotsInOutput, int numBins = Defaults.NumBins)
            : this(env, labelColumn, slotsInOutput, numBins, (outputColumnName, inputColumnName ?? outputColumnName))
        {
        }

        /// <summary>
        /// Trains and returns a <see cref="ITransformer"/>.
        /// </summary>
        public ITransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            using (var ch = _host.Start("Selecting Slots"))
            {
                ch.Info("Computing mutual information");
                var sw = new Stopwatch();
                sw.Start();
                var colSet = new HashSet<string>();
                foreach (var col in _columns)
                {
                    if (!colSet.Add(col.inputColumnName))
                        ch.Warning("Column '{0}' specified multiple time.", col);
                }
                var colArr = colSet.ToArray();
                var colSizes = new int[colArr.Length];
                var scores = MutualInformationFeatureSelectionUtils.TrainCore(_host, input, _labelColumn, colArr, _numBins, colSizes);
                sw.Stop();
                ch.Info("Finished mutual information computation in {0}", sw.Elapsed);

                ch.Info("Selecting features to drop");
                var threshold = ComputeThreshold(scores, _slotsInOutput, out int tiedScoresToKeep);

                // If no slots should be dropped in a column, use CopyColumn to generate the corresponding output column.
                SlotsDroppingTransformer.ColumnOptions[] dropSlotsColumns;
                (string outputColumnName, string inputColumnName)[] copyColumnPairs;
                CreateDropAndCopyColumns(colArr.Length, scores, threshold, tiedScoresToKeep, _columns.Where(col => colSet.Contains(col.inputColumnName)).ToArray(), out int[] selectedCount, out dropSlotsColumns, out copyColumnPairs);

                for (int i = 0; i < selectedCount.Length; i++)
                    ch.Info("Selected {0} slots out of {1} in column '{2}'", selectedCount[i], colSizes[i], colArr[i]);
                ch.Info("Total number of slots selected: {0}", selectedCount.Sum());

                if (dropSlotsColumns.Length <= 0)
                    return new ColumnCopyingTransformer(_host, copyColumnPairs);
                else if (copyColumnPairs.Length <= 0)
                    return new SlotsDroppingTransformer(_host, dropSlotsColumns);

                var transformerChain = new TransformerChain<SlotsDroppingTransformer>(
                    new ITransformer[] {
                        new ColumnCopyingTransformer(_host, copyColumnPairs),
                        new SlotsDroppingTransformer(_host, dropSlotsColumns)
                    });
                return transformerChain;
            }
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
                if (!inputSchema.TryFindColumn(colPair.inputColumnName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.inputColumnName);
                if (!MutualInformationFeatureSelectionUtils.IsValidColumnType(col.ItemType))
                    throw _host.ExceptUserArg(nameof(inputSchema),
                        "Column '{0}' does not have compatible type. Expected types are float, double, int, bool and key.", colPair.inputColumnName);
                var metadata = new List<SchemaShape.Column>();
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.CategoricalSlotRanges, out var categoricalSlotMeta))
                    metadata.Add(categoricalSlotMeta);
                metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
                result[colPair.outputColumnName] = new SchemaShape.Column(colPair.outputColumnName, col.Kind, col.ItemType, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
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
            host.CheckUserArg(options.SlotsInOutput > 0, nameof(options.SlotsInOutput));
            host.CheckNonWhiteSpace(options.LabelColumn, nameof(options.LabelColumn));
            host.Check(options.NumBins > 1, "numBins must be greater than 1.");

            (string outputColumnName, string inputColumnName)[] cols = options.Columns.Select(col => (col, col)).ToArray();
            return new MutualInformationFeatureSelectingEstimator(env, options.LabelColumn, options.SlotsInOutput, options.NumBins, cols).Fit(input).Transform(input) as IDataTransform;
        }

        /// <summary>
        /// Computes the threshold for the scores such that the top k slots are preserved.
        /// If there are less than k scores greater than zero, the threshold is set to zero and
        /// the tiedScoresToKeep is set to zero, so that we only keep scores strictly greater than zero.
        /// </summary>
        /// <param name="scores">The score for each column and each slot.</param>
        /// <param name="topk">How many slots to preserve.</param>
        /// <param name="tiedScoresToKeep">If there are ties, how many of them to keep.</param>
        /// <returns>The threshold.</returns>
        private static float ComputeThreshold(float[][] scores, int topk, out int tiedScoresToKeep)
        {
            // Use a min-heap for the topk elements.
            var heap = new Heap<float>((f1, f2) => f1 > f2, topk);

            for (int i = 0; i < scores.Length; i++)
            {
                for (int j = 0; j < scores[i].Length; j++)
                {
                    var score = scores[i][j];
                    Contracts.Assert(score >= 0);
                    if (heap.Count < topk)
                    {
                        if (score > 0)
                            heap.Add(score);
                    }
                    else if (heap.Top < score)
                    {
                        Contracts.Assert(heap.Count == topk);
                        heap.Pop();
                        heap.Add(score);
                    }
                }
            }

            var threshold = heap.Count < topk ? 0 : heap.Top;
            tiedScoresToKeep = 0;
            if (threshold == 0)
                return threshold;
            while (heap.Count > 0)
            {
                var top = heap.Pop();
                Contracts.Assert(top >= threshold);
                if (top > threshold)
                    break;
                tiedScoresToKeep++;
            }
            return threshold;
        }

        private static void CreateDropAndCopyColumns(int size, float[][] scores, float threshold, int tiedScoresToKeep, (string outputColumnName, string inputColumnName)[] cols,
            out int[] selectedCount, out SlotsDroppingTransformer.ColumnOptions[] dropSlotsColumns, out (string outputColumnName, string inputColumnName)[] copyColumnsPairs)
        {
            Contracts.Assert(size > 0);
            Contracts.Assert(Utils.Size(scores) == size);
            Contracts.Assert(Utils.Size(cols) == size);
            Contracts.Assert(threshold > 0 || (threshold == 0 && tiedScoresToKeep == 0));

            var dropCols = new List<SlotsDroppingTransformer.ColumnOptions>();
            var copyCols = new List<(string outputColumnName, string inputColumnName)>();
            selectedCount = new int[scores.Length];
            for (int i = 0; i < size; i++)
            {
                var slots = new List<(int min, int? max)>();
                var score = scores[i];
                selectedCount[i] = 0;
                for (int j = 0; j < score.Length; j++)
                {
                    var sc = score[j];
                    if (sc > threshold)
                    {
                        selectedCount[i]++;
                        continue;
                    }
                    if (sc == threshold && tiedScoresToKeep > 0)
                    {
                        tiedScoresToKeep--;
                        selectedCount[i]++;
                        continue;
                    }

                    // Adjacent slots are combined into a float range.
                    int min = j;
                    while (++j < score.Length)
                    {
                        sc = score[j];
                        if (sc > threshold)
                        {
                            selectedCount[i]++;
                            break;
                        }
                        if (sc == threshold && tiedScoresToKeep > 0)
                        {
                            tiedScoresToKeep--;
                            selectedCount[i]++;
                            break;
                        }
                    }
                    int max = j - 1;
                    slots.Add((min, max));
                }
                if (slots.Count <= 0)
                    copyCols.Add(cols[i]);
                else
                    dropCols.Add(new SlotsDroppingTransformer.ColumnOptions(cols[i].outputColumnName, cols[i].inputColumnName, slots.ToArray()));
            }
            dropSlotsColumns = dropCols.ToArray();
            copyColumnsPairs = copyCols.ToArray();
        }
    }

    internal static class MutualInformationFeatureSelectionUtils
    {
        /// <summary>
        /// Returns the feature selection scores for each slot of each column.
        /// </summary>
        /// <param name="host">The host.</param>
        /// <param name="input">The input dataview.</param>
        /// <param name="labelColumnName">The label column.</param>
        /// <param name="columns">The columns for which to compute the feature selection scores.</param>
        /// <param name="numBins">The number of bins to use for numeric features.</param>
        /// <param name="colSizes">The columns' sizes before dropping any slots.</param>
        /// <returns>A list of scores for each column and each slot.</returns>
        internal static float[][] TrainCore(IHost host, IDataView input, string labelColumnName, string[] columns, int numBins, int[] colSizes)
        {
            var impl = new Impl(host);
            return impl.GetScores(input, labelColumnName, columns, numBins, colSizes);
        }

        internal static bool IsValidColumnType(DataViewType type)
        {
            // REVIEW: Consider supporting all integer and unsigned types.
            ulong keyCount = type.GetKeyCount();
            return
                (0 < keyCount && keyCount < Utils.ArrayMaxSize) || type is BooleanDataViewType ||
                type == NumberDataViewType.Single || type == NumberDataViewType.Double || type == NumberDataViewType.Int32;
        }

        private sealed class Impl
        {
            private readonly IHost _host;
            private readonly BinFinderBase _binFinder;
            private int _numBins;
            private VBuffer<int> _labels; // always dense
            private int _numLabels;
            private int[][] _contingencyTable;
            private int[] _labelSums;
            private int[] _featureSums;
            private readonly List<float> _singles;
            private readonly List<double> _doubles;
            private ValueMapper<VBuffer<bool>, VBuffer<int>> _boolMapper;

            public Impl(IHost host)
            {
                Contracts.AssertValue(host);
                _host = host;
                _binFinder = new GreedyBinFinder();
                _singles = new List<float>();
                _doubles = new List<double>();
            }

            public float[][] GetScores(IDataView input, string labelColumnName, string[] columns, int numBins, int[] colSizes)
            {
                _numBins = numBins;
                var schema = input.Schema;
                var size = columns.Length;

                if (!schema.TryGetColumnIndex(labelColumnName, out int labelCol))
                {
                    throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectingEstimator.Options.LabelColumn),
                        "Label column '{0}' not found", labelColumnName);
                }

                var labelType = schema[labelCol].Type;
                if (!IsValidColumnType(labelType))
                {
                    throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectingEstimator.Options.LabelColumn),
                        "Label column '{0}' does not have compatible type", labelColumnName);
                }

                var colSrcs = new int[size + 1];
                colSrcs[size] = labelCol;
                for (int i = 0; i < size; i++)
                {
                    var colName = columns[i];
                    if (!schema.TryGetColumnIndex(colName, out int colSrc))
                    {
                        throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectingEstimator.Options.Columns),
                            "Source column '{0}' not found", colName);
                    }

                    var colType = schema[colSrc].Type;
                    if (colType is VectorType vectorType && !vectorType.IsKnownSize)
                    {
                        throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectingEstimator.Options.Columns),
                            "Variable length column '{0}' is not allowed", colName);
                    }

                    if (!IsValidColumnType(colType.GetItemType()))
                    {
                        throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectingEstimator.Options.Columns),
                            "Column '{0}' of type '{1}' does not have compatible type.", colName, colType);
                    }

                    colSrcs[i] = colSrc;
                    colSizes[i] = colType.GetValueCount();
                }

                var scores = new float[size][];
                using (var ch = _host.Start("Computing mutual information scores"))
                using (var pch = _host.StartProgressChannel("Computing mutual information scores"))
                {
                    using (var trans = Transposer.Create(_host, input, false, colSrcs))
                    {
                        int i = 0;
                        var header = new ProgressHeader(new[] { "columns" });
                        var b = trans.Schema.TryGetColumnIndex(labelColumnName, out labelCol);
                        Contracts.Assert(b);

                        GetLabels(trans, labelType, labelCol);
                        _contingencyTable = new int[_numLabels][];
                        _labelSums = new int[_numLabels];
                        pch.SetHeader(header, e => e.SetProgress(0, i, size));
                        for (i = 0; i < size; i++)
                        {
                            b = trans.Schema.TryGetColumnIndex(columns[i], out int col);
                            Contracts.Assert(b);
                            ch.Trace("Computing scores for column '{0}'", columns[i]);
                            scores[i] = ComputeMutualInformation(trans, col);
#if DEBUG
                            ch.Trace("Scores for column '{0}': {1}", columns[i], string.Join(", ", scores[i]));
#endif
                            pch.Checkpoint(i + 1);
                        }
                    }
                }

                return scores;
            }

            private void GetLabels(Transposer trans, DataViewType labelType, int labelCol)
            {
                int min;
                int lim;
                var labels = default(VBuffer<int>);
                // Note: NAs have their own separate bin.
                if (labelType == NumberDataViewType.Int32)
                {
                    var tmp = default(VBuffer<int>);
                    trans.GetSingleSlotValue(labelCol, ref tmp);
                    BinInts(in tmp, ref labels, _numBins, out min, out lim);
                    _numLabels = lim - min;
                }
                else if (labelType == NumberDataViewType.Single)
                {
                    var tmp = default(VBuffer<Single>);
                    trans.GetSingleSlotValue(labelCol, ref tmp);
                    BinSingles(in tmp, ref labels, _numBins, out min, out lim);
                    _numLabels = lim - min;
                }
                else if (labelType == NumberDataViewType.Double)
                {
                    var tmp = default(VBuffer<Double>);
                    trans.GetSingleSlotValue(labelCol, ref tmp);
                    BinDoubles(in tmp, ref labels, _numBins, out min, out lim);
                    _numLabels = lim - min;
                }
                else if (labelType is BooleanDataViewType)
                {
                    var tmp = default(VBuffer<bool>);
                    trans.GetSingleSlotValue(labelCol, ref tmp);
                    BinBools(in tmp, ref labels);
                    _numLabels = 3;
                    min = -1;
                    lim = 2;
                }
                else
                {
                    ulong labelKeyCount = labelType.GetKeyCount();
                    Contracts.Assert(labelKeyCount < Utils.ArrayMaxSize);
                    KeyLabelGetter<int> del = GetKeyLabels<int>;
                    var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(labelType.RawType);
                    var parameters = new object[] { trans, labelCol, labelType };
                    _labels = (VBuffer<int>)methodInfo.Invoke(this, parameters);
                    _numLabels = labelType.GetKeyCountAsInt32(_host) + 1;

                    // No need to densify or shift in this case.
                    return;
                }

                // Densify and shift labels.
                VBufferUtils.Densify(ref labels);
                Contracts.Assert(labels.IsDense);
                var labelsEditor = VBufferEditor.CreateFromBuffer(ref labels);
                for (int i = 0; i < labels.Length; i++)
                {
                    labelsEditor.Values[i] -= min;
                    Contracts.Assert(labelsEditor.Values[i] < _numLabels);
                }
                _labels = labelsEditor.Commit();
            }

            private delegate VBuffer<int> KeyLabelGetter<T>(Transposer trans, int labelCol, DataViewType labeColumnType);

            private VBuffer<int> GetKeyLabels<T>(Transposer trans, int labelCol, DataViewType labelColumnType)
            {
                var tmp = default(VBuffer<T>);
                var labels = default(VBuffer<int>);
                trans.GetSingleSlotValue(labelCol, ref tmp);
                BinKeys<T>(labelColumnType)(in tmp, ref labels);
                VBufferUtils.Densify(ref labels);
                return labels;
            }

            /// <summary>
            /// Computes the mutual information for one column.
            /// </summary>
            private Single[] ComputeMutualInformation(Transposer trans, int col)
            {
                // Note: NAs have their own separate bin.
                var type = trans.Schema[col].Type;
                var itemType = type.GetItemType();
                if (itemType == NumberDataViewType.Int32)
                {
                    return ComputeMutualInformation(trans, col,
                        (ref VBuffer<int> src, ref VBuffer<int> dst, out int min, out int lim) =>
                        {
                            BinInts(in src, ref dst, _numBins, out min, out lim);
                        });
                }
                if (itemType == NumberDataViewType.Single)
                {
                    return ComputeMutualInformation(trans, col,
                        (ref VBuffer<Single> src, ref VBuffer<int> dst, out int min, out int lim) =>
                        {
                            BinSingles(in src, ref dst, _numBins, out min, out lim);
                        });
                }
                if (itemType == NumberDataViewType.Double)
                {
                    return ComputeMutualInformation(trans, col,
                        (ref VBuffer<Double> src, ref VBuffer<int> dst, out int min, out int lim) =>
                        {
                            BinDoubles(in src, ref dst, _numBins, out min, out lim);
                        });
                }
                if (itemType is BooleanDataViewType)
                {
                    return ComputeMutualInformation(trans, col,
                        (ref VBuffer<bool> src, ref VBuffer<int> dst, out int min, out int lim) =>
                        {
                            min = -1;
                            lim = 2;
                            BinBools(in src, ref dst);
                        });
                }
                ulong keyCount = itemType.GetKeyCount();
                Contracts.Assert(keyCount < Utils.ArrayMaxSize);
                Func<DataViewType, Mapper<int>> del = MakeKeyMapper<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(itemType.RawType);
                ComputeMutualInformationDelegate<int> cmiDel = ComputeMutualInformation;
                var cmiMethodInfo = cmiDel.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(itemType.RawType);
                return (Single[])cmiMethodInfo.Invoke(this, new object[] { trans, col, methodInfo.Invoke(null, new object[] { itemType }) });
            }

            private delegate float[] ComputeMutualInformationDelegate<T>(Transposer trans, int col, Mapper<T> mapper);

            private delegate void Mapper<T>(ref VBuffer<T> src, ref VBuffer<int> dst, out int min, out int lim);

            private static Mapper<T> MakeKeyMapper<T>(DataViewType type)
            {
                ulong keyCount = type.GetKeyCount();
                Contracts.Assert(0 < keyCount && keyCount < Utils.ArrayMaxSize);
                var mapper = BinKeys<T>(type);
                return
                    (ref VBuffer<T> src, ref VBuffer<int> dst, out int min, out int lim) =>
                    {
                        min = 0;
                        lim = (int)type.GetKeyCount() + 1;
                        mapper(in src, ref dst);
                    };
            }

            /// <summary>
            /// Computes the mutual information for one column.
            /// </summary>
            private float[] ComputeMutualInformation<T>(Transposer trans, int col, Mapper<T> mapper)
            {
                var slotCount = trans.Schema[col].Type.GetValueCount();
                var scores = new float[slotCount];
                int iScore = 0;
                VBuffer<int> slotValues = default(VBuffer<int>);
                using (var cursor = trans.GetSlotCursor(col))
                {
                    var getter = cursor.GetGetter<T>();
                    while (cursor.MoveNext())
                    {
                        VBuffer<T> tmp = default(VBuffer<T>);
                        getter(ref tmp);
                        mapper(ref tmp, ref slotValues, out int min, out int lim);
                        Contracts.Assert(iScore < slotCount);
                        scores[iScore++] = ComputeMutualInformation(in slotValues, lim - min, min);
                    }
                }
                return scores;
            }

            /// <summary>
            /// Computes the mutual information for one slot.
            /// </summary>
            private float ComputeMutualInformation(in VBuffer<int> features, int numFeatures, int offset)
            {
                Contracts.Assert(_labels.Length == features.Length);
                if (Utils.Size(_contingencyTable[0]) < numFeatures)
                {
                    for (int i = 0; i < _numLabels; i++)
                        Array.Resize(ref _contingencyTable[i], numFeatures);
                    Array.Resize(ref _featureSums, numFeatures);
                }
                for (int i = 0; i < _numLabels; i++)
                    Array.Clear(_contingencyTable[i], 0, numFeatures);
                Array.Clear(_labelSums, 0, _numLabels);
                Array.Clear(_featureSums, 0, numFeatures);

                FillTable(in features, offset, numFeatures);
                for (int i = 0; i < _numLabels; i++)
                {
                    for (int j = 0; j < numFeatures; j++)
                    {
                        _labelSums[i] += _contingencyTable[i][j];
                        _featureSums[j] += _contingencyTable[i][j];
                    }
                }

                double score = 0;
                for (int i = 0; i < _numLabels; i++)
                {
                    for (int j = 0; j < numFeatures; j++)
                    {
                        if (_contingencyTable[i][j] > 0)
                            score += _contingencyTable[i][j] / (double)_labels.Length * Math.Log(_contingencyTable[i][j] * (double)_labels.Length / ((double)_labelSums[i] * _featureSums[j]), 2);
                    }
                }

                Contracts.Assert(score >= 0);
                return (float)score;
            }

            /// <summary>
            /// Fills the contingency table.
            /// </summary>
            private void FillTable(in VBuffer<int> features, int offset, int numFeatures)
            {
                Contracts.Assert(_labels.IsDense);
                Contracts.Assert(_labels.Length == features.Length);
                var featureValues = features.GetValues();
                var labelsValues = _labels.GetValues();
                if (features.IsDense)
                {
                    for (int i = 0; i < labelsValues.Length; i++)
                    {
                        var label = labelsValues[i];
                        var feature = featureValues[i] - offset;
                        Contracts.Assert(0 <= label && label < _numLabels);
                        Contracts.Assert(0 <= feature && feature < numFeatures);
                        _contingencyTable[label][feature]++;
                    }
                    return;
                }

                var featureIndices = features.GetIndices();
                int ii = 0;
                for (int i = 0; i < labelsValues.Length; i++)
                {
                    var label = labelsValues[i];
                    int feature;
                    if (ii == featureIndices.Length || i < featureIndices[ii])
                        feature = -offset;
                    else
                    {
                        feature = featureValues[ii] - offset;
                        ii++;
                    }
                    Contracts.Assert(0 <= label && label < _numLabels);
                    Contracts.Assert(0 <= feature && feature < numFeatures);
                    _contingencyTable[label][feature]++;
                }
                Contracts.Assert(ii == featureIndices.Length);
            }

            /// <summary>
            /// Maps from keys to ints.
            /// </summary>
            private static ValueMapper<VBuffer<T>, VBuffer<int>> BinKeys<T>(DataViewType colType)
            {
                var conv = Data.Conversion.Conversions.Instance.GetStandardConversion<T, uint>(colType, NumberDataViewType.UInt32, out bool identity);
                ValueMapper<T, int> mapper;
                if (identity)
                {
                    mapper = (ValueMapper<T, int>)(Delegate)(ValueMapper<uint, int>)(
                        (in uint src, ref int dst) =>
                        {
                            dst = (int)src;
                        });
                }
                else
                {
                    mapper =
                        (in T src, ref int dst) =>
                        {
                            uint t = 0;
                            conv(in src, ref t);
                            dst = (int)t;
                        };
                }
                return CreateVectorMapper(mapper);
            }

            /// <summary>
            /// Maps Ints.
            /// </summary>
            private void BinInts(in VBuffer<int> input, ref VBuffer<int> output,
                int numBins, out int min, out int lim)
            {
                Contracts.Assert(_singles.Count == 0);

                var bounds = _binFinder.FindBins(numBins, _singles, input.Length - input.GetValues().Length);
                min = -1 - bounds.FindIndexSorted(0);
                lim = min + bounds.Length + 1;
                int offset = min;
                ValueMapper<int, int> mapper =
                    (in int src, ref int dst) =>
                        dst = offset + 1 + bounds.FindIndexSorted((Single)src);
                mapper.MapVector(in input, ref output);
                _singles.Clear();
            }

            /// <summary>
            /// Maps from Singles to ints. NaNs (and only NaNs) are mapped to the first bin.
            /// </summary>
            private void BinSingles(in VBuffer<Single> input, ref VBuffer<int> output,
                int numBins, out int min, out int lim)
            {
                Contracts.Assert(_singles.Count == 0);
                var inputValues = input.GetValues();
                for (int i = 0; i < inputValues.Length; i++)
                {
                    var val = inputValues[i];
                    if (!Single.IsNaN(val))
                        _singles.Add(val);
                }

                var bounds = _binFinder.FindBins(numBins, _singles, input.Length - inputValues.Length);
                min = -1 - bounds.FindIndexSorted(0);
                lim = min + bounds.Length + 1;
                int offset = min;
                ValueMapper<Single, int> mapper =
                    (in Single src, ref int dst) =>
                        dst = Single.IsNaN(src) ? offset : offset + 1 + bounds.FindIndexSorted(src);
                mapper.MapVector(in input, ref output);
                _singles.Clear();
            }

            /// <summary>
            /// Maps from Doubles to ints. NaNs (and only NaNs) are mapped to the first bin.
            /// </summary>
            private void BinDoubles(in VBuffer<Double> input, ref VBuffer<int> output,
                int numBins, out int min, out int lim)
            {
                Contracts.Assert(_doubles.Count == 0);
                var inputValues = input.GetValues();
                for (int i = 0; i < inputValues.Length; i++)
                {
                    var val = inputValues[i];
                    if (!Double.IsNaN(val))
                        _doubles.Add(val);
                }

                var bounds = _binFinder.FindBins(numBins, _doubles, input.Length - inputValues.Length);
                var offset = min = -1 - bounds.FindIndexSorted(0);
                lim = min + bounds.Length + 1;
                ValueMapper<Double, int> mapper =
                    (in Double src, ref int dst) =>
                        dst = Double.IsNaN(src) ? offset : offset + 1 + bounds.FindIndexSorted(src);
                mapper.MapVector(in input, ref output);
                _doubles.Clear();
            }

            private void BinBools(in VBuffer<bool> input, ref VBuffer<int> output)
            {
                if (_boolMapper == null)
                    _boolMapper = CreateVectorMapper<bool, int>(BinOneBool);
                _boolMapper(in input, ref output);
            }

            private void BinOneBool(in bool src, ref int dst)
            {
                dst = Convert.ToInt32(src);
            }
        }

        /// <summary>
        /// Given a mapper from T to int, creates a mapper from VBuffer{T} to VBuffer&lt;int&gt;.
        /// Assumes that the mapper maps default(TSrc) to default(TDst) so that the returned mapper preserves sparsity.
        /// </summary>
        private static ValueMapper<VBuffer<TSrc>, VBuffer<TDst>> CreateVectorMapper<TSrc, TDst>(ValueMapper<TSrc, TDst> map)
            where TDst : IEquatable<TDst>
        {
#if DEBUG
            TSrc tmpSrc = default(TSrc);
            TDst tmpDst = default(TDst);
            map(in tmpSrc, ref tmpDst);
            Contracts.Assert(tmpDst.Equals(default(TDst)));
#endif
            return map.MapVector;
        }

        private static void MapVector<TSrc, TDst>(this ValueMapper<TSrc, TDst> map, in VBuffer<TSrc> input, ref VBuffer<TDst> output)
        {
            var inputValues = input.GetValues();
            var editor = VBufferEditor.Create(ref output, input.Length, inputValues.Length);
            for (int i = 0; i < inputValues.Length; i++)
            {
                TSrc val = inputValues[i];
                map(in val, ref editor.Values[i]);
            }

            if (!input.IsDense && inputValues.Length > 0)
            {
                input.GetIndices().CopyTo(editor.Indices);
            }

            output = editor.Commit();
        }
    }
}
