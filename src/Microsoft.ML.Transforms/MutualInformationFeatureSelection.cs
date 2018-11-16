// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

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
using System.Diagnostics;
using System.Linq;
using System.Reflection;

[assembly: LoadableClass(MutualInformationFeatureSelectionEstimator.Summary, typeof(IDataTransform), typeof(MutualInformationFeatureSelectionEstimator), typeof(MutualInformationFeatureSelectionEstimator.Arguments), typeof(SignatureDataTransform),
    MutualInformationFeatureSelectionEstimator.UserName, "MutualInformationFeatureSelection", "MutualInformationFeatureSelectionTransform", MutualInformationFeatureSelectionEstimator.ShortName)]

namespace Microsoft.ML.Transforms.FeatureSelection
{
    /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
    public sealed class MutualInformationFeatureSelectionEstimator : IEstimator<ITransformer>
    {
        internal const string Summary =
            "Selects the top k slots across all specified columns ordered by their mutual information with the label column.";

        internal const string UserName = "Mutual Information Feature Selection Transform";
        internal const string ShortName = "MIFeatureSelection";
        internal const string FriendlyName = "Mutual Information Feature Selection";
        internal static string RegistrationName = "MutualInformationFeatureSelectionTransform";

        public static class Defaults
        {
            public const string LabelColumn = DefaultColumnNames.Label;
            public const int SlotsInOutput = 1000;
            public const int NumBins = 256;
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Columns to use for feature selection", ShortName = "col",
                SortOrder = 1)]
            public string[] Column;

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

        //// TODO commenting
        //public sealed class ColumnInfo
        //{
        //    public readonly string Input;
        //    public readonly string Output;
        //    public readonly string LabelColumn;
        //    public readonly int SlotsInOutput;
        //    public readonly int NumBins;

        //    public ColumnInfo(string input, string output = null, string labelColumn = Defaults.LabelColumn,
        //        int slotsInOutput = Defaults.SlotsInOutput, int numBins = Defaults.NumBins)
        //    {
        //        // TODO check values
        //        Input = input;
        //        Output = output ?? input;
        //        LabelColumn = labelColumn;
        //        SlotsInOutput = slotsInOutput;
        //        NumBins = numBins;
        //    }
        //}

        private IHost _host;
        private readonly (string input, string output)[] _columns;
        private readonly string _labelColumn;
        private readonly int _slotsInOutput;
        private readonly int _numBins;

        public MutualInformationFeatureSelectionEstimator(IHostEnvironment env,
            string labelColumn = Defaults.LabelColumn,
            int slotsInOutput = Defaults.SlotsInOutput,
            int numBins = Defaults.NumBins,
            params(string input, string output)[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _columns = columns;
            _labelColumn = labelColumn;
            _slotsInOutput = slotsInOutput;
            _numBins = numBins;
        }

        public MutualInformationFeatureSelectionEstimator(IHostEnvironment env, string input, string output = null,
            string labelColumn = Defaults.LabelColumn, int slotsInOutput = Defaults.SlotsInOutput, int numBins = Defaults.NumBins)
            : this(env, labelColumn, slotsInOutput, numBins, (input, output ?? input))
        {
        }

        public ITransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            //host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));
            //host.CheckUserArg(args.SlotsInOutput > 0, nameof(args.SlotsInOutput));
            //host.CheckNonWhiteSpace(args.LabelColumn, nameof(args.LabelColumn));
            //host.Check(args.NumBins > 1, "numBins must be greater than 1.");

            using (var ch = _host.Start("Selecting Slots"))
            {
                ch.Info("Computing mutual information");
                var sw = new Stopwatch();
                sw.Start();
                var colSet = new HashSet<string>();
                foreach (var col in _columns)
                {
                    if (!colSet.Add(col.input))
                        ch.Warning("Column '{0}' specified multiple time.", col);
                }
                var colArr = colSet.ToArray();
                var colSizes = new int[colArr.Length];
                var scores = MutualInformationFeatureSelectionUtils.TrainCore(_host, input, _labelColumn, colArr, _numBins, colSizes);
                sw.Stop();
                ch.Info("Finished mutual information computation in {0}", sw.Elapsed);

                ch.Info("Selecting features to drop");
                var threshold = ComputeThreshold(scores, _slotsInOutput, out int tiedScoresToKeep);

                var columns = CreateDropSlotsColumns(colArr.Length, scores, threshold, tiedScoresToKeep, out int[] selectedCount, _columns);

                if (columns.Count <= 0)
                {
                    ch.Info("No features are being dropped.");
                    //return new NopTransform(_host);
                }

                for (int i = 0; i < selectedCount.Length; i++)
                    ch.Info("Selected {0} slots out of {1} in column '{2}'", selectedCount[i], colSizes[i], colArr[i]);
                ch.Info("Total number of slots selected: {0}", selectedCount.Sum());

                //var dsArgs = new DropSlotsTransform.Arguments();
                //dsArgs.Column = columns.ToArray();
                return new DropSlotsTransform(_host, columns.ToArray());
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colPair in _columns)
            {
                if (!inputSchema.TryFindColumn(colPair.input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colPair.input);
                // TODO check types work (right input)
                var metadata = new List<SchemaShape.Column>();
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    metadata.Add(slotMeta);
                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.CategoricalSlotRanges, out var categoricalSlotMeta))
                    metadata.Add(categoricalSlotMeta);
                metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));
                result[colPair.output] = new SchemaShape.Column(colPair.output, col.Kind, col.ItemType, false, new SchemaShape(metadata.ToArray()));
            }
            return new SchemaShape(result.Values);
        }

        ///// <summary>
        ///// A helper method to create <see cref="IDataTransform"/> for selecting the top k slots ordered by their mutual information.
        ///// </summary>
        ///// <param name="env">Host Environment.</param>
        ///// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        ///// <param name="labelColumn">Column to use for labels.</param>
        ///// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        ///// <param name="numBins">Max number of bins for R4/R8 columns, power of 2 recommended.</param>
        ///// <param name="columns">Columns to use for feature selection.</param>
        //internal static IDataTransform Create(IHostEnvironment env,
        //    IDataView input,
        //    string labelColumn = Defaults.LabelColumn,
        //    int slotsInOutput = Defaults.SlotsInOutput,
        //    int numBins = Defaults.NumBins,
        //    params string[] columns)
        //{
        //    var args = new Arguments()
        //    {
        //        Column = columns,
        //        LabelColumn = labelColumn,
        //        SlotsInOutput = slotsInOutput,
        //        NumBins = numBins
        //    };
        //    return Create(env, args, input);
        //}

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
            host.CheckUserArg(args.SlotsInOutput > 0, nameof(args.SlotsInOutput));
            host.CheckNonWhiteSpace(args.LabelColumn, nameof(args.LabelColumn));
            host.Check(args.NumBins > 1, "numBins must be greater than 1.");

            (string input, string output)[] cols = args.Column.Select(col => (col, col)).ToArray();
            return new MutualInformationFeatureSelectionEstimator(env, args.LabelColumn, args.SlotsInOutput, args.NumBins, cols).Fit(input).Transform(input) as IDataTransform;
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

        private static List<DropSlotsTransform.ColumnInfo> CreateDropSlotsColumns(int size, float[][] scores,
            float threshold, int tiedScoresToKeep, out int[] selectedCount, (string input, string output)[] cols)
        {
            Contracts.Assert(size > 0);
            Contracts.Assert(Utils.Size(scores) == size);
            Contracts.Assert(Utils.Size(cols) == size);
            Contracts.Assert(threshold > 0 || (threshold == 0 && tiedScoresToKeep == 0));

            var columns = new List<DropSlotsTransform.ColumnInfo>();
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
                if (slots.Count > 0)
                {
                    var col = new DropSlotsTransform.ColumnInfo(cols[i].input, cols[i].output, slots.ToArray());
                    columns.Add(col);
                }
            }
            return columns;
        }
    }

    // TODO commenting
    public static class MutualInformationFeatureSelectorExtensions
    {
        private sealed class OutPipelineColumn<T> : Vector<T>
        {
            public readonly Vector<T> Input;
            public readonly PipelineColumn LabelColumn;

            public OutPipelineColumn(Vector<T> input, Scalar<float> labelColumn, int slotsInOutput, int numBins)
                : base(new Reconciler<T>(labelColumn, slotsInOutput, numBins), input, labelColumn)
            {
                Input = input;
                LabelColumn = labelColumn;
            }

            public OutPipelineColumn(Vector<T> input, Scalar<bool> labelColumn, int slotsInOutput, int numBins)
               : base(new Reconciler<T>(labelColumn, slotsInOutput, numBins), input, labelColumn)
            {
                Input = input;
                LabelColumn = labelColumn;
            }
        }

        private sealed class Reconciler<T> : EstimatorReconciler
        {
            private readonly PipelineColumn _labelColumn;
            private readonly int _slotsInOutput;
            private readonly int _numBins;

            public Reconciler(PipelineColumn labelColumn, int slotsInOutput, int numBins)
            {
                _labelColumn = labelColumn;
                _slotsInOutput = slotsInOutput;
                _numBins = numBins;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn<T>)outCol).Input], outputNames[outCol]));

                return new MutualInformationFeatureSelectionEstimator(env,inputNames[_labelColumn], _slotsInOutput, _numBins, pairs.ToArray());
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for float/double columns, power of 2 recommended.</param>
        public static Vector<float> SelectFeaturesBasedOnMutualInformation(
            this Vector<float> input,
            Scalar<bool> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectionEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectionEstimator.Defaults.NumBins) => new OutPipelineColumn<float>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for float/double columns, power of 2 recommended.</param>
        public static Vector<float> SelectFeaturesBasedOnMutualInformation(
            this Vector<float> input,
            Scalar<float> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectionEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectionEstimator.Defaults.NumBins) => new OutPipelineColumn<float>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for float/double columns, power of 2 recommended.</param>
        public static Vector<double> SelectFeaturesBasedOnMutualInformation(
            this Vector<double> input,
            Scalar<bool> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectionEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectionEstimator.Defaults.NumBins) => new OutPipelineColumn<double>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for float/double columns, power of 2 recommended.</param>
        public static Vector<double> SelectFeaturesBasedOnMutualInformation(
            this Vector<double> input,
            Scalar<float> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectionEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectionEstimator.Defaults.NumBins) => new OutPipelineColumn<double>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for float/double columns, power of 2 recommended.</param>
        public static Vector<bool> SelectFeaturesBasedOnMutualInformation(
            this Vector<bool> input,
            Scalar<bool> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectionEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectionEstimator.Defaults.NumBins) => new OutPipelineColumn<bool>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for float/double columns, power of 2 recommended.</param>
        public static Vector<bool> SelectFeaturesBasedOnMutualInformation(
            this Vector<bool> input,
            Scalar<float> labelColumn,
            int slotsInOutput = MutualInformationFeatureSelectionEstimator.Defaults.SlotsInOutput,
            int numBins = MutualInformationFeatureSelectionEstimator.Defaults.NumBins) => new OutPipelineColumn<bool>(input, labelColumn, slotsInOutput, numBins);
    }

    public static class MutualInformationFeatureSelectionUtils
    {
        /// <summary>
        /// Returns the feature selection scores for each slot of each column.
        /// </summary>
        /// <param name="host">The host.</param>
        /// <param name="input">The input dataview.</param>
        /// <param name="labelColumnName">The label column.</param>
        /// <param name="columns">The columns for which to compute the feature selection scores.</param>
        /// <param name="numBins">The number of bins to use for numeric features.</param>
        /// <returns>A list of scores for each column and each slot.</returns>
        public static float[][] Train(IHost host, IDataView input, string labelColumnName, string[] columns, int numBins)
        {
            Contracts.CheckValue(host, nameof(host));
            host.CheckValue(input, nameof(input));
            host.CheckNonWhiteSpace(labelColumnName, nameof(labelColumnName));
            host.CheckValue(columns, nameof(columns));
            host.Check(columns.Length > 0, "At least one column must be specified.");
            host.Check(numBins > 1, "numBins must be greater than 1.");

            HashSet<string> colSet = new HashSet<string>();
            foreach (string col in columns)
            {
                if (!colSet.Add(col))
                    throw host.Except("Column '{0}' specified multiple times.", col);
            }

            var colSizes = new int[columns.Length];
            return TrainCore(host, input, labelColumnName, columns, numBins, colSizes);
        }

        internal static float[][] TrainCore(IHost host, IDataView input, string labelColumnName, string[] columns, int numBins, int[] colSizes)
        {
            var impl = new Impl(host);
            return impl.GetScores(input, labelColumnName, columns, numBins, colSizes);
        }

        private sealed class Impl
        {
            private readonly IHost _host;
            private readonly BinFinderBase _binFinder;
            private int _numBins;
            private int[] _labels;
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
                    throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectionEstimator.Arguments.LabelColumn),
                        "Label column '{0}' not found", labelColumnName);
                }

                var labelType = schema.GetColumnType(labelCol);
                if (!IsValidColumnType(labelType))
                {
                    throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectionEstimator.Arguments.LabelColumn),
                        "Label column '{0}' does not have compatible type", labelColumnName);
                }

                var colSrcs = new int[size + 1];
                colSrcs[size] = labelCol;
                for (int i = 0; i < size; i++)
                {
                    var colName = columns[i];
                    if (!schema.TryGetColumnIndex(colName, out int colSrc))
                    {
                        throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectionEstimator.Arguments.Column),
                            "Source column '{0}' not found", colName);
                    }

                    var colType = schema.GetColumnType(colSrc);
                    if (colType.IsVector && !colType.IsKnownSizeVector)
                    {
                        throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectionEstimator.Arguments.Column),
                            "Variable length column '{0}' is not allowed", colName);
                    }

                    if (!IsValidColumnType(colType.ItemType))
                    {
                        throw _host.ExceptUserArg(nameof(MutualInformationFeatureSelectionEstimator.Arguments.Column),
                            "Column '{0}' of type '{1}' does not have compatible type.", colName, colType);
                    }

                    colSrcs[i] = colSrc;
                    colSizes[i] = colType.ValueCount;
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

            private static bool IsValidColumnType(ColumnType type)
            {
                // REVIEW: Consider supporting all integer and unsigned types.
                return
                    (0 < type.KeyCount && type.KeyCount < Utils.ArrayMaxSize) || type.IsBool ||
                    type == NumberType.R4 || type == NumberType.R8 || type == NumberType.I4;
            }

            private void GetLabels(Transposer trans, ColumnType labelType, int labelCol)
            {
                int min;
                int lim;
                var labels = default(VBuffer<int>);
                // Note: NAs have their own separate bin.
                if (labelType == NumberType.I4)
                {
                    var tmp = default(VBuffer<int>);
                    trans.GetSingleSlotValue(labelCol, ref tmp);
                    BinInts(ref tmp, ref labels, _numBins, out min, out lim);
                    _numLabels = lim - min;
                }
                else if (labelType == NumberType.R4)
                {
                    var tmp = default(VBuffer<float>);
                    trans.GetSingleSlotValue(labelCol, ref tmp);
                    BinSingles(ref tmp, ref labels, _numBins, out min, out lim);
                    _numLabels = lim - min;
                }
                else if (labelType == NumberType.R8)
                {
                    var tmp = default(VBuffer<double>);
                    trans.GetSingleSlotValue(labelCol, ref tmp);
                    BinDoubles(ref tmp, ref labels, _numBins, out min, out lim);
                    _numLabels = lim - min;
                }
                else if (labelType.IsBool)
                {
                    var tmp = default(VBuffer<bool>);
                    trans.GetSingleSlotValue(labelCol, ref tmp);
                    BinBools(ref tmp, ref labels);
                    _numLabels = 3;
                    min = -1;
                    lim = 2;
                }
                else
                {
                    Contracts.Assert(0 < labelType.KeyCount && labelType.KeyCount < Utils.ArrayMaxSize);
                    KeyLabelGetter<int> del = GetKeyLabels<int>;
                    var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(labelType.RawType);
                    var parameters = new object[] { trans, labelCol, labelType };
                    _labels = (int[])methodInfo.Invoke(this, parameters);
                    _numLabels = labelType.KeyCount + 1;

                    // No need to densify or shift in this case.
                    return;
                }

                // Densify and shift labels.
                VBufferUtils.Densify(ref labels);
                Contracts.Assert(labels.IsDense);
                _labels = labels.Values;
                if (labels.Length < _labels.Length)
                    Array.Resize(ref _labels, labels.Length);
                for (int i = 0; i < _labels.Length; i++)
                {
                    _labels[i] -= min;
                    Contracts.Assert(_labels[i] < _numLabels);
                }
            }

            private delegate int[] KeyLabelGetter<T>(Transposer trans, int labelCol, ColumnType labeColumnType);

            private int[] GetKeyLabels<T>(Transposer trans, int labelCol, ColumnType labeColumnType)
            {
                var tmp = default(VBuffer<T>);
                var labels = default(VBuffer<int>);
                trans.GetSingleSlotValue(labelCol, ref tmp);
                BinKeys<T>(labeColumnType)(in tmp, ref labels);
                VBufferUtils.Densify(ref labels);
                var values = labels.Values;
                if (labels.Length < values.Length)
                    Array.Resize(ref values, labels.Length);
                return values;
            }

            /// <summary>
            /// Computes the mutual information for one column.
            /// </summary>
            private float[] ComputeMutualInformation(Transposer trans, int col)
            {
                // Note: NAs have their own separate bin.
                var type = trans.Schema.GetColumnType(col);
                if (type.ItemType == NumberType.I4)
                {
                    return ComputeMutualInformation(trans, col,
                        (ref VBuffer<int> src, ref VBuffer<int> dst, out int min, out int lim) =>
                        {
                            BinInts(ref src, ref dst, _numBins, out min, out lim);
                        });
                }
                if (type.ItemType == NumberType.R4)
                {
                    return ComputeMutualInformation(trans, col,
                        (ref VBuffer<float> src, ref VBuffer<int> dst, out int min, out int lim) =>
                        {
                            BinSingles(ref src, ref dst, _numBins, out min, out lim);
                        });
                }
                if (type.ItemType == NumberType.R8)
                {
                    return ComputeMutualInformation(trans, col,
                        (ref VBuffer<double> src, ref VBuffer<int> dst, out int min, out int lim) =>
                        {
                            BinDoubles(ref src, ref dst, _numBins, out min, out lim);
                        });
                }
                if (type.ItemType.IsBool)
                {
                    return ComputeMutualInformation(trans, col,
                        (ref VBuffer<bool> src, ref VBuffer<int> dst, out int min, out int lim) =>
                        {
                            min = -1;
                            lim = 2;
                            BinBools(ref src, ref dst);
                        });
                }
                Contracts.Assert(0 < type.ItemType.KeyCount && type.ItemType.KeyCount < Utils.ArrayMaxSize);
                Func<ColumnType, Mapper<int>> del = MakeKeyMapper<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.ItemType.RawType);
                ComputeMutualInformationDelegate<int> cmiDel = ComputeMutualInformation;
                var cmiMethodInfo = cmiDel.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(type.ItemType.RawType);
                return (float[])cmiMethodInfo.Invoke(this, new object[] { trans, col, methodInfo.Invoke(null, new object[] { type.ItemType }) });
            }

            private delegate float[] ComputeMutualInformationDelegate<T>(Transposer trans, int col, Mapper<T> mapper);

            private delegate void Mapper<T>(ref VBuffer<T> src, ref VBuffer<int> dst, out int min, out int lim);

            private static Mapper<T> MakeKeyMapper<T>(ColumnType type)
            {
                Contracts.Assert(0 < type.KeyCount && type.KeyCount < Utils.ArrayMaxSize);
                var mapper = BinKeys<T>(type);
                return
                    (ref VBuffer<T> src, ref VBuffer<int> dst, out int min, out int lim) =>
                    {
                        min = 0;
                        lim = type.KeyCount + 1;
                        mapper(in src, ref dst);
                    };
            }

            /// <summary>
            /// Computes the mutual information for one column.
            /// </summary>
            private float[] ComputeMutualInformation<T>(Transposer trans, int col, Mapper<T> mapper)
            {
                var slotCount = trans.Schema.GetColumnType(col).ValueCount;
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
                Contracts.Assert(_labels.Length == features.Length);
                if (features.IsDense)
                {
                    for (int i = 0; i < _labels.Length; i++)
                    {
                        var label = _labels[i];
                        var feature = features.Values[i] - offset;
                        Contracts.Assert(0 <= label && label < _numLabels);
                        Contracts.Assert(0 <= feature && feature < numFeatures);
                        _contingencyTable[label][feature]++;
                    }
                    return;
                }

                int ii = 0;
                for (int i = 0; i < _labels.Length; i++)
                {
                    var label = _labels[i];
                    int feature;
                    if (ii == features.Count || i < features.Indices[ii])
                        feature = -offset;
                    else
                    {
                        feature = features.Values[ii] - offset;
                        ii++;
                    }
                    Contracts.Assert(0 <= label && label < _numLabels);
                    Contracts.Assert(0 <= feature && feature < numFeatures);
                    _contingencyTable[label][feature]++;
                }
                Contracts.Assert(ii == features.Count);
            }

            /// <summary>
            /// Maps from keys to ints.
            /// </summary>
            private static ValueMapper<VBuffer<T>, VBuffer<int>> BinKeys<T>(ColumnType colType)
            {
                var conv = Runtime.Data.Conversion.Conversions.Instance.GetStandardConversion<T, uint>(colType, NumberType.U4, out bool identity);
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
            private void BinInts(ref VBuffer<int> input, ref VBuffer<int> output,
                int numBins, out int min, out int lim)
            {
                Contracts.Assert(_singles.Count == 0);

                var bounds = _binFinder.FindBins(numBins, _singles, input.Length - input.Count);
                min = -1 - bounds.FindIndexSorted(0);
                lim = min + bounds.Length + 1;
                int offset = min;
                ValueMapper<int, int> mapper =
                    (in int src, ref int dst) =>
                        dst = offset + 1 + bounds.FindIndexSorted((float)src);
                mapper.MapVector(in input, ref output);
                _singles.Clear();
            }

            /// <summary>
            /// Maps from Singles to ints. NaNs (and only NaNs) are mapped to the first bin.
            /// </summary>
            private void BinSingles(ref VBuffer<float> input, ref VBuffer<int> output,
                int numBins, out int min, out int lim)
            {
                Contracts.Assert(_singles.Count == 0);
                if (input.Values != null)
                {
                    for (int i = 0; i < input.Count; i++)
                    {
                        var val = input.Values[i];
                        if (!float.IsNaN(val))
                            _singles.Add(val);
                    }
                }

                var bounds = _binFinder.FindBins(numBins, _singles, input.Length - input.Count);
                min = -1 - bounds.FindIndexSorted(0);
                lim = min + bounds.Length + 1;
                int offset = min;
                ValueMapper<float, int> mapper =
                    (in float src, ref int dst) =>
                        dst = float.IsNaN(src) ? offset : offset + 1 + bounds.FindIndexSorted(src);
                mapper.MapVector(in input, ref output);
                _singles.Clear();
            }

            /// <summary>
            /// Maps from Doubles to ints. NaNs (and only NaNs) are mapped to the first bin.
            /// </summary>
            private void BinDoubles(ref VBuffer<double> input, ref VBuffer<int> output,
                int numBins, out int min, out int lim)
            {
                Contracts.Assert(_doubles.Count == 0);
                if (input.Values != null)
                {
                    for (int i = 0; i < input.Count; i++)
                    {
                        var val = input.Values[i];
                        if (!double.IsNaN(val))
                            _doubles.Add(val);
                    }
                }

                var bounds = _binFinder.FindBins(numBins, _doubles, input.Length - input.Count);
                var offset = min = -1 - bounds.FindIndexSorted(0);
                lim = min + bounds.Length + 1;
                ValueMapper<double, int> mapper =
                    (in double src, ref int dst) =>
                        dst = double.IsNaN(src) ? offset : offset + 1 + bounds.FindIndexSorted(src);
                mapper.MapVector(in input, ref output);
                _doubles.Clear();
            }

            private void BinBools(ref VBuffer<bool> input, ref VBuffer<int> output)
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
            var values = output.Values;
            if (Utils.Size(values) < input.Count)
                values = new TDst[input.Count];
            for (int i = 0; i < input.Count; i++)
            {
                TSrc val = input.Values[i];
                map(in val, ref values[i]);
            }

            var indices = output.Indices;
            if (!input.IsDense && input.Count > 0)
            {
                if (Utils.Size(indices) < input.Count)
                    indices = new int[input.Count];
                Array.Copy(input.Indices, indices, input.Count);
            }

            output = new VBuffer<TDst>(input.Length, input.Count, values, indices);
        }
    }
}
