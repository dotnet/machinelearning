// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(LearnerFeatureSelectionTransform.Summary, typeof(IDataTransform), typeof(LearnerFeatureSelectionTransform), typeof(LearnerFeatureSelectionTransform.Arguments), typeof(SignatureDataTransform),
    "Learner Feature Selection Transform", "LearnerFeatureSelectionTransform", "LearnerFeatureSelection")]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Selects the slots for which the absolute value of the corresponding weight in a linear learner
    /// is greater than a threshold.
    /// Instantiates a DropSlots transform to actually drop the slots.
    /// </summary>
    public static class LearnerFeatureSelectionTransform
    {
        internal const string Summary = "Selects the slots for which the absolute value of the corresponding weight in a linear learner is greater than a threshold.";

        public sealed class Arguments
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If the corresponding absolute value of the weight for a slot is greater than this threshold, the slot is preserved", ShortName = "ft", SortOrder = 2)]
            public Single? Threshold;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of slots to preserve", ShortName = "topk", SortOrder = 1)]
            public int? NumSlotsToKeep;

            [Argument(ArgumentType.Multiple, HelpText = "Filter", ShortName = "f", SortOrder = 1, SignatureType = typeof(SignatureFeatureScorerTrainer))]
            public IComponentFactory<ITrainer<IPredictorWithFeatureWeights<Single>>> Filter =
                ComponentFactoryUtils.CreateFromFunction(env =>
                    // ML.Transforms doesn't have a direct reference to ML.StandardLearners, so use ComponentCatalog to create the Filter
                    ComponentCatalog.CreateInstance<ITrainer<IPredictorWithFeatureWeights<Single>>>(env, typeof(SignatureFeatureScorerTrainer), "SDCA", options: null));

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for features", ShortName = "feat,col", SortOrder = 3, Purpose = SpecialPurpose.ColumnName)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 4, Purpose = SpecialPurpose.ColumnName)]
            public string LabelColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 5, Purpose = SpecialPurpose.ColumnName)]
            public string WeightColumn = DefaultColumnNames.Weight;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for grouping", ShortName = "group", Purpose = SpecialPurpose.ColumnName)]
            public string GroupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name column name", ShortName = "name", Purpose = SpecialPurpose.ColumnName)]
            public string NameColumn = DefaultColumnNames.Name;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Columns with custom kinds declared through key assignments, e.g., col[Kind]=Name to assign column named 'Name' kind 'Kind'")]
            public KeyValuePair<string, string>[] CustomColumn;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Normalize option for the feature column", ShortName = "norm")]
            public NormalizeOption NormalizeFeatures = NormalizeOption.Auto;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Whether we should cache input training data", ShortName = "cache")]
            public bool? CacheData;

            internal void Check(IExceptionContext ectx)
            {
                if (Threshold.HasValue == NumSlotsToKeep.HasValue)
                {
                    throw ectx.ExceptUserArg(nameof(Threshold),
                        $"Either {nameof(Threshold)} or {nameof(NumSlotsToKeep)} to keep must be specified (but not both).");
                }
                ectx.CheckUserArg((Threshold ?? 0) >= 0, nameof(Threshold), "Must be non-negative");
                ectx.CheckUserArg((NumSlotsToKeep ?? int.MaxValue) > 0, nameof(NumSlotsToKeep), "Must be positive");
            }
        }

        internal static string RegistrationName = "LearnerFeatureSelectionTransform";

        /// <summary>
        /// Create method corresponding to SignatureDataTransform.
        /// </summary>
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(args, nameof(args));
            host.CheckValue(input, nameof(input));
            args.Check(host);

            var scores = default(VBuffer<Single>);
            TrainCore(host, input, args, ref scores);

            using (var ch = host.Start("Dropping Slots"))
            {
                int selectedCount;
                var column = CreateDropSlotsColumn(args, ref scores, out selectedCount);

                if (column == null)
                {
                    ch.Info("No features are being dropped.");
                    return NopTransform.CreateIfNeeded(host, input);
                }

                ch.Info(MessageSensitivity.Schema, "Selected {0} slots out of {1} in column '{2}'", selectedCount, scores.Length, args.FeatureColumn);

                var dsArgs = new DropSlotsTransform.Arguments();
                dsArgs.Column = new[] { column };
                ch.Done();
                return new DropSlotsTransform(host, dsArgs, input);
            }
        }

        private static DropSlotsTransform.Column CreateDropSlotsColumn(Arguments args, ref VBuffer<Single> scores, out int selectedCount)
        {
            // Not checking the scores.Length, because:
            // 1. If it's the same as the features column length, we should be constructing the right DropSlots arguments.
            // 2. If it's less, we assume that the rest of the scores are zero and we drop the slots.
            // 3. If it's greater, the drop slots ignores the ranges that are outside the valid range of indices for the column.
            Contracts.Assert(args.Threshold.HasValue != args.NumSlotsToKeep.HasValue);
            var col = new DropSlotsTransform.Column();
            col.Source = args.FeatureColumn;
            selectedCount = 0;

            // Degenerate case, dropping all slots.
            if (scores.Count == 0)
            {
                var range = new DropSlotsTransform.Range();
                col.Slots = new DropSlotsTransform.Range[] { range };
                return col;
            }

            int tiedScoresToKeep;
            float threshold;
            if (args.Threshold.HasValue)
            {
                threshold = args.Threshold.Value;
                tiedScoresToKeep = threshold > 0 ? int.MaxValue : 0;
            }
            else
            {
                Contracts.Assert(args.NumSlotsToKeep.HasValue);
                threshold = ComputeThreshold(scores.Values, scores.Count, args.NumSlotsToKeep.Value, out tiedScoresToKeep);
            }

            var slots = new List<DropSlotsTransform.Range>();
            for (int i = 0; i < scores.Count; i++)
            {
                var score = Math.Abs(scores.Values[i]);
                if (score > threshold)
                {
                    selectedCount++;
                    continue;
                }
                if (score == threshold && tiedScoresToKeep > 0)
                {
                    tiedScoresToKeep--;
                    selectedCount++;
                    continue;
                }

                var range = new DropSlotsTransform.Range();
                range.Min = i;
                while (++i < scores.Count)
                {
                    score = Math.Abs(scores.Values[i]);
                    if (score > threshold)
                    {
                        selectedCount++;
                        break;
                    }
                    if (score == threshold && tiedScoresToKeep > 0)
                    {
                        tiedScoresToKeep--;
                        selectedCount++;
                        break;
                    }
                }
                range.Max = i - 1;
                slots.Add(range);
            }

            if (!scores.IsDense)
            {
                int ii = 0;
                var count = slots.Count;
                for (int i = 0; i < count; i++)
                {
                    var range = slots[i];
                    Contracts.Assert(range.Max != null);
                    var min = range.Min;
                    var max = range.Max.Value;
                    Contracts.Assert(min <= max);
                    Contracts.Assert(max < scores.Count);

                    range.Min = min == 0 ? 0 : scores.Indices[min - 1] + 1;
                    range.Max = max == scores.Count - 1 ? scores.Length - 1 : scores.Indices[max + 1] - 1;

                    // Add the gaps before this range.
                    for (; ii < min; ii++)
                    {
                        var gapMin = ii == 0 ? 0 : scores.Indices[ii - 1] + 1;
                        var gapMax = scores.Indices[ii] - 1;
                        if (gapMin <= gapMax)
                        {
                            var gap = new DropSlotsTransform.Range();
                            gap.Min = gapMin;
                            gap.Max = gapMax;
                            slots.Add(gap);
                        }
                    }
                    ii = max;
                }

                // Add the gaps after the last range.
                for (; ii <= scores.Count; ii++)
                {
                    var gapMin = ii == 0 ? 0 : scores.Indices[ii - 1] + 1;
                    var gapMax = ii == scores.Count ? scores.Length - 1 : scores.Indices[ii] - 1;
                    if (gapMin <= gapMax)
                    {
                        var gap = new DropSlotsTransform.Range();
                        gap.Min = gapMin;
                        gap.Max = gapMax;
                        slots.Add(gap);
                    }
                }

                // Remove all slots past scores.Length.
                var lastRange = new DropSlotsTransform.Range();
                lastRange.Min = scores.Length;
                slots.Add(lastRange);
            }

            if (slots.Count > 0)
            {
                col.Slots = slots.ToArray();
                return col;
            }

            return null;
        }

        private static float ComputeThreshold(float[] scores, int count, int topk, out int tiedScoresToKeep)
        {
            // Use a min-heap for the topk elements
            var heap = new Heap<float>((f1, f2) => f1 > f2, topk);

            for (int i = 0; i < count; i++)
            {
                var score = Math.Abs(scores[i]);
                if (float.IsNaN(score))
                    continue;
                if (heap.Count < topk)
                    heap.Add(score);
                else if (heap.Top < score)
                {
                    Contracts.Assert(heap.Count == topk);
                    heap.Pop();
                    heap.Add(score);
                }
            }

            var threshold = heap.Top;
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

        private static void TrainCore(IHost host, IDataView input, Arguments args, ref VBuffer<Single> scores)
        {
            Contracts.AssertValue(host);
            host.AssertValue(args);
            host.AssertValue(input);
            host.Assert(args.Threshold.HasValue != args.NumSlotsToKeep.HasValue);

            using (var ch = host.Start("Train"))
            {
                ch.Trace("Constructing trainer");
                ITrainer trainer = args.Filter.CreateComponent(host);

                IDataView view = input;

                ISchema schema = view.Schema;
                var label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(args.LabelColumn), args.LabelColumn, DefaultColumnNames.Label);
                var feature = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(args.FeatureColumn), args.FeatureColumn, DefaultColumnNames.Features);
                var group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(args.GroupColumn), args.GroupColumn, DefaultColumnNames.GroupId);
                var weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(args.WeightColumn), args.WeightColumn, DefaultColumnNames.Weight);
                var name = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(args.NameColumn), args.NameColumn, DefaultColumnNames.Name);

                TrainUtils.AddNormalizerIfNeeded(host, ch, trainer, ref view, feature, args.NormalizeFeatures);

                ch.Trace("Binding columns");

                var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, args.CustomColumn);
                var data = new RoleMappedData(view, label, feature, group, weight, name, customCols);

                var predictor = TrainUtils.Train(host, ch, data, trainer, null,
                    null, 0, args.CacheData);

                var rfs = predictor as IPredictorWithFeatureWeights<Single>;
                Contracts.AssertValue(rfs);
                rfs.GetFeatureWeights(ref scores);
                ch.Done();
            }
        }

        /// <summary>
        /// Returns a score for each slot of the features column.
        /// </summary>
        public static void Train(IHostEnvironment env, IDataView input, Arguments args, ref VBuffer<Single> scores)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(args, nameof(args));
            host.CheckValue(input, nameof(input));
            args.Check(host);

            TrainCore(host, input, args, ref scores);
        }
    }
}
