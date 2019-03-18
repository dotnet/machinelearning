// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(LearnerFeatureSelectionTransform.Summary, typeof(IDataTransform), typeof(LearnerFeatureSelectionTransform), typeof(LearnerFeatureSelectionTransform.Options), typeof(SignatureDataTransform),
    "Learner Feature Selection Transform", "LearnerFeatureSelectionTransform", "LearnerFeatureSelection")]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Selects the slots for which the absolute value of the corresponding weight in a linear learner
    /// is greater than a threshold.
    /// Instantiates a DropSlots transform to actually drop the slots.
    /// </summary>
    internal static class LearnerFeatureSelectionTransform
    {
        internal const string Summary = "Selects the slots for which the absolute value of the corresponding weight in a linear learner is greater than a threshold.";

#pragma warning disable CS0649 // The fields will still be set via the reflection driven mechanisms.
        public sealed class Options
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If the corresponding absolute value of the weight for a slot is greater than this threshold, the slot is preserved", ShortName = "ft", SortOrder = 2)]
            public Single? Threshold;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of slots to preserve", ShortName = "topk", SortOrder = 1)]
            public int? NumSlotsToKeep;

            // If we make this public again it should be an *estimator* of this type of predictor, rather than the (deprecated) ITrainer, but the utility
            // of this would be limited because estimators and transformers now act more or less like this transform used to.
            [Argument(ArgumentType.Multiple, HelpText = "Filter", ShortName = "f", SortOrder = 1, SignatureType = typeof(SignatureFeatureScorerTrainer))]
            public IComponentFactory<ITrainer<IPredictorWithFeatureWeights<Single>>> Filter =
                ComponentFactoryUtils.CreateFromFunction(env =>
                    // ML.Transforms doesn't have a direct reference to ML.StandardTrainers, so use ComponentCatalog to create the Filter
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

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Columns with custom kinds declared through key assignments, for example, col[Kind]=Name to assign column named 'Name' kind 'Kind'",
                Name = "CustomColumn")]
            public KeyValuePair<string, string>[] CustomColumns;

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
#pragma warning restore CS0649

        internal static string RegistrationName = "LearnerFeatureSelectionTransform";

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(options, nameof(options));
            host.CheckValue(input, nameof(input));
            options.Check(host);

            var scores = default(VBuffer<Single>);
            TrainCore(host, input, options, ref scores);

            using (var ch = host.Start("Dropping Slots"))
            {
                int selectedCount;
                var column = CreateDropSlotsColumn(options, in scores, out selectedCount);

                if (column == null)
                {
                    ch.Info("No features are being dropped.");
                    return NopTransform.CreateIfNeeded(host, input);
                }

                ch.Info(MessageSensitivity.Schema, "Selected {0} slots out of {1} in column '{2}'", selectedCount, scores.Length, options.FeatureColumn);

                return new SlotsDroppingTransformer(host, column).Transform(input) as IDataTransform;
            }
        }

        private static SlotsDroppingTransformer.ColumnOptions CreateDropSlotsColumn(Options options, in VBuffer<Single> scores, out int selectedCount)
        {
            // Not checking the scores.Length, because:
            // 1. If it's the same as the features column length, we should be constructing the right DropSlots arguments.
            // 2. If it's less, we assume that the rest of the scores are zero and we drop the slots.
            // 3. If it's greater, the drop slots ignores the ranges that are outside the valid range of indices for the column.
            Contracts.Assert(options.Threshold.HasValue != options.NumSlotsToKeep.HasValue);
            var col = new SlotsDroppingTransformer.Column();
            col.Source = options.FeatureColumn;
            selectedCount = 0;
            var scoresValues = scores.GetValues();

            // Degenerate case, dropping all slots.
            if (scoresValues.Length == 0)
                return new SlotsDroppingTransformer.ColumnOptions(options.FeatureColumn);

            int tiedScoresToKeep;
            float threshold;
            if (options.Threshold.HasValue)
            {
                threshold = options.Threshold.Value;
                tiedScoresToKeep = threshold > 0 ? int.MaxValue : 0;
            }
            else
            {
                Contracts.Assert(options.NumSlotsToKeep.HasValue);
                threshold = ComputeThreshold(scoresValues, options.NumSlotsToKeep.Value, out tiedScoresToKeep);
            }

            var slots = new List<(int min, int? max)>();
            for (int i = 0; i < scoresValues.Length; i++)
            {
                var score = Math.Abs(scoresValues[i]);
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

                int min = i;
                while (++i < scoresValues.Length)
                {
                    score = Math.Abs(scoresValues[i]);
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
                int max = i - 1;
                slots.Add((min, max));
            }

            if (!scores.IsDense)
            {
                var scoresIndices = scores.GetIndices();
                int ii = 0;
                var count = slots.Count;
                for (int i = 0; i < count; i++)
                {
                    var range = slots[i];
                    Contracts.Assert(range.max != null);
                    var min = range.min;
                    var max = range.max.Value;
                    Contracts.Assert(min <= max);
                    Contracts.Assert(max < scoresValues.Length);

                    range.min = min == 0 ? 0 : scoresIndices[min - 1] + 1;
                    range.max = max == scoresIndices.Length - 1 ? scores.Length - 1 : scoresIndices[max + 1] - 1;

                    // Add the gaps before this range.
                    for (; ii < min; ii++)
                    {
                        var gapMin = ii == 0 ? 0 : scoresIndices[ii - 1] + 1;
                        var gapMax = scoresIndices[ii] - 1;
                        if (gapMin <= gapMax)
                        {
                            slots.Add((gapMin, gapMax));
                        }
                    }
                    ii = max;
                }

                // Add the gaps after the last range.
                for (; ii <= scoresIndices.Length; ii++)
                {
                    var gapMin = ii == 0 ? 0 : scoresIndices[ii - 1] + 1;
                    var gapMax = ii == scoresIndices.Length ? scores.Length - 1 : scoresIndices[ii] - 1;
                    if (gapMin <= gapMax)
                    {
                        slots.Add((gapMin, gapMax));
                    }
                }

                // Remove all slots past scores.Length.
                slots.Add((scores.Length, null));
            }

            if (slots.Count > 0)
                return new SlotsDroppingTransformer.ColumnOptions(options.FeatureColumn, slots: slots.ToArray());

            return null;
        }

        private static float ComputeThreshold(ReadOnlySpan<float> scores, int topk, out int tiedScoresToKeep)
        {
            // Use a min-heap for the topk elements
            var heap = new Heap<float>((f1, f2) => f1 > f2, topk);

            for (int i = 0; i < scores.Length; i++)
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

        private static void TrainCore(IHost host, IDataView input, Options options, ref VBuffer<Single> scores)
        {
            Contracts.AssertValue(host);
            host.AssertValue(options);
            host.AssertValue(input);
            host.Assert(options.Threshold.HasValue != options.NumSlotsToKeep.HasValue);

            using (var ch = host.Start("Train"))
            {
                ch.Trace("Constructing trainer");
                ITrainer trainer = options.Filter.CreateComponent(host);

                IDataView view = input;

                var schema = view.Schema;
                var label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(options.LabelColumn), options.LabelColumn, DefaultColumnNames.Label);
                var feature = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(options.FeatureColumn), options.FeatureColumn, DefaultColumnNames.Features);
                var group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(options.GroupColumn), options.GroupColumn, DefaultColumnNames.GroupId);
                var weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(options.WeightColumn), options.WeightColumn, DefaultColumnNames.Weight);
                var name = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(options.NameColumn), options.NameColumn, DefaultColumnNames.Name);

                TrainUtils.AddNormalizerIfNeeded(host, ch, trainer, ref view, feature, options.NormalizeFeatures);

                ch.Trace("Binding columns");

                var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, options.CustomColumns);
                var data = new RoleMappedData(view, label, feature, group, weight, name, customCols);

                var predictor = TrainUtils.Train(host, ch, data, trainer, null,
                    null, 0, options.CacheData);

                var rfs = predictor as IPredictorWithFeatureWeights<Single>;
                Contracts.AssertValue(rfs);
                rfs.GetFeatureWeights(ref scores);
            }
        }

        /// <summary>
        /// Returns a score for each slot of the features column.
        /// </summary>
        public static void Train(IHostEnvironment env, IDataView input, Options options, ref VBuffer<Single> scores)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(options, nameof(options));
            host.CheckValue(input, nameof(input));
            options.Check(host);

            TrainCore(host, input, options, ref scores);
        }
    }
}
