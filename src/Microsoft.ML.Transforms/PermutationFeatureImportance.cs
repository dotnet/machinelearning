// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    internal static class PermutationFeatureImportance<TModel, TMetric, TResult> where TResult : IMetricsStatistics<TMetric>
        where TModel : class
    {
        public static ImmutableArray<TResult>
            GetImportanceMetricsMatrix(
                IHostEnvironment env,
                IPredictionTransformer<TModel> model,
                IDataView data,
                Func<TResult> resultInitializer,
                Func<IDataView, TMetric> evaluationFunc,
                Func<TMetric, TMetric, TMetric> deltaFunc,
                string features,
                int permutationCount,
                bool useFeatureWeightFilter = false,
                int? topExamples = null)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(nameof(PermutationFeatureImportance<TModel, TMetric, TResult>));
            host.CheckValue(model, nameof(model));
            host.CheckValue(data, nameof(data));
            host.CheckNonEmpty(features, nameof(features));

            topExamples = topExamples ?? Utils.ArrayMaxSize;
            host.Check(topExamples > 0, "Provide how many examples to use (positive number) or set to null to use whole dataset.");

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            var metricsDelta = new List<TResult>();

            using (var ch = host.Start("GetImportanceMetrics"))
            {
                ch.Trace("Scoring and evaluating baseline.");
                var baselineMetrics = evaluationFunc(model.Transform(data));

                // Get slot names.
                var featuresColumn = data.Schema[features];
                int numSlots = featuresColumn.Type.GetVectorSize();
                data.Schema.TryGetColumnIndex(features, out int featuresColumnIndex);

                ch.Info("Number of slots: " + numSlots);
                if (data.Schema[featuresColumnIndex].HasSlotNames(numSlots))
                    data.Schema[featuresColumnIndex].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref slotNames);

                if (slotNames.Length != numSlots)
                    slotNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(numSlots);

                VBuffer<float> weights = default;
                var workingFeatureIndices = Enumerable.Range(0, numSlots).ToList();
                int zeroWeightsCount = 0;

                // By default set to the number of all features available.
                var evaluatedFeaturesCount = numSlots;
                if (useFeatureWeightFilter)
                {
                    var predictorWithWeights = model.Model as IPredictorWithFeatureWeights<Single>;
                    if (predictorWithWeights != null)
                    {
                        predictorWithWeights.GetFeatureWeights(ref weights);

                        const int maxReportedZeroFeatures = 10;
                        StringBuilder msgFilteredOutFeatures = new StringBuilder("The following features have zero weight and will not be evaluated: \n \t");
                        var prefix = "";
                        foreach (var k in weights.Items(all: true))
                        {
                            if (k.Value == 0)
                            {
                                zeroWeightsCount++;

                                // Print info about first few features we're not going to evaluate.
                                if (zeroWeightsCount <= maxReportedZeroFeatures)
                                {
                                    msgFilteredOutFeatures.Append(prefix);
                                    msgFilteredOutFeatures.Append(GetSlotName(slotNames, k.Key));
                                    prefix = ", ";
                                }
                            }
                            else
                                workingFeatureIndices.Add(k.Key);
                        }

                        // Old FastTree models has less weights than slots.
                        if (weights.Length < numSlots)
                        {
                            ch.Warning(
                                "Predictor had fewer features than slots. All unknown features will get default 0 weight.");
                            zeroWeightsCount += numSlots - weights.Length;
                            var indexes = weights.GetIndices().ToArray();
                            var values = weights.GetValues().ToArray();
                            var count = values.Length;
                            weights = new VBuffer<float>(numSlots, count, values, indexes);
                        }

                        evaluatedFeaturesCount = workingFeatureIndices.Count;
                        ch.Info("Number of zero weights: {0} out of {1}.", zeroWeightsCount, weights.Length);

                        // Print what features have 0 weight
                        if (zeroWeightsCount > 0)
                        {
                            if (zeroWeightsCount > maxReportedZeroFeatures)
                            {
                                msgFilteredOutFeatures.Append(string.Format("... (printing out  {0} features here).\n Use 'Index' column in the report for info on what features are not evaluated.", maxReportedZeroFeatures));
                            }
                            ch.Info(msgFilteredOutFeatures.ToString());
                        }
                    }
                }

                if (workingFeatureIndices.Count == 0 && zeroWeightsCount == 0)
                {
                    // Use all features otherwise.
                    workingFeatureIndices.AddRange(Enumerable.Range(0, numSlots));
                }

                if (zeroWeightsCount == numSlots)
                {
                    ch.Warning("All features have 0 weight thus can not do thorough evaluation");
                    return metricsDelta.ToImmutableArray();
                }

                // Note: this will not work on the huge dataset.
                var maxSize = topExamples;
                List<float> initialfeatureValuesList = new List<float>();

                // Cursor through the data to cache slot 0 values for the upcoming permutation.
                var valuesRowCount = 0;
                // REVIEW: Seems like if the labels are NaN, so that all metrics are NaN, this command will be useless.
                // In which case probably erroring out is probably the most useful thing.
                using (var cursor = data.GetRowCursor(featuresColumn))
                {
                    var featuresGetter = cursor.GetGetter<VBuffer<float>>(featuresColumn);
                    var featuresBuffer = default(VBuffer<float>);

                    while (initialfeatureValuesList.Count < maxSize && cursor.MoveNext())
                    {
                        featuresGetter(ref featuresBuffer);
                        initialfeatureValuesList.Add(featuresBuffer.GetItemOrDefault(workingFeatureIndices[0]));
                    }

                    valuesRowCount = initialfeatureValuesList.Count;
                }

                if (valuesRowCount > 0)
                {
                    ch.Info("Detected {0} examples for evaluation.", valuesRowCount);
                }
                else
                {
                    ch.Warning("Detected no examples for evaluation.");
                    return metricsDelta.ToImmutableArray();
                }

                float[] featureValuesBuffer = initialfeatureValuesList.ToArray();
                float[] nextValues = new float[valuesRowCount];

                // Now iterate through all the working slots, do permutation and calc the delta of metrics.
                int processedCnt = 0;
                int nextFeatureIndex = 0;
                var shuffleRand = RandomUtils.Create(host.Rand.Next());
                using (var pch = host.StartProgressChannel("SDCA preprocessing with lookup"))
                {
                    pch.SetHeader(new ProgressHeader("processed slots"), e => e.SetProgress(0, processedCnt));
                    foreach (var workingIndx in workingFeatureIndices)
                    {
                        // Index for the feature we will permute next.  Needed to build in advance a buffer for the permutation.
                        if (processedCnt < workingFeatureIndices.Count - 1)
                            nextFeatureIndex = workingFeatureIndices[processedCnt + 1];

                        // Used for pre-caching the next feature
                        int nextValuesIndex = 0;

                        SchemaDefinition input = SchemaDefinition.Create(typeof(FeaturesBuffer));
                        Contracts.Assert(input.Count == 1);
                        input[0].ColumnName = features;

                        SchemaDefinition output = SchemaDefinition.Create(typeof(FeaturesBuffer));
                        Contracts.Assert(output.Count == 1);
                        output[0].ColumnName = features;
                        output[0].ColumnType = featuresColumn.Type;

                        // Perform multiple permutations for one feature to build a confidence interval
                        var metricsDeltaForFeature = resultInitializer();
                        for (int permutationIteration = 0; permutationIteration < permutationCount; permutationIteration++)
                        {
                            Utils.Shuffle<float>(shuffleRand, featureValuesBuffer);

                            Action<FeaturesBuffer, FeaturesBuffer, PermuterState> permuter =
                                (src, dst, state) =>
                                {
                                    src.Features.CopyTo(ref dst.Features);
                                    VBufferUtils.ApplyAt(ref dst.Features, workingIndx,
                                        (int ii, ref float d) =>
                                            d = featureValuesBuffer[state.SampleIndex++]);

                                    // Is it time to pre-cache the next feature?
                                    if (permutationIteration == permutationCount - 1 &&
                                        processedCnt < workingFeatureIndices.Count - 1)
                                    {
                                        // Fill out the featureValueBuffer for the next feature while updating the current feature
                                        // This is the reason I need PermuterState in LambdaTransform.CreateMap.
                                        nextValues[nextValuesIndex] = src.Features.GetItemOrDefault(nextFeatureIndex);
                                        if (nextValuesIndex < valuesRowCount - 1)
                                            nextValuesIndex++;
                                    }
                                };

                            IDataView viewPermuted = LambdaTransform.CreateMap(
                                host, data, permuter, null, input, output);
                            if (valuesRowCount == topExamples)
                                viewPermuted = SkipTakeFilter.Create(host, new SkipTakeFilter.TakeOptions() { Count = valuesRowCount }, viewPermuted);

                            var metrics = evaluationFunc(model.Transform(viewPermuted));

                            var delta = deltaFunc(metrics, baselineMetrics);
                            metricsDeltaForFeature.Add(delta);
                        }

                        // Add the metrics delta to the list
                        metricsDelta.Add(metricsDeltaForFeature);

                        // Swap values for next iteration of permutation.
                        if (processedCnt < workingFeatureIndices.Count - 1)
                        {
                            Array.Clear(featureValuesBuffer, 0, featureValuesBuffer.Length);
                            nextValues.CopyTo(featureValuesBuffer, 0);
                            Array.Clear(nextValues, 0, nextValues.Length);
                        }
                        processedCnt++;
                    }
                    pch.Checkpoint(processedCnt, processedCnt);
                }
            }

            return metricsDelta.ToImmutableArray();
        }

        private static ReadOnlyMemory<char> GetSlotName(VBuffer<ReadOnlyMemory<char>> slotNames, int index)
        {
            var slotName = slotNames.GetItemOrDefault(index);
            return slotName.IsEmpty
                ? slotName
                : string.Format("f{0}", index).AsMemory();
        }

        /// <summary>
        /// This is used as a hack to force Lambda Transform behave sequentially.
        /// </summary>
        private sealed class PermuterState
        {
            public int SampleIndex;
        }

        /// <summary>
        /// Helper structure used for features permutation in Lambda Transform.
        /// </summary>
        private sealed class FeaturesBuffer
        {
            public VBuffer<float> Features;
        }

        /// <summary>
        /// Helper class for report's Lambda transform.
        /// </summary>
        private sealed class FeatureIndex
        {
#pragma warning disable 0649
            public int Index;
#pragma warning restore 0649
        }

        /// <summary>
        ///  One more helper class for report's Lambda transform.
        /// </summary>
        private sealed class FeatureName
        {
#pragma warning disable 0649
            public ReadOnlyMemory<char> Name;
#pragma warning restore 0649
        }
    }
}