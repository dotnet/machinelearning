// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Transforms;
using Float = System.Single;

namespace Microsoft.ML.Transforms
{

    using Stopwatch = System.Diagnostics.Stopwatch;

    public sealed class PermutationFeatureImportanceRegression
    {
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
            public VBuffer<Float> Features;
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

        private readonly IHostEnvironment _env;

        private VBuffer<ReadOnlyMemory<char>> _slotNames;

        /// <summary>
        /// Dictionary containing diff between Feature's Metric value (when the feature is permuted) and the baseline metric.
        /// </summary>
        private readonly List<(string featureName, RegressionEvaluator.Result metricsDelta)> _metricsStats;

        public PermutationFeatureImportanceRegression(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            _env = env;
            _metricsStats = new List<(string featureName, RegressionEvaluator.Result metricsDelta)>();
        }

        /// <summary>
        /// Given predictor, loader (and few other params) this method will construct scoring pipeline and evaluator to get baseline metrics of the model.
        /// Then each feature slot will be randomly permuted (individually) and evaluator's metrics will be compared to baseline,
        /// </summary>
        /// <returns> IDV with info how  permuting of a feature slot affects evaluator.
        /// Example (X1 is less important feature):
        /// Features	L1(avg)	    L2(avg)	        RMS(avg)	LOSS-FN(avg)
        /// X1	        269.783 	20657720.411	380.933	    20657722.786
        /// X2          31455.671 	3529601866.937	38302.485	3529601872.502
        /// X3	        1496.281	118470005.501	2116.855	118470007.2876
        /// </returns>
        //public IDataView GetImportanceMetricsMatrix(IPredictor predictor, IDataView loader,
        //    string features, string label, bool useFeatureWeightFilter, out int evaluatedFeaturesCount, int topExamples,
        //    int progressIterations = 10, string group = null, string weight = null, string name = null,
        //    IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> customColumns = null,
        //    IComponentFactory<IMamlEvaluator> evalComp = null,
        //    IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> scorer = null)

        public List<(string featureName, RegressionEvaluator.Result metricsDelta)>
            GetImportanceMetricsMatrix(ITransformer model, IDataView data,
            string label = DefaultColumnNames.Label, string features = DefaultColumnNames.Features,
            int topExamples = 0, int progressIterations = 10)
        {
            var host = _env.Register(nameof(GetImportanceMetricsMatrix));
            //host.CheckValue(predictor, nameof(predictor));
            //host.CheckValue(loader, nameof(loader));
            //host.CheckValue(features, nameof(features));
            //host.CheckValue(label, nameof(label));

            //IDataView resultView = null;

            // Todo: fix this
            var mlContext = new MLContext();

            using (var ch = host.Start("GetImportanceMetrics"))
            {
                ch.Trace("Scoring and evaluating baseline.");
                var baselineMetrics = mlContext.Regression.Evaluate(model.Transform(data), label: label);

                // Get slot names.
                var featuresColumn = data.Schema[features];
                int numSlots = featuresColumn.Type.VectorSize;
                data.Schema.TryGetColumnIndex(features, out int featuresColumnIndex);

                // Todo: check this
                //TrainerUtils.CheckFeatureFloatVector(roleMapped);

                ch.Info("Number of slots: " + numSlots);
                if (data.Schema.HasSlotNames(featuresColumnIndex, numSlots))
                    data.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, featuresColumnIndex, ref _slotNames);

                if (_slotNames.Length != numSlots)
                    _slotNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(numSlots);

                var workingFeatureIndices = Enumerable.Range(0, numSlots).ToList();

                // Note: this will not work on the huge dataset.
                var maxSize = topExamples > 0 ? topExamples : Utils.ArrayMaxSize;
                List<Float> initialfeatureValuesList = new List<Float>();

                // Cursor through the data to cache slot 0 values for the upcoming permutation.
                var valuesRowCount = 0;
                // REVIEW: Seems like if the labels are NaN, so that all metrics are NaN, this command will be useless.
                // In which case probably erroring out is probably the most useful thing.
                using (var cursor = data.GetRowCursor(col => col == featuresColumnIndex))
                {
                    var featuresGetter = cursor.GetGetter<VBuffer<float>>(featuresColumnIndex);
                    var featuresBuffer = default(VBuffer<float>);

                    while (initialfeatureValuesList.Count < maxSize && cursor.MoveNext())
                    {
                        featuresGetter(ref featuresBuffer);
                        initialfeatureValuesList.Add(featuresBuffer.GetItemOrDefault(workingFeatureIndices[0]));
                    }

                    valuesRowCount = initialfeatureValuesList.Count;
                    // Todo: KeptRowCount does not exist, do I need it?
                    //ch.Assert(valuesRowCount == cursor.KeptRowCount);
                }

                if (valuesRowCount > 0)
                {
                    ch.Info("Detected {0} examples for evaluation.", valuesRowCount);
                }
                else
                {
                    ch.Warning("Detected no examples for evaluation.");
                    return null;
                }

                Float[] featureValuesBuffer = initialfeatureValuesList.ToArray();
                Float[] nextValues = new float[valuesRowCount];

                // Now iterate through all the working slots, do permutation and calc the delta of metrics.
                int processedCnt = 0;
                int j = 0;
                int nextFeatureIndex = 0;
                int shuffleSeed = host.Rand.Next();
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Restart();
                foreach (var workingIndx in workingFeatureIndices)
                {
                    // Index for the feature we will permute next.  Needed to build in advance a buffer for the permutation.
                    if (j < workingFeatureIndices.Count - 1)
                        nextFeatureIndex = workingFeatureIndices[j + 1];

                    int nextValuesIndex = 0;

                    Utils.Shuffle(RandomUtils.Create(shuffleSeed), featureValuesBuffer);

                    Action<FeaturesBuffer, FeaturesBuffer, PermuterState> permuter =
                        (src, dst, state) =>
                        {
                            src.Features.CopyTo(ref dst.Features);
                            VBufferUtils.ApplyAt(ref dst.Features, workingIndx,
                                (int ii, ref Float d) =>
                                    d = featureValuesBuffer[state.SampleIndex++]);

                            if (j < workingFeatureIndices.Count - 1)
                            {
                                // This is the reason I need PermuterState in LambdaTransform.CreateMap.
                                nextValues[nextValuesIndex] = src.Features.GetItemOrDefault(nextFeatureIndex);
                                if (nextValuesIndex < valuesRowCount - 1)
                                    nextValuesIndex++;
                            }
                        };

                    SchemaDefinition input = SchemaDefinition.Create(typeof(FeaturesBuffer));
                    Contracts.Assert(input.Count == 1);
                    input[0].ColumnName = features;

                    SchemaDefinition output = SchemaDefinition.Create(typeof(FeaturesBuffer));
                    Contracts.Assert(output.Count == 1);
                    output[0].ColumnName = features;
                    output[0].ColumnType = featuresColumn.Type;

                    IDataView viewPermuted = LambdaTransform.CreateMap(
                        host, data, permuter, null, input, output);
                    if (topExamples > 0 && valuesRowCount == topExamples)
                        viewPermuted = SkipTakeFilter.Create(host, new SkipTakeFilter.TakeArguments() { Count = valuesRowCount }, viewPermuted);

                    var metrics = mlContext.Regression.Evaluate(model.Transform(viewPermuted), label: label);

                    UpdateFeatureMetricStats(baselineMetrics, metrics, ch, workingIndx);

                    // Swap values for next iteration of permutation.
                    Array.Clear(featureValuesBuffer, 0, featureValuesBuffer.Length);
                    nextValues.CopyTo(featureValuesBuffer, 0);
                    Array.Clear(nextValues, 0, nextValues.Length);

                    // Print out timings.
                    if (processedCnt > 0 && (processedCnt % progressIterations == 0))
                    {
                        stopwatch.Stop();
                        ch.Info(string.Format("Processed slots {0} - {1} in {2}", processedCnt - progressIterations,
                            processedCnt, stopwatch.Elapsed));
                        stopwatch.Restart();
                    }

                    processedCnt++;
                    j++;
                }

                if ((processedCnt - 1) % progressIterations != 0)
                {
                    stopwatch.Stop();
                    ch.Info(string.Format("Processed slots up to {0} in {1}", processedCnt - 1, stopwatch.Elapsed));
                }

                //resultView = BuildFinalReport(workingFeatureIndices);
            }

            //return resultView;
            return _metricsStats;
        }

        private void UpdateFeatureMetricStats(RegressionEvaluator.Result baselineMetrics, RegressionEvaluator.Result featureMetrics, IChannel ch, int slotIndex)
        {
            var delta = featureMetrics - baselineMetrics;
            var featureName = GetSlotName(slotIndex);
            _metricsStats.Add((featureName, delta));
        }

        /// <summary>
        /// Helper method that will add build IDV with each "row" containing info how each feature slot permutation impacted metrics
        /// </summary>
        private IDataView BuildFinalReport(List<int> workingFeatureIndices)
        {
            return null;
            //var indices = workingFeatureIndices.ToArray();

            //IDataView resultDV = null;
            //if (workingFeatureIndices.Count > 0 && _metricsStats.Count > 0)
            //{
            //    var builder = new ArrayDataViewBuilder(_env);
            //    builder.AddColumn("Index", NumberType.I4, indices);
            //    foreach (var metricColumn in _metricsStats)
            //        builder.AddColumn(metricColumn.Key + "_Delta", NumberType.R8, metricColumn.Value.ToArray());

            //    resultDV = builder.GetDataView();

            //    var schemaIn =
            //    new SchemaDefinition
            //    {
            //        new SchemaDefinition.Column{MemberName = "Index", ColumnName = "Index"}
            //    };

            //    var schemaOut =
            //        new SchemaDefinition
            //    {
            //        new SchemaDefinition.Column { MemberName = "Name", ColumnName = "Feature" }
            //    };

            //    resultDV = new CustomMappingTransformer<FeatureIndex, FeatureName>(_env,
            //        (featureIndex, featureName) => featureName.Name = GetSlotName(featureIndex.Index), null, schemaIn, schemaOut).Transform(resultDV);
            //}

            //return resultDV;
        }

        private string GetSlotName(int index)
        {
            var slotName = _slotNames.GetItemOrDefault(index);
            return !slotName.IsEmpty
                ? slotName.ToString()
                : string.Format("f{0}", index);
        }
    }
}