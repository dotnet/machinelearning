//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using HelperCommands;
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

[assembly: LoadableClass(typeof(PermutationFeatureImportanceCommand), typeof(PermutationFeatureImportanceCommand.Arguments), typeof(SignatureCommand),
    "Permutation Feature Importance", "PermutationFeatureImportance", "pfi")]

namespace HelperCommands
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    /// <summary>
    /// This command detects the importance of features by permuting (corrupting) features individually and measuring the bad impact on the initial metrics (like accuracy for example).
    /// </summary>
    public sealed class PermutationFeatureImportanceCommand : DataCommand.ImplBase<PermutationFeatureImportanceCommand.Arguments>
    {
        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for features", ShortName = "feat",
                SortOrder = 2)]
            public string FeatureColumn = DefaultColumnNames.Features;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for labels", ShortName = "lab",
                SortOrder = 3)]
            public string LabelColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for example weight",
                ShortName = "weight", SortOrder = 4)]
            public string WeightColumn = DefaultColumnNames.Weight;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for grouping", ShortName = "group",
                SortOrder = 5)]
            public string GroupColumn = DefaultColumnNames.GroupId;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name column name", ShortName = "name", SortOrder = 6)]
            public string NameColumn = DefaultColumnNames.Name;

            [Argument(ArgumentType.LastOccurenceWins,
                HelpText =
                    "Columns with custom kinds declared through key assignments, e.g., col[Kind]=Name to assign column named 'Name' kind 'Kind'",
                ShortName = "col", SortOrder = 7)]
            public KeyValuePair<string, string>[] CustomColumn;

            [Argument(ArgumentType.Multiple, HelpText = "Scorer to use", NullName = "<Auto>", SortOrder = 8)]
            public IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> Scorer;

            [Argument(ArgumentType.Multiple, HelpText = "Evaluator to use", ShortName = "eval", NullName = "<Auto>",
                SortOrder = 9)]
            public IComponentFactory<IMamlEvaluator> Evaluator;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use features weight to pre-filter features", ShortName = "usefw", SortOrder = 10)]
            public bool UseFeatureWeightFilter = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of examples to evaluate on. Zero means examples (up to ~ 2 bln) from input will be used", ShortName = "top", SortOrder = 11)]
            public int TopExamples = 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Show progress as every X features are evaluated", ShortName = "p", SortOrder = 12)]
            public int ProgressIterations = 10;

            [Argument(ArgumentType.Required,
                HelpText =
                    "Name of the file where we should write the report of feature name and its impact on the evaluator metrics",
                ShortName = "rout")]
            public string OutReportFile;
        }

        public PermutationFeatureImportanceCommand(IHostEnvironment env, Arguments args)
                : base(env, args, nameof(PermutationFeatureImportanceCommand))
        {
            Host.CheckUserArg(!string.IsNullOrEmpty(Args.InputModelFile), nameof(Arguments.InputModelFile), "The input model file is required.");

            Host.CheckUserArg(!string.IsNullOrEmpty(Args.DataFile), nameof(Arguments.DataFile), "The data file is required for evaluating feature importance.");
            if (!File.Exists(args.DataFile))
                throw Host.ExceptUserArg(nameof(Arguments.DataFile), "dataFile '{0}' does not point to an existing path.", args.DataFile);

            Utils.CheckOptionalUserDirectory(args.OutReportFile, nameof(Arguments.OutReportFile));
        }

        public override void Run()
        {
            using (var ch = Host.Start("PermutationFeatureImportance"))
            {
                // Begin timing.
                Stopwatch stopwatch = Stopwatch.StartNew();

                IDataLoader loader = CreateLoader();
                IPredictor predictor;

                using (var file = Host.OpenInputFile(Args.InputModelFile))
                using (var strm = file.OpenReadStream())
                using (var rep = RepositoryReader.Open(strm, ch))
                {
                    ch.Trace("Loading predictor");
                    ModelLoadContext.LoadModel<IPredictor, SignatureLoadModel>(Host, out predictor, rep, ModelFileUtils.DirPredictor);
                }

                ch.Trace("Binding columns");
                ISchema schema = loader.Schema;

                string label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.LabelColumn),
                    Args.LabelColumn, DefaultColumnNames.Label);
                string features = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.FeatureColumn),
                    Args.FeatureColumn, DefaultColumnNames.Features);
                string group = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.GroupColumn),
                    Args.GroupColumn, DefaultColumnNames.GroupId);
                string weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.WeightColumn),
                    Args.WeightColumn, DefaultColumnNames.Weight);
                string name = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.NameColumn),
                    Args.NameColumn, DefaultColumnNames.Name);
                var customCols = TrainUtils.CheckAndGenerateCustomColumns(ch, Args.CustomColumn);

                if (Args.TopExamples > 0)
                    ch.Info("Limiting evaluation to first {0} examples.", Args.TopExamples);

                var pfi = new PermutationFeatureImportance(Host);
                int evaluatedFeaturesCount = 0;
                var resultView = pfi.GetImportanceMetricsMatrix(predictor, loader, features, label, Args.UseFeatureWeightFilter, out evaluatedFeaturesCount, Args.TopExamples, Args.ProgressIterations, group, weight,
                    name, customCols, Args.Evaluator, Args.Scorer);

                if (resultView == null)
                    ch.Warning("No feature importance data is available.");
                else
                    PrintResults(resultView, evaluatedFeaturesCount, ch);

                // Stop timing.
                stopwatch.Stop();
                ch.Info("Time elapsed: {0}", stopwatch.Elapsed);
            }
        }

        private void PrintResults(IDataView resultView, int evaluatedFeaturesCount, IChannel ch)
        {
            ch.Info("Deltas between permuted features and baseline evaluator:");

            // Send the first 10 lines to console.
            const int displayLinesMax = 10;
            var saverConsole = new TextSaver(Host, new TextSaver.Arguments() { Dense = true, OutputSchema = false });
            var previewData = evaluatedFeaturesCount <= displayLinesMax
                ? resultView
                : SkipTakeFilter.Create(Host, new SkipTakeFilter.Arguments() { Take = displayLinesMax },
                    resultView);

            // Report's 1st column will be "Feature" and 2nd is "Index".
            int[] cols = new int[2];
            var result = resultView.Schema.TryGetColumnIndex("Feature", out cols[0]);
            Contracts.Assert(result);
            result = resultView.Schema.TryGetColumnIndex("Index", out cols[1]);
            Contracts.Assert(result);
            cols = cols.Concat(Enumerable.Range(0, resultView.Schema.ColumnCount).Where(c => !cols.Contains(c))).ToArray();

            saverConsole.WriteData(previewData, true, cols);
            if (evaluatedFeaturesCount > displayLinesMax)
                ch.Info(" .... See report file for full info.");

            if (!string.IsNullOrWhiteSpace(Args.OutReportFile))
            {
                var saverFile = new TextSaver(Host, new TextSaver.Arguments { Dense = true, OutputSchema = false });
                using (var file = Host.CreateOutputFile(Args.OutReportFile))
                using (var stream = file.CreateWriteStream())
                    saverFile.SaveData(stream, resultView, cols);
            }
        }
    }

    /// <summary>
    /// Class that permutes sequentially features to measure Importance of a feature.
    /// </summary>
    public sealed class PermutationFeatureImportance
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
            public ReadOnlyMemory<char> Name;
        }

        private readonly IHostEnvironment _env;

        private VBuffer<ReadOnlyMemory<char>> _slotNames;

        /// <summary>
        /// Dictionary containing diff between Feature's Metric value (when the feature is permuted) and the baseline metric.
        /// </summary>
        private readonly Dictionary<string, List<Double>> _metricsStats;

        public PermutationFeatureImportance(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            _env = env;
            _metricsStats = new Dictionary<string, List<Double>>();
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
        public IDataView GetImportanceMetricsMatrix(IPredictor predictor, IDataView loader,
            string features, string label, bool useFeatureWeightFilter, out int evaluatedFeaturesCount, int topExamples,
            int progressIterations = 10, string group = null, string weight = null, string name = null,
            IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> customColumns = null,
            IComponentFactory<IMamlEvaluator> evalComp = null,
            IComponentFactory<IDataView, ISchemaBoundMapper, RoleMappedSchema, IDataScorerTransform> scorer = null)
        {
            var host = _env.Register(nameof(GetImportanceMetricsMatrix));
            host.CheckValue(predictor, nameof(predictor));
            host.CheckValue(loader, nameof(loader));
            host.CheckValue(features, nameof(features));
            host.CheckValue(label, nameof(label));

            IDataView resultView = null;

            using (var ch = host.Start("GetImportanceMetrics"))
            {
                // First initialize the role mapping on the input data.
                var dataEval = new RoleMappedData(loader, label, features, group, weight, name, customColumns, opt: true);
                // Get baseline metrics from evaluator first.
                IDataScorerTransform scorePipe = ScoreUtils.GetScorer(scorer, predictor, loader, features,
                    group, customColumns, host, dataEval.Schema);
                IMamlEvaluator evaluator = evalComp?.CreateComponent(host) ?? EvaluateUtils.GetEvaluator(host, scorePipe.Schema);

                // Recast the role mappings onto the now scored data.
                dataEval = new RoleMappedData(scorePipe, label, features,
                    group, weight, name, customColumns, opt: true);

                ch.Trace("Scoring and evaluating baseline");

                var metricDict = evaluator.Evaluate(dataEval);
                IDataView baselineMetricsView;
                if (!metricDict.TryGetValue(MetricKinds.OverallMetrics, out baselineMetricsView))
                    throw _env.Except("Evaluator did not output any overall metrics.");

                ch.AssertValue(baselineMetricsView);
                var baselineMetrics = EvaluateUtils.GetMetrics(baselineMetricsView);

                // Get slot names.
                RoleMappedData roleMapped = dataEval;
                TrainerUtils.CheckFeatureFloatVector(roleMapped);
                IDataView view = roleMapped.Data;
                int numSlots = roleMapped.Schema.Feature.Type.VectorSize;

                ch.Info("Number of slots: " + numSlots);
                if (view.Schema.HasSlotNames(roleMapped.Schema.Feature.Index, numSlots))
                    view.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, roleMapped.Schema.Feature.Index, ref _slotNames);

                if (_slotNames.Length != numSlots)
                    _slotNames = VBufferUtils.CreateEmpty<ReadOnlyMemory<char>>(numSlots);

                // Filter slots with 0 weight.
                VBuffer<Single> weights = default(VBuffer<Single>);
                List<int> workingFeatureIndices = new List<int>();
                int zeroWeightsCount = 0;

                // By default set to the number of all features available.
                evaluatedFeaturesCount = numSlots;
                if (useFeatureWeightFilter)
                {
                    var predictorWithWeights = predictor as IPredictorWithFeatureWeights<Single>;
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
                                    msgFilteredOutFeatures.Append(GetSlotName(k.Key));
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
                            weights = new VBuffer<Single>(numSlots, weights.Count, weights.Values, weights.Indices);
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
                    return null;
                }

                // Note: this will not work on the huge dataset.
                var maxSize = topExamples > 0 ? topExamples : Utils.ArrayMaxSize;
                List<Float> initialfeatureValuesList = new List<Float>();

                // Cursor through the data to cache slot 0 values for the upcoming permutation.
                var valuesRowCount = 0;
                // REVIEW olgali: Seems like if the labels are NaN, so that all metrics are NaN, this command will be useless.
                // In which case probably erroring out is probably the most useful thing.
                using (var cursor = new FloatLabelCursor(roleMapped, CursOpt.Label | CursOpt.Features | CursOpt.AllowBadEverything))
                {
                    while (initialfeatureValuesList.Count < maxSize && cursor.MoveNext())
                        initialfeatureValuesList.Add(cursor.Features.GetItemOrDefault(workingFeatureIndices[0]));

                    valuesRowCount = initialfeatureValuesList.Count;
                    ch.Assert(valuesRowCount == cursor.KeptRowCount);
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
                    input[0].ColumnName = roleMapped.Schema.Feature.Name;

                    SchemaDefinition output = SchemaDefinition.Create(typeof(FeaturesBuffer));
                    Contracts.Assert(output.Count == 1);
                    output[0].ColumnName = roleMapped.Schema.Feature.Name;
                    output[0].ColumnType = roleMapped.Schema.Feature.Type;

                    IDataView viewPermuted =
                        LambdaTransform.CreateMap<FeaturesBuffer, FeaturesBuffer, PermuterState>(host, view,
                            permuter,
                            null, input, output);
                    if (topExamples > 0 && valuesRowCount == topExamples)
                        viewPermuted = SkipTakeFilter.Create(host, new SkipTakeFilter.TakeArguments() { Count = valuesRowCount }, viewPermuted);

                    // First initialize the role mapping on the input data.
                    var dataEvalPermuted = new RoleMappedData(viewPermuted, label, features,
                        group, weight, name, customColumns, opt:true);

                    IDataScorerTransform scorePipePermuted = ScoreUtils.GetScorer(scorer, predictor, viewPermuted,
                        features,
                        group, customColumns, host, dataEvalPermuted.Schema);

                    // Recast the role mappings onto the now scored data.
                    dataEvalPermuted = new RoleMappedData(scorePipePermuted, label, features,
                        group, weight, name, customColumns, opt:true);
                    var evaluatorScr = evalComp?.CreateComponent(host) ?? EvaluateUtils.GetEvaluator(host, scorePipe.Schema);
                    metricDict = evaluatorScr.Evaluate(dataEvalPermuted);
                    IDataView metricsViewPermuted;
                    if (!metricDict.TryGetValue(MetricKinds.OverallMetrics, out metricsViewPermuted))
                        throw _env.Except("Evaluator did not output any overall metrics.");

                    Contracts.AssertValue(metricsViewPermuted);
                    var metrics = EvaluateUtils.GetMetrics(metricsViewPermuted);

                    UpdateFeatureMetricStats(baselineMetrics, metrics, ch);

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

                resultView = BuildFinalReport(workingFeatureIndices);
            }

            return resultView;
        }

        /// <summary>
        /// Helper method that will calculate (featureMetric - baselineMetric) and store it the <see cref="_metricsStats"/>
        /// </summary>
        private void UpdateFeatureMetricStats(IEnumerable<KeyValuePair<string, Double>> baselineMetrics, IEnumerable<KeyValuePair<string, Double>> featureMetrics, IChannel ch)
        {
            ch.AssertValue(baselineMetrics);
            ch.AssertValue(featureMetrics);

            foreach (var baselineMetric in baselineMetrics)
            {
                var featureMetric = featureMetrics.FirstOrDefault(x => x.Key == baselineMetric.Key);
                if (!string.IsNullOrEmpty(featureMetric.Key))
                {
                    var delta = featureMetric.Value - baselineMetric.Value;

                    List<Double> metricDeltas;
                    bool needToAddKey = false;
                    if (!_metricsStats.ContainsKey(baselineMetric.Key))
                    {
                        metricDeltas = new List<Double>();
                        needToAddKey = true;
                    }
                    else
                    {
                        metricDeltas = _metricsStats[baselineMetric.Key];
                    }

                    metricDeltas.Add(delta);

                    if (needToAddKey)
                    {
                        _metricsStats.Add(baselineMetric.Key, metricDeltas);
                    }
                }
            }
        }

        /// <summary>
        /// Helper method that will add build IDV with each "row" containing info how each feature slot permutation impacted metrics
        /// </summary>
        private IDataView BuildFinalReport(List<int> workingFeatureIndices)
        {
            var indices = workingFeatureIndices.ToArray();

            IDataView resultDV = null;
            if (workingFeatureIndices.Count > 0 && _metricsStats.Count > 0)
            {
                var builder = new ArrayDataViewBuilder(_env);
                builder.AddColumn("Index", NumberType.I4, indices);
                foreach (var metricColumn in _metricsStats)
                    builder.AddColumn(metricColumn.Key + "_Delta", NumberType.R8, metricColumn.Value.ToArray());

                resultDV = builder.GetDataView();

                var schemaIn =
                new SchemaDefinition
                {
                    new SchemaDefinition.Column{MemberName = "Index", ColumnName = "Index"}
                };

                var schemaOut =
                    new SchemaDefinition
                {
                    new SchemaDefinition.Column { MemberName = "Name", ColumnName = "Feature" }
                };

                resultDV = new CustomMappingTransformer<FeatureIndex, FeatureName>(_env,
                    (featureIndex, featureName) => featureName.Name = GetSlotName(featureIndex.Index), null, schemaIn, schemaOut).Transform(resultDV);
            }

            return resultDV;
        }

        private ReadOnlyMemory<char> GetSlotName(int index)
        {
            var slotName = _slotNames.GetItemOrDefault(index);
            return !slotName.IsEmpty
                ? slotName
                : new ReadOnlyMemory<char>(string.Format("f{0}", index).ToCharArray());
        }
    }
}