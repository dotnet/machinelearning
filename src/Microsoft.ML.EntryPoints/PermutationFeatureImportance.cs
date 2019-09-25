// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(void), typeof(PermutationFeatureImportanceEntryPoints), null, typeof(SignatureEntryPointModule), "PermutationFeatureImportance")]

namespace Microsoft.ML.Transforms
{
    internal static class PermutationFeatureImportanceEntryPoints
    {
        [TlcModule.EntryPoint(Name = "Transforms.PermutationFeatureImportance", Desc = "Permutation Feature Importance (PFI)", UserName = "PFI", ShortName = "PFI")]
        public static PermutationFeatureImportanceOutput PermutationFeatureImportance(IHostEnvironment env, PermutationFeatureImportanceArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Pfi");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var modelOps = new ModelOperationsCatalog(env);
            var model = modelOps.Load(input.ModelPath.OpenReadStream(), out DataViewSchema schema) as TransformerChain<ITransformer>;
            // If model.LastTransformer is not an IPredictionTransformer, get the part of the TransformerChain
            // up to the last ITransformer that is indeed an IPredictionTransformer. This piece of the TransformerChain
            // is used to extract the IPredictionTransformer and also to transform the input data.
            // Will throw if there is no IPredictionTransformer in the TransformerChain.
            if (!(model.LastTransformer is IPredictionTransformer<IPredictorProducing<float>>
                || model.LastTransformer is IPredictionTransformer<IPredictorProducing<VBuffer<float>>>))
                model = model.RewindToLastPredictionTransformer();
            var predictor = model.LastTransformer as ISingleFeaturePredictionTransformer<object>;
            Contracts.Assert(predictor != null, "Permutation Feature Importance (PFI) is not supported for the predictor.");

            var transformedData = model.Transform(input.Data);
            IDataView result = PermutationFeatureImportanceUtils.GetMetrics(env, predictor, transformedData, input);
            return new PermutationFeatureImportanceOutput { Metrics = result };
        }
    }

    internal sealed class PermutationFeatureImportanceOutput
    {
        [TlcModule.Output(Desc = "The PFI metrics")]
        public IDataView Metrics;
    }

    internal sealed class PermutationFeatureImportanceArguments : TransformInputBase
    {
        [Argument(ArgumentType.Required, HelpText = "The path to the model file", ShortName = "path", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public IFileHandle ModelPath;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Label column name", ShortName = "label", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string LabelColumnName = "Label";

        [Argument(ArgumentType.AtMostOnce, HelpText = "Group ID column", ShortName = "groupId", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string RowGroupColumnName = "GroupId";

        [Argument(ArgumentType.AtMostOnce, HelpText = "Use feature weights to pre-filter features", ShortName = "usefw", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public bool UseFeatureWeightFilter = false;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Limit the number of examples to evaluate on", ShortName = "numexamples", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public int? NumberOfExamplesToUse = null;

        [Argument(ArgumentType.AtMostOnce, HelpText = "The number of permutations to perform", ShortName = "permutations", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public int PermutationCount = 1;
    }

    internal static class PermutationFeatureImportanceUtils
    {
        internal static IDataView GetMetrics(
            IHostEnvironment env,
            ISingleFeaturePredictionTransformer<object> predictor,
            IDataView data,
            PermutationFeatureImportanceArguments input)
        {
            IDataView result;
            if (predictor is BinaryPredictionTransformer<IPredictorProducing<float>>)
                result = GetBinaryMetrics(env, predictor, data, input);
            else if (predictor is MulticlassPredictionTransformer<IPredictorProducing<VBuffer<float>>>)
                result = GetMulticlassMetrics(env, predictor, data, input);
            else if (predictor is RegressionPredictionTransformer<IPredictorProducing<float>>)
                result = GetRegressionMetrics(env, predictor, data, input);
            else if (predictor is RankingPredictionTransformer<IPredictorProducing<float>>)
                result = GetRankingMetrics(env, predictor, data, input);
            else
                throw Contracts.Except(
                    "Unsupported predictor type. Predictor must be binary classifier, " +
                    "multiclass classifier, regressor, or ranker.");

            return result;
        }

        private static IDataView GetBinaryMetrics(
            IHostEnvironment env,
            ISingleFeaturePredictionTransformer<object> predictor,
            IDataView data,
            PermutationFeatureImportanceArguments input)
        {
            var binaryCatalog = new BinaryClassificationCatalog(env);
            var permutationMetrics = binaryCatalog
                .PermutationFeatureImportance(predictor,
                                              data,
                                              labelColumnName: input.LabelColumnName,
                                              useFeatureWeightFilter: input.UseFeatureWeightFilter,
                                              numberOfExamplesToUse: input.NumberOfExamplesToUse,
                                              permutationCount: input.PermutationCount);

            var slotNames = GetSlotNames(data);
            Contracts.Assert(slotNames.Length == permutationMetrics.Length,
                "Mismatch between number of feature slots and number of features permuted.");

            List<BinaryMetrics> metrics = new List<BinaryMetrics>();
            for (int i = 0; i < permutationMetrics.Length; i++)
            {
                var pMetric = permutationMetrics[i];
                metrics.Add(new BinaryMetrics
                {
                    FeatureName = slotNames[i],
                    AreaUnderRocCurve = pMetric.AreaUnderRocCurve.Mean,
                    AreaUnderRocCurveStdErr = pMetric.AreaUnderRocCurve.StandardError,
                    Accuracy = pMetric.Accuracy.Mean,
                    AccuracyStdErr = pMetric.Accuracy.StandardError,
                    PositivePrecision = pMetric.PositivePrecision.Mean,
                    PositivePrecisionStdErr = pMetric.PositivePrecision.StandardError,
                    PositiveRecall = pMetric.PositiveRecall.Mean,
                    PositiveRecallStdErr = pMetric.PositiveRecall.StandardError,
                    NegativePrecision = pMetric.NegativePrecision.Mean,
                    NegativePrecisionStdErr = pMetric.NegativePrecision.StandardError,
                    NegativeRecall = pMetric.NegativeRecall.Mean,
                    NegativeRecallStdErr = pMetric.NegativeRecall.StandardError,
                    F1Score = pMetric.F1Score.Mean,
                    F1ScoreStdErr = pMetric.F1Score.StandardError,
                    AreaUnderPrecisionRecallCurve = pMetric.AreaUnderPrecisionRecallCurve.Mean,
                    AreaUnderPrecisionRecallCurveStdErr = pMetric.AreaUnderPrecisionRecallCurve.StandardError
                });
            }

            var dataOps = new DataOperationsCatalog(env);
            var result = dataOps.LoadFromEnumerable(metrics);
            return result;
        }

        private static IDataView GetMulticlassMetrics(
            IHostEnvironment env,
            ISingleFeaturePredictionTransformer<object> predictor,
            IDataView data,
            PermutationFeatureImportanceArguments input)
        {
            var multiclassCatalog = new MulticlassClassificationCatalog(env);
            var permutationMetrics = multiclassCatalog
                .PermutationFeatureImportance(predictor,
                                              data,
                                              labelColumnName: input.LabelColumnName,
                                              useFeatureWeightFilter: input.UseFeatureWeightFilter,
                                              numberOfExamplesToUse: input.NumberOfExamplesToUse,
                                              permutationCount: input.PermutationCount);

            var slotNames = GetSlotNames(data);
            Contracts.Assert(slotNames.Length == permutationMetrics.Length,
                "Mismatch between number of feature slots and number of features permuted.");

            List<MulticlassMetrics> metrics = new List<MulticlassMetrics>();
            for (int i = 0; i < permutationMetrics.Length; i++)
            {
                var pMetric = permutationMetrics[i];
                metrics.Add(new MulticlassMetrics
                {
                    FeatureName = slotNames[i],
                    MacroAccuracy = pMetric.MacroAccuracy.Mean,
                    MacroAccuracyStdErr = pMetric.MacroAccuracy.StandardError,
                    MicroAccuracy = pMetric.MicroAccuracy.Mean,
                    MicroAccuracyStdErr = pMetric.MicroAccuracy.StandardError,
                    LogLoss = pMetric.LogLoss.Mean,
                    LogLossStdErr = pMetric.LogLoss.StandardError,
                    LogLossReduction = pMetric.LogLossReduction.Mean,
                    LogLossReductionStdErr = pMetric.LogLossReduction.StandardError,
                    TopKAccuracy = pMetric.TopKAccuracy.Mean,
                    TopKAccuracyStdErr = pMetric.TopKAccuracy.StandardError,
                    PerClassLogLoss = pMetric.PerClassLogLoss.Select(x => x.Mean).ToArray(),
                    PerClassLogLossStdErr = pMetric.PerClassLogLoss.Select(x => x.StandardError).ToArray()
                }); ;
            }

            // Convert unknown size vectors to known size.
            var metric = metrics.First();
            SchemaDefinition schema = SchemaDefinition.Create(typeof(MulticlassMetrics));
            ConvertVectorToKnownSize(nameof(metric.PerClassLogLoss), metric.PerClassLogLoss.Length, ref schema);
            ConvertVectorToKnownSize(nameof(metric.PerClassLogLossStdErr), metric.PerClassLogLossStdErr.Length, ref schema);

            var dataOps = new DataOperationsCatalog(env);
            var result = dataOps.LoadFromEnumerable(metrics, schema);
            return result;
        }

        private static IDataView GetRegressionMetrics(
            IHostEnvironment env,
            ISingleFeaturePredictionTransformer<object> predictor,
            IDataView data,
            PermutationFeatureImportanceArguments input)
        {
            var regressionCatalog = new RegressionCatalog(env);
            var permutationMetrics = regressionCatalog
                .PermutationFeatureImportance(predictor,
                                              data,
                                              labelColumnName: input.LabelColumnName,
                                              useFeatureWeightFilter: input.UseFeatureWeightFilter,
                                              numberOfExamplesToUse: input.NumberOfExamplesToUse,
                                              permutationCount: input.PermutationCount);

            var slotNames = GetSlotNames(data);
            Contracts.Assert(slotNames.Length == permutationMetrics.Length,
                "Mismatch between number of feature slots and number of features permuted.");

            List<RegressionMetrics> metrics = new List<RegressionMetrics>();
            for (int i = 0; i < permutationMetrics.Length; i++)
            {
                var pMetric = permutationMetrics[i];
                metrics.Add(new RegressionMetrics
                {
                    FeatureName = slotNames[i],
                    MeanAbsoluteError = pMetric.MeanAbsoluteError.Mean,
                    MeanAbsoluteErrorStdErr = pMetric.MeanAbsoluteError.StandardError,
                    MeanSquaredError = pMetric.MeanSquaredError.Mean,
                    MeanSquaredErrorStdErr = pMetric.MeanSquaredError.StandardError,
                    RootMeanSquaredError = pMetric.RootMeanSquaredError.Mean,
                    RootMeanSquaredErrorStdErr = pMetric.RootMeanSquaredError.StandardError,
                    LossFunction = pMetric.LossFunction.Mean,
                    LossFunctionStdErr = pMetric.LossFunction.StandardError,
                    RSquared = pMetric.RSquared.Mean,
                    RSquaredStdErr = pMetric.RSquared.StandardError
                });
            }

            var dataOps = new DataOperationsCatalog(env);
            var result = dataOps.LoadFromEnumerable(metrics);
            return result;
        }

        private static IDataView GetRankingMetrics(
            IHostEnvironment env,
            ISingleFeaturePredictionTransformer<object> predictor,
            IDataView data,
            PermutationFeatureImportanceArguments input)
        {
            var rankingCatalog = new RankingCatalog(env);
            var permutationMetrics = rankingCatalog
                .PermutationFeatureImportance(predictor,
                                              data,
                                              labelColumnName: input.LabelColumnName,
                                              rowGroupColumnName: input.RowGroupColumnName,
                                              useFeatureWeightFilter: input.UseFeatureWeightFilter,
                                              numberOfExamplesToUse: input.NumberOfExamplesToUse,
                                              permutationCount: input.PermutationCount);

            var slotNames = GetSlotNames(data);
            Contracts.Assert(slotNames.Length == permutationMetrics.Length,
                "Mismatch between number of feature slots and number of features permuted.");

            List<RankingMetrics> metrics = new List<RankingMetrics>();
            for (int i = 0; i < permutationMetrics.Length; i++)
            {
                var pMetric = permutationMetrics[i];
                metrics.Add(new RankingMetrics
                {
                    FeatureName = slotNames[i],
                    DiscountedCumulativeGains = pMetric.DiscountedCumulativeGains.Select(x => x.Mean).ToArray(),
                    DiscountedCumulativeGainsStdErr = pMetric.DiscountedCumulativeGains.Select(x => x.StandardError).ToArray(),
                    NormalizedDiscountedCumulativeGains = pMetric.NormalizedDiscountedCumulativeGains.Select(x => x.Mean).ToArray(),
                    NormalizedDiscountedCumulativeGainsStdErr = pMetric.NormalizedDiscountedCumulativeGains.Select(x => x.StandardError).ToArray()
                });
            }

            // Convert unknown size vectors to known size.
            var metric = metrics.First();
            SchemaDefinition schema = SchemaDefinition.Create(typeof(RankingMetrics));
            ConvertVectorToKnownSize(nameof(metric.DiscountedCumulativeGains), metric.DiscountedCumulativeGains.Length, ref schema);
            ConvertVectorToKnownSize(nameof(metric.NormalizedDiscountedCumulativeGains), metric.NormalizedDiscountedCumulativeGains.Length, ref schema);
            ConvertVectorToKnownSize(nameof(metric.DiscountedCumulativeGainsStdErr), metric.DiscountedCumulativeGainsStdErr.Length, ref schema);
            ConvertVectorToKnownSize(nameof(metric.NormalizedDiscountedCumulativeGainsStdErr), metric.NormalizedDiscountedCumulativeGainsStdErr.Length, ref schema);

            var dataOps = new DataOperationsCatalog(env);
            var result = dataOps.LoadFromEnumerable(metrics, schema);
            return result;
        }

        private static string[] GetSlotNames(IDataView data)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;
            data.Schema["Features"].GetSlotNames(ref slots);

            var column = data.GetColumn<VBuffer<float>>(
                data.Schema["Features"]);

            List<string> slotNames = new List<string>();

            foreach (var item in column.First<VBuffer<float>>().Items(all: true))
            {
                slotNames.Add(slots.GetValues()[item.Key].ToString());
            };

            return slotNames.ToArray();
        }

        private static void ConvertVectorToKnownSize(string metricName, int size, ref SchemaDefinition schema)
        {
            var type = ((VectorDataViewType)schema[metricName].ColumnType).ItemType;
            schema[metricName].ColumnType = new VectorDataViewType(type, size);
        }

        private class BinaryMetrics
        {
            public string FeatureName { get; set; }

            public double AreaUnderRocCurve { get; set; }

            public double AreaUnderRocCurveStdErr { get; set; }

            public double Accuracy { get; set; }

            public double AccuracyStdErr { get; set; }

            public double PositivePrecision { get; set; }

            public double PositivePrecisionStdErr { get; set; }

            public double PositiveRecall { get; set; }

            public double PositiveRecallStdErr { get; set; }

            public double NegativePrecision { get; set; }

            public double NegativePrecisionStdErr { get; set; }

            public double NegativeRecall { get; set; }

            public double NegativeRecallStdErr { get; set; }

            public double F1Score { get; set; }

            public double F1ScoreStdErr { get; set; }

            public double AreaUnderPrecisionRecallCurve { get; set; }

            public double AreaUnderPrecisionRecallCurveStdErr { get; set; }
        }

        private class MulticlassMetrics
        {
            public string FeatureName { get; set; }

            public double MacroAccuracy { get; set; }

            public double MacroAccuracyStdErr { get; set; }

            public double MicroAccuracy { get; set; }

            public double MicroAccuracyStdErr { get; set; }

            public double LogLoss { get; set; }

            public double LogLossStdErr { get; set; }

            public double LogLossReduction { get; set; }

            public double LogLossReductionStdErr { get; set; }

            public double TopKAccuracy { get; set; }

            public double TopKAccuracyStdErr { get; set; }

            public double[] PerClassLogLoss { get; set; }

            public double[] PerClassLogLossStdErr { get; set; }
        }

        private class RegressionMetrics
        {
            public string FeatureName { get; set; }

            public double MeanAbsoluteError { get; set; }

            public double MeanAbsoluteErrorStdErr { get; set; }

            public double MeanSquaredError { get; set; }

            public double MeanSquaredErrorStdErr { get; set; }

            public double RootMeanSquaredError { get; set; }

            public double RootMeanSquaredErrorStdErr { get; set; }

            public double LossFunction { get; set; }

            public double LossFunctionStdErr { get; set; }

            public double RSquared { get; set; }

            public double RSquaredStdErr { get; set; }
        }

        private class RankingMetrics
        {
            public string FeatureName { get; set; }

            public double[] DiscountedCumulativeGains { get; set; }

            public double[] DiscountedCumulativeGainsStdErr { get; set; }

            public double[] NormalizedDiscountedCumulativeGains { get; set; }

            public double[] NormalizedDiscountedCumulativeGainsStdErr { get; set; }
        }
    }
}
