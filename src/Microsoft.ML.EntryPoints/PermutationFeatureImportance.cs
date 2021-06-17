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

            input.PredictorModel.PrepareData(env, input.Data, out RoleMappedData roleMappedData, out IPredictor predictor);
            Contracts.Assert(predictor != null, "No predictor found in model");
            IDataView result = PermutationFeatureImportanceUtils.GetMetrics(env, predictor, roleMappedData, input);
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
        public PredictorModel PredictorModel;

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
            IPredictor predictor,
            RoleMappedData roleMappedData,
            PermutationFeatureImportanceArguments input)
        {
            Contracts.Check(roleMappedData.Schema.Feature != null, "Feature column not found.");
            Contracts.Check(roleMappedData.Schema.Label != null, "Label column not found.");
            IDataView result;
            if (predictor.PredictionKind == PredictionKind.BinaryClassification)
                result = GetBinaryMetrics(env, predictor, roleMappedData, input);
            else if (predictor.PredictionKind == PredictionKind.MulticlassClassification)
                result = GetMulticlassMetrics(env, predictor, roleMappedData, input);
            else if (predictor.PredictionKind == PredictionKind.Regression)
                result = GetRegressionMetrics(env, predictor, roleMappedData, input);
            else if (predictor.PredictionKind == PredictionKind.Ranking)
                result = GetRankingMetrics(env, predictor, roleMappedData, input);
            else
                throw Contracts.Except(
                    "Unsupported predictor type. Predictor must be binary classifier, " +
                    "multiclass classifier, regressor, or ranker.");

            return result;
        }

        private static IDataView GetBinaryMetrics(
            IHostEnvironment env,
            IPredictor predictor,
            RoleMappedData roleMappedData,
            PermutationFeatureImportanceArguments input)
        {
            var roles = roleMappedData.Schema.GetColumnRoleNames();
            var featureColumnName = roles.Where(x => x.Key.Value == RoleMappedSchema.ColumnRole.Feature.Value).First().Value;
            var labelColumnName = roles.Where(x => x.Key.Value == RoleMappedSchema.ColumnRole.Label.Value).First().Value;
            var pred = new BinaryPredictionTransformer<IPredictorProducing<float>>(
                env, predictor as IPredictorProducing<float>, roleMappedData.Data.Schema, featureColumnName);
            var binaryCatalog = new BinaryClassificationCatalog(env);
            var permutationMetrics = binaryCatalog
                .PermutationFeatureImportance(pred,
                                              roleMappedData.Data,
                                              labelColumnName: labelColumnName,
                                              useFeatureWeightFilter: input.UseFeatureWeightFilter,
                                              numberOfExamplesToUse: input.NumberOfExamplesToUse,
                                              permutationCount: input.PermutationCount);

            var slotNames = GetSlotNames(roleMappedData.Schema);
            Contracts.Assert(slotNames.Length == permutationMetrics.Length,
                "Mismatch between number of feature slots and number of features permuted.");

            List<BinaryMetrics> metrics = new List<BinaryMetrics>();
            for (int i = 0; i < permutationMetrics.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(slotNames[i]))
                    continue;
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
            IPredictor predictor,
            RoleMappedData roleMappedData,
            PermutationFeatureImportanceArguments input)
        {
            var roles = roleMappedData.Schema.GetColumnRoleNames();
            var featureColumnName = roles.Where(x => x.Key.Value == RoleMappedSchema.ColumnRole.Feature.Value).First().Value;
            var labelColumnName = roles.Where(x => x.Key.Value == RoleMappedSchema.ColumnRole.Label.Value).First().Value;
            var pred = new MulticlassPredictionTransformer<IPredictorProducing<VBuffer<float>>>(
                env, predictor as IPredictorProducing<VBuffer<float>>, roleMappedData.Data.Schema, featureColumnName, labelColumnName);
            var multiclassCatalog = new MulticlassClassificationCatalog(env);
            var permutationMetrics = multiclassCatalog
                .PermutationFeatureImportance(pred,
                                              roleMappedData.Data,
                                              labelColumnName: labelColumnName,
                                              useFeatureWeightFilter: input.UseFeatureWeightFilter,
                                              numberOfExamplesToUse: input.NumberOfExamplesToUse,
                                              permutationCount: input.PermutationCount);

            var slotNames = GetSlotNames(roleMappedData.Schema);
            Contracts.Assert(slotNames.Length == permutationMetrics.Length,
                "Mismatch between number of feature slots and number of features permuted.");

            List<MulticlassMetrics> metrics = new List<MulticlassMetrics>();
            for (int i = 0; i < permutationMetrics.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(slotNames[i]))
                    continue;
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
                });
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
            IPredictor predictor,
            RoleMappedData roleMappedData,
            PermutationFeatureImportanceArguments input)
        {
            var roles = roleMappedData.Schema.GetColumnRoleNames();
            var featureColumnName = roles.Where(x => x.Key.Value == RoleMappedSchema.ColumnRole.Feature.Value).First().Value;
            var labelColumnName = roles.Where(x => x.Key.Value == RoleMappedSchema.ColumnRole.Label.Value).First().Value;
            var pred = new RegressionPredictionTransformer<IPredictorProducing<float>>(
                env, predictor as IPredictorProducing<float>, roleMappedData.Data.Schema, featureColumnName);
            var regressionCatalog = new RegressionCatalog(env);
            var permutationMetrics = regressionCatalog
                .PermutationFeatureImportance(pred,
                                              roleMappedData.Data,
                                              labelColumnName: labelColumnName,
                                              useFeatureWeightFilter: input.UseFeatureWeightFilter,
                                              numberOfExamplesToUse: input.NumberOfExamplesToUse,
                                              permutationCount: input.PermutationCount);

            var slotNames = GetSlotNames(roleMappedData.Schema);
            Contracts.Assert(slotNames.Length == permutationMetrics.Length,
                "Mismatch between number of feature slots and number of features permuted.");

            List<RegressionMetrics> metrics = new List<RegressionMetrics>();
            for (int i = 0; i < permutationMetrics.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(slotNames[i]))
                    continue;
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
            IPredictor predictor,
            RoleMappedData roleMappedData,
            PermutationFeatureImportanceArguments input)
        {
            Contracts.Check(roleMappedData.Schema.Group != null, "Group ID column not found.");
            var roles = roleMappedData.Schema.GetColumnRoleNames();
            var featureColumnName = roles.Where(x => x.Key.Value == RoleMappedSchema.ColumnRole.Feature.Value).First().Value;
            var labelColumnName = roles.Where(x => x.Key.Value == RoleMappedSchema.ColumnRole.Label.Value).First().Value;
            var groupIdColumnName = roles.Where(x => x.Key.Value == RoleMappedSchema.ColumnRole.Group.Value).First().Value;
            var pred = new RankingPredictionTransformer<IPredictorProducing<float>>(
                env, predictor as IPredictorProducing<float>, roleMappedData.Data.Schema, featureColumnName);
            var rankingCatalog = new RankingCatalog(env);
            var permutationMetrics = rankingCatalog
                .PermutationFeatureImportance(pred,
                                              roleMappedData.Data,
                                              labelColumnName: labelColumnName,
                                              rowGroupColumnName: groupIdColumnName,
                                              useFeatureWeightFilter: input.UseFeatureWeightFilter,
                                              numberOfExamplesToUse: input.NumberOfExamplesToUse,
                                              permutationCount: input.PermutationCount);

            var slotNames = GetSlotNames(roleMappedData.Schema);
            Contracts.Assert(slotNames.Length == permutationMetrics.Length,
                "Mismatch between number of feature slots and number of features permuted.");

            List<RankingMetrics> metrics = new List<RankingMetrics>();
            for (int i = 0; i < permutationMetrics.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(slotNames[i]))
                    continue;
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

        private static string[] GetSlotNames(RoleMappedSchema schema)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;
            schema.Feature.Value.GetSlotNames(ref slots);
            var slotValues = slots.DenseValues();

            List<string> slotNames = new List<string>();
            foreach (var value in slotValues)
            {
                slotNames.Add(value.ToString());
            }

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
