// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubModelSelector
{
    public abstract class BaseSubModelSelector<TOutput> : ISubModelSelector<TOutput>
    {
        protected readonly IHost Host;

        public abstract Single ValidationDatasetProportion { get; }

        protected abstract PredictionKind PredictionKind { get; }

        protected BaseSubModelSelector(IHostEnvironment env, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
        }

        protected void Print(IChannel ch, IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> models, string metricName)
        {
            // REVIEW: The output format was faithfully reproduced from the original format, but it's unclear
            // to me that this is right. Why have two bars in the header line, but only one bar in the results?
            ch.Info("List of models and the metrics after sorted");
            ch.Info("| {0}(Sorted) || Name of Model |", metricName);
            foreach (var model in models)
            {
                var metric = 0.0;
                var found = false;
                foreach (var kvp in model.Metrics)
                {
                    if (kvp.Key == metricName)
                    {
                        metric = kvp.Value;
                        found = true;
                    }
                }
                if (!found)
                    throw ch.Except("Metrics did not contain the requested metric '{0}'", metricName);
                ch.Info("| {0} |{1}", metric, model.Predictor.GetType().Name);
            }
        }

        public virtual IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> Prune(IList<FeatureSubsetModel<IPredictorProducing<TOutput>>> models)
        {
            return models;
        }

        private SubComponent<IEvaluator, SignatureEvaluator> GetEvaluatorSubComponent()
        {
            switch (PredictionKind)
            {
                case PredictionKind.BinaryClassification:
                    return new SubComponent<IEvaluator, SignatureEvaluator>(BinaryClassifierEvaluator.LoadName);
                case PredictionKind.Regression:
                    return new SubComponent<IEvaluator, SignatureEvaluator>(RegressionEvaluator.LoadName);
                case PredictionKind.MultiClassClassification:
                    return new SubComponent<IEvaluator, SignatureEvaluator>(MultiClassClassifierEvaluator.LoadName);
                default:
                    throw Host.Except("Unrecognized prediction kind '{0}'", PredictionKind);
            }
        }

        public virtual void CalculateMetrics(FeatureSubsetModel<IPredictorProducing<TOutput>> model,
            ISubsetSelector subsetSelector, Subset subset, Batch batch, bool needMetrics)
        {
            if (!needMetrics || model == null || model.Metrics != null)
                return;

            using (var ch = Host.Start("Calculate metrics"))
            {
                RoleMappedData testData = subsetSelector.GetTestData(subset, batch);
                // Because the training and test datasets are drawn from the same base dataset, the test data role mappings
                // are the same as for the train data.
                IDataScorerTransform scorePipe = ScoreUtils.GetScorer(model.Predictor, testData, Host, testData.Schema);
                // REVIEW: Should we somehow allow the user to customize the evaluator?
                // By what mechanism should we allow that?
                var evalComp = GetEvaluatorSubComponent();
                RoleMappedData scoredTestData = new RoleMappedData(scorePipe,
                    GetColumnRoles(testData.Schema, scorePipe.Schema));
                IEvaluator evaluator = evalComp.CreateInstance(Host);
                // REVIEW: with the new evaluators, metrics of individual models are no longer
                // printed to the Console. Consider adding an option on the combiner to print them.
                // REVIEW: Consider adding an option to the combiner to save a data view
                // containing all the results of the individual models.
                var metricsDict = evaluator.Evaluate(scoredTestData);
                if (!metricsDict.TryGetValue(MetricKinds.OverallMetrics, out IDataView metricsView))
                    throw Host.Except("Evaluator did not produce any overall metrics");
                // REVIEW: We're assuming that the metrics of interest are always doubles here.
                var metrics = EvaluateUtils.GetMetrics(metricsView, getVectorMetrics: false);
                model.Metrics = metrics.ToArray();
                ch.Done();
            }
        }

        private IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetColumnRoles(
            RoleMappedSchema testSchema, ISchema scoredSchema)
        {
            switch (PredictionKind)
            {
                case PredictionKind.BinaryClassification:
                    yield return RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Label, testSchema.Label.Name);
                    var scoreInfo = EvaluateUtils.GetScoreColumnInfo(Host, scoredSchema, null, nameof(BinaryClassifierMamlEvaluator.ArgumentsBase.ScoreColumn),
                        MetadataUtils.Const.ScoreColumnKind.BinaryClassification);
                    yield return RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, scoreInfo.Name);
                    // Get the optional probability column.
                    var probInfo = EvaluateUtils.GetOptAuxScoreColumnInfo(Host, scoredSchema, null, nameof(BinaryClassifierMamlEvaluator.Arguments.ProbabilityColumn),
                        scoreInfo.Index, MetadataUtils.Const.ScoreValueKind.Probability, t => t == NumberType.Float);
                    if (probInfo != null)
                        yield return RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Probability, probInfo.Name);
                    yield break;
                case PredictionKind.Regression:
                    yield return RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Label, testSchema.Label.Name);
                    scoreInfo = EvaluateUtils.GetScoreColumnInfo(Host, scoredSchema, null, nameof(RegressionMamlEvaluator.Arguments.ScoreColumn),
                        MetadataUtils.Const.ScoreColumnKind.Regression);
                    yield return RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, scoreInfo.Name);
                    yield break;
                case PredictionKind.MultiClassClassification:
                    yield return RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Label, testSchema.Label.Name);
                    scoreInfo = EvaluateUtils.GetScoreColumnInfo(Host, scoredSchema, null, nameof(MultiClassMamlEvaluator.Arguments.ScoreColumn),
                        MetadataUtils.Const.ScoreColumnKind.MultiClassClassification);
                    yield return RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, scoreInfo.Name);
                    yield break;
                default:
                    throw Host.Except("Unrecognized prediction kind '{0}'", PredictionKind);
            }
        }
    }
}
