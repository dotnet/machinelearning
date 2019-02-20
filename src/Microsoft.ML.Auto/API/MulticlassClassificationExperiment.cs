// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public class MulticlassExperimentSettings : ExperimentSettings
    {
        public IProgress<RunResult<MultiClassClassifierMetrics>> ProgressCallback;
        public MulticlassClassificationMetric OptimizingMetric = MulticlassClassificationMetric.AccuracyMicro;
        public MulticlassClassificationTrainer[] WhitelistedTrainers;
    }

    public enum MulticlassClassificationMetric
    {
        AccuracyMicro,
        AccuracyMacro,
        LogLoss,
        LogLossReduction,
        TopKAccuracy,
    }

    public enum MulticlassClassificationTrainer
    {
        AveragedPerceptronOVA,
        FastForestOVA,
        FastTreeOVA,
        LightGbm,
        LinearSupportVectorMachinesOVA,
        LogisticRegression,
        LogisticRegressionOVA,
        StochasticDualCoordinateAscent,
        StochasticGradientDescentOVA,
        SymbolicStochasticGradientDescentOVA,
    }

    public class MulticlassClassificationExperiment
    {
        private readonly MLContext _context;
        private readonly MulticlassExperimentSettings _settings;

        internal MulticlassClassificationExperiment(MLContext context, MulticlassExperimentSettings settings)
        {
            _context = context;
            _settings = settings;
        }

        public IEnumerable<RunResult<MultiClassClassifierMetrics>> Execute(IDataView trainData, string labelColumn = DefaultColumnNames.Label, IEstimator<ITransformer> preFeaturizers = null)
        {
            var columnInformation = new ColumnInformation() { LabelColumn = labelColumn };
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<MultiClassClassifierMetrics>> Execute(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<MultiClassClassifierMetrics>> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, validationData, preFeaturizers);
        }

        internal IEnumerable<RunResult<MultiClassClassifierMetrics>> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            throw new NotImplementedException();
        }

        internal IEnumerable<RunResult<MultiClassClassifierMetrics>> Execute(MLContext context,
            IDataView trainData,
            ColumnInformation columnInfo,
            IDataView validationData = null,
            IEstimator<ITransformer> preFeaturizers = null)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            //UserInputValidationUtil.ValidateAutoFitArgs(trainData, labelColunName, validationData, settings, columnPurposes)

            // run autofit & get all pipelines run in that process
            var autoFitter = new AutoFitter<MultiClassClassifierMetrics>(context, TaskKind.MulticlassClassification, trainData, 
                columnInfo, validationData, preFeaturizers, new OptimizingMetricInfo(_settings.OptimizingMetric),
                _settings.ProgressCallback, _settings, new MultiMetricsAgent(_settings.OptimizingMetric),
                TrainerExtensionUtil.GetTrainerNames(_settings.WhitelistedTrainers));

            return autoFitter.Fit();
        }
    }

    public static class MulticlassExperimentResultExtensions
    {
        public static RunResult<MultiClassClassifierMetrics> Best(this IEnumerable<RunResult<MultiClassClassifierMetrics>> results, MulticlassClassificationMetric metric = MulticlassClassificationMetric.AccuracyMicro)
        {
            var metricsAgent = new MultiMetricsAgent(metric);
            double maxScore = results.Select(r => metricsAgent.GetScore(r.Metrics)).Max();
            return results.First(r => metricsAgent.GetScore(r.Metrics) == maxScore);
        }
    }
}
