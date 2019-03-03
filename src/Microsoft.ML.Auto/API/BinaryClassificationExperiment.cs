// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public sealed class BinaryExperimentSettings : ExperimentSettings
    {
        public BinaryClassificationMetric OptimizingMetric { get; set; } = BinaryClassificationMetric.Accuracy;
        public ICollection<BinaryClassificationTrainer> Trainers { get; } =
                    Enum.GetValues(typeof(BinaryClassificationTrainer)).OfType<BinaryClassificationTrainer>().ToList();
        public IProgress<RunResult<BinaryClassificationMetrics>> ProgressHandler { get; set; }
    }

    public enum BinaryClassificationMetric
    {
        Accuracy,
        AreaUnderRocCurve,
        AreaUnderPrecisionRecallCurve,
        F1Score,
        PositivePrecision,
        PositiveRecall,
        NegativePrecision,
        NegativeRecall,
    }

    public enum BinaryClassificationTrainer
    {
        AveragedPerceptron,
        FastForest,
        FastTree,
        LightGbm,
        LinearSupportVectorMachines,
        LogisticRegression,
        StochasticDualCoordinateAscent,
        StochasticGradientDescent,
        SymbolicStochasticGradientDescent,
    }

    public sealed class BinaryClassificationExperiment
    {
        private readonly MLContext _context;
        private readonly BinaryExperimentSettings _settings;

        internal BinaryClassificationExperiment(MLContext context, BinaryExperimentSettings settings)
        {
            _context = context;
            _settings = settings;
        }

        public IEnumerable<RunResult<BinaryClassificationMetrics>> Execute(IDataView trainData, string labelColumn = DefaultColumnNames.Label, IEstimator<ITransformer> preFeaturizers = null)
        {
            var columnInformation = new ColumnInformation() { LabelColumn = labelColumn };
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<BinaryClassificationMetrics>> Execute(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<BinaryClassificationMetrics>> Execute(IDataView trainData, IDataView validationData, string labelColumn = DefaultColumnNames.Label, IEstimator<ITransformer> preFeaturizers = null)
        {
            var columnInformation = new ColumnInformation() { LabelColumn = labelColumn };
            return Execute(_context, trainData, columnInformation, validationData, preFeaturizers);
        }

        public IEnumerable<RunResult<BinaryClassificationMetrics>> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, validationData, preFeaturizers);
        }

        internal IEnumerable<RunResult<BinaryClassificationMetrics>> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            throw new NotImplementedException();
        }

        internal IEnumerable<RunResult<BinaryClassificationMetrics>> Execute(MLContext context,
            IDataView trainData,
            ColumnInformation columnInfo,
            IDataView validationData = null,
            IEstimator<ITransformer> preFeaturizers = null)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData, columnInfo, validationData);

            // run autofit & get all pipelines run in that process
            var experiment = new Experiment<BinaryClassificationMetrics>(context, TaskKind.BinaryClassification, trainData, columnInfo, 
                validationData, preFeaturizers, new OptimizingMetricInfo(_settings.OptimizingMetric), _settings.ProgressHandler, 
                _settings, new BinaryMetricsAgent(_settings.OptimizingMetric), 
                TrainerExtensionUtil.GetTrainerNames(_settings.Trainers));

            return experiment.Execute();
        }
    }

    public static class BinaryExperimentResultExtensions
    {
        public static RunResult<BinaryClassificationMetrics> Best(this IEnumerable<RunResult<BinaryClassificationMetrics>> results, BinaryClassificationMetric metric = BinaryClassificationMetric.Accuracy)
        {
            var metricsAgent = new BinaryMetricsAgent(metric);
            return RunResultUtil.GetBestRunResult(results, metricsAgent);
        }
    }
}
