// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public sealed class MulticlassExperimentSettings : ExperimentSettings
    {
        public MulticlassClassificationMetric OptimizingMetric { get; set; } = MulticlassClassificationMetric.MicroAccuracy;
        public ICollection<MulticlassClassificationTrainer> Trainers { get; } =
            Enum.GetValues(typeof(MulticlassClassificationTrainer)).OfType<MulticlassClassificationTrainer>().ToList();
        public IProgress<RunResult<MulticlassClassificationMetrics>> ProgressHandler { get; set; }
    }

    public enum MulticlassClassificationMetric
    {
        MicroAccuracy,
        MacroAccuracy,
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
        LbfgsMaximumEntropy,
        LbfgsLogisticRegressionOVA,
        SdcaMaximumEntropy,
        SgdCalibratedOVA,
        SymbolicSgdLogisticRegressionOVA,
    }

    public sealed class MulticlassClassificationExperiment
    {
        private readonly MLContext _context;
        private readonly MulticlassExperimentSettings _settings;

        internal MulticlassClassificationExperiment(MLContext context, MulticlassExperimentSettings settings)
        {
            _context = context;
            _settings = settings;
        }

        public IEnumerable<RunResult<MulticlassClassificationMetrics>> Execute(IDataView trainData, string labelColumn = DefaultColumnNames.Label,
            string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumn = labelColumn,
                SamplingKeyColumn = samplingKeyColumn
            };
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<MulticlassClassificationMetrics>> Execute(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<MulticlassClassificationMetrics>> Execute(IDataView trainData, IDataView validationData, string labelColumn = DefaultColumnNames.Label, IEstimator<ITransformer> preFeaturizers = null)
        {
            var columnInformation = new ColumnInformation() { LabelColumn = labelColumn };
            return Execute(_context, trainData, columnInformation, validationData, preFeaturizers);
        }

        public IEnumerable<RunResult<MulticlassClassificationMetrics>> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, validationData, preFeaturizers);
        }

        internal IEnumerable<RunResult<MulticlassClassificationMetrics>> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            throw new NotImplementedException();
        }

        internal IEnumerable<RunResult<MulticlassClassificationMetrics>> Execute(MLContext context,
            IDataView trainData,
            ColumnInformation columnInfo,
            IDataView validationData = null,
            IEstimator<ITransformer> preFeaturizers = null)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData, columnInfo, validationData);

            // run autofit & get all pipelines run in that process
            var experiment = new Experiment<MulticlassClassificationMetrics>(context, TaskKind.MulticlassClassification, trainData, 
                columnInfo, validationData, preFeaturizers, new OptimizingMetricInfo(_settings.OptimizingMetric),
                _settings.ProgressHandler, _settings, new MultiMetricsAgent(_settings.OptimizingMetric),
                TrainerExtensionUtil.GetTrainerNames(_settings.Trainers));
            
            return experiment.Execute();
        }
    }

    public static class MulticlassExperimentResultExtensions
    {
        public static RunResult<MulticlassClassificationMetrics> Best(this IEnumerable<RunResult<MulticlassClassificationMetrics>> results, MulticlassClassificationMetric metric = MulticlassClassificationMetric.MicroAccuracy)
        {
            var metricsAgent = new MultiMetricsAgent(metric);
            return RunResultUtil.GetBestRunResult(results, metricsAgent);
        }
    }
}
