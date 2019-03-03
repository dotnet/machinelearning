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
    public sealed class RegressionExperimentSettings : ExperimentSettings
    {
        public RegressionMetric OptimizingMetric { get; set; } = RegressionMetric.RSquared;
        public ICollection<RegressionTrainer> Trainers { get; } =
                     Enum.GetValues(typeof(RegressionTrainer)).OfType<RegressionTrainer>().ToList();
        public IProgress<RunResult<RegressionMetrics>> ProgressHandler { get; set; }
    }

    public enum RegressionMetric
    {
        MeanAbsoluteError,
        MeanSquaredError,
        RootMeanSquaredError,
        RSquared
    }

    public enum RegressionTrainer
    {
        FastForest,
        FastTree,
        FastTreeTweedie,
        LightGbm,
        OnlineGradientDescent,
        OrdinaryLeastSquares,
        PoissonRegression,
        StochasticDualCoordinateAscent,
    }

    public sealed class RegressionExperiment
    {
        private readonly MLContext _context;
        private readonly RegressionExperimentSettings _settings;

        internal RegressionExperiment(MLContext context, RegressionExperimentSettings settings)
        {
            _context = context;
            _settings = settings;
        }

        public IEnumerable<RunResult<RegressionMetrics>> Execute(IDataView trainData, string labelColumn = DefaultColumnNames.Label, IEstimator<ITransformer> preFeaturizers = null)
        {
            var columnInformation = new ColumnInformation() { LabelColumn = labelColumn };
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<RegressionMetrics>> Execute(IDataView trainData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<RegressionMetrics>> Execute(IDataView trainData, IDataView validationData, string labelColumn = DefaultColumnNames.Label, IEstimator<ITransformer> preFeaturizers = null)
        {
            var columnInformation = new ColumnInformation() { LabelColumn = labelColumn };
            return Execute(_context, trainData, columnInformation, validationData, preFeaturizers);
        }

        public IEnumerable<RunResult<RegressionMetrics>> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, validationData, preFeaturizers);
        }

        internal IEnumerable<RunResult<RegressionMetrics>> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            throw new NotImplementedException();
        }

        internal IEnumerable<RunResult<RegressionMetrics>> Execute(MLContext context,
            IDataView trainData,
            ColumnInformation columnInfo,
            IDataView validationData = null,
            IEstimator<ITransformer> preFeaturizers = null)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData, columnInfo, validationData);

            // run autofit & get all pipelines run in that process
            var experiment = new Experiment<RegressionMetrics>(context, TaskKind.Regression, trainData, columnInfo, 
                validationData, preFeaturizers, new OptimizingMetricInfo(_settings.OptimizingMetric), 
                _settings.ProgressHandler, _settings, new RegressionMetricsAgent(_settings.OptimizingMetric),
                TrainerExtensionUtil.GetTrainerNames(_settings.Trainers));

            return experiment.Execute();
        }
    }

    public static class RegressionExperimentResultExtensions
    {
        public static RunResult<RegressionMetrics> Best(this IEnumerable<RunResult<RegressionMetrics>> results, RegressionMetric metric = RegressionMetric.RSquared)
        {
            var metricsAgent = new RegressionMetricsAgent(metric);
            return RunResultUtil.GetBestRunResult(results, metricsAgent);
        }
    }
}
