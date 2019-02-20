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
    public class RegressionExperimentSettings : ExperimentSettings
    {
        public IProgress<RunResult<RegressionMetrics>> ProgressCallback;
        public RegressionMetric OptimizingMetric;
        public RegressionTrainer[] WhitelistedTrainers;
    }

    public enum RegressionMetric
    {
        L1,
        L2,
        Rms,
        RSquared
    }

    public enum RegressionTrainer
    {
        LightGbm
    }

    public class RegressionExperiment
    {
        private readonly MLContext _context;
        private readonly RegressionExperimentSettings _settings;

        internal RegressionExperiment(MLContext context, RegressionExperimentSettings settings)
        {
            _context = context;
            _settings = settings;
        }

        public IEnumerable<RunResult<RegressionMetrics>> Execute(IDataView trainData, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<RegressionMetrics>> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, validationData, preFeaturizers);
        }

        internal RunResult<RegressionMetrics> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
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
            //UserInputValidationUtil.ValidateAutoFitArgs(trainData, labelColunName, validationData, settings, columnPurposes);

            // run autofit & get all pipelines run in that process
            var autoFitter = new AutoFitter<RegressionMetrics>(context, TaskKind.Regression, trainData, columnInfo, 
                validationData, preFeaturizers, OptimizingMetric.RSquared, _settings?.ProgressCallback,
                _settings);

            return autoFitter.Fit();
        }
    }

    public static class RegressionExperimentResultExtensions
    {
        public static RunResult<RegressionMetrics> Best(this IEnumerable<RunResult<RegressionMetrics>> results)
        {
            double maxScore = results.Select(r => r.Metrics.RSquared).Max();
            return results.First(r => r.Metrics.RSquared == maxScore);
        }
    }
}
