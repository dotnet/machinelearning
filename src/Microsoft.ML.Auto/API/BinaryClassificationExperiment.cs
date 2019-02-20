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
    public class BinaryExperimentSettings : ExperimentSettings
    {
        public IProgress<RunResult<BinaryClassificationMetrics>> ProgressCallback;
        public BinaryClassificationMetric OptimizingMetric;
        public BinaryClassificationTrainer[] WhitelistedTrainers;
    }

    public enum BinaryClassificationMetric
    {
        Accuracy
    }

    public enum BinaryClassificationTrainer
    {
        LightGbm
    }

    public class BinaryClassificationExperiment
    {
        private readonly MLContext _context;
        private readonly BinaryExperimentSettings _settings;

        internal BinaryClassificationExperiment(MLContext context, BinaryExperimentSettings settings)
        {
            _context = context;
            _settings = settings;
        }

        public IEnumerable<RunResult<BinaryClassificationMetrics>> Execute(IDataView trainData, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, null, preFeaturizers);
        }

        public IEnumerable<RunResult<BinaryClassificationMetrics>> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
        {
            return Execute(_context, trainData, columnInformation, validationData, preFeaturizers);
        }

        internal RunResult<BinaryClassificationMetrics> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizers = null)
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
            //UserInputValidationUtil.ValidateAutoFitArgs(trainData, labelColunName, validationData, settings, columnPurposes)

            // run autofit & get all pipelines run in that process
            var autoFitter = new AutoFitter<BinaryClassificationMetrics>(context, TaskKind.BinaryClassification, trainData, columnInfo, 
                validationData, preFeaturizers, OptimizingMetric.Accuracy, _settings?.ProgressCallback,
                _settings);

            return autoFitter.Fit();
        }
    }

    public static class BinaryExperimentResultExtensions
    {
        public static RunResult<BinaryClassificationMetrics> Best(this IEnumerable<RunResult<BinaryClassificationMetrics>> results)
        {
            double maxScore = results.Select(r => r.Metrics.Accuracy).Max();
            return results.First(r => r.Metrics.Accuracy == maxScore);
        }
    }
}
