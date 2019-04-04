// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Auto
{
    public abstract class ExperimentBase<TMetrics> where TMetrics : class
    {
        protected readonly MLContext Context;

        private readonly IMetricsAgent<TMetrics> _metricsAgent;
        private readonly OptimizingMetricInfo _optimizingMetricInfo;
        private readonly ExperimentSettings _settings;
        private readonly TaskKind _task;
        private readonly IEnumerable<TrainerName> _trainerWhitelist;

        internal ExperimentBase(MLContext context,
            IMetricsAgent<TMetrics> metricsAgent,
            OptimizingMetricInfo optimizingMetricInfo,
            ExperimentSettings settings,
            TaskKind task,
            IEnumerable<TrainerName> trainerWhitelist)
        {
            Context = context;
            _metricsAgent = metricsAgent;
            _optimizingMetricInfo = optimizingMetricInfo;
            _settings = settings;
            _task = task;
            _trainerWhitelist = trainerWhitelist;
        }

        public IEnumerable<RunDetails<TMetrics>> Execute(IDataView trainData, string labelColumn = DefaultColumnNames.Label,
            string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizers = null, IProgress<RunDetails<TMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumn,
                SamplingKeyColumnName = samplingKeyColumn
            };
            return Execute(trainData, columnInformation, preFeaturizers, progressHandler);
        }

        public IEnumerable<RunDetails<TMetrics>> Execute(IDataView trainData, ColumnInformation columnInformation, 
            IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetails<TMetrics>> progressHandler = null)
        {
            // Cross val threshold for # of dataset rows --
            // If dataset has < threshold # of rows, use cross val.
            // Else, run experiment using train-validate split.
            const int crossValRowCountThreshold = 15000;

            var rowCount = DatasetDimensionsUtil.CountRows(trainData, crossValRowCountThreshold);

            if (rowCount < crossValRowCountThreshold)
            {
                const int numCrossValFolds = 10;
                var splitResult = SplitUtil.CrossValSplit(Context, trainData, numCrossValFolds, columnInformation?.SamplingKeyColumnName);
                return ExecuteCrossValSummary(splitResult.trainDatasets, columnInformation, splitResult.validationDatasets, preFeaturizer, progressHandler);
            }
            else
            {
                var splitResult = SplitUtil.TrainValidateSplit(Context, trainData, columnInformation?.SamplingKeyColumnName);
                return ExecuteTrainValidate(splitResult.trainData, columnInformation, splitResult.validationData, preFeaturizer, progressHandler);
            }
        }

        public IEnumerable<RunDetails<TMetrics>> Execute(IDataView trainData, IDataView validationData, string labelColumn = DefaultColumnNames.Label, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetails<TMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation() { LabelColumnName = labelColumn };
            return Execute(trainData, validationData, columnInformation, preFeaturizer, progressHandler);
        }

        public IEnumerable<RunDetails<TMetrics>> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetails<TMetrics>> progressHandler = null)
        {
            if (validationData == null)
            {
                var splitResult = SplitUtil.TrainValidateSplit(Context, trainData, columnInformation?.SamplingKeyColumnName);
                trainData = splitResult.trainData;
                validationData = splitResult.validationData;
            }
            return ExecuteTrainValidate(trainData, columnInformation, validationData, preFeaturizer, progressHandler);
        }

        public IEnumerable<CrossValidationRunDetails<TMetrics>> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<CrossValidationRunDetails<TMetrics>> progressHandler = null)
        {
            UserInputValidationUtil.ValidateNumberOfCVFoldsArg(numberOfCVFolds);
            var splitResult = SplitUtil.CrossValSplit(Context, trainData, numberOfCVFolds, columnInformation?.SamplingKeyColumnName);
            return ExecuteCrossVal(splitResult.trainDatasets, columnInformation, splitResult.validationDatasets, preFeaturizer, progressHandler);
        }

        public IEnumerable<CrossValidationRunDetails<TMetrics>> Execute(IDataView trainData, 
            uint numberOfCVFolds, string labelColumn = DefaultColumnNames.Label,
            string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, 
            Progress<CrossValidationRunDetails<TMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumn,
                SamplingKeyColumnName = samplingKeyColumn
            };
            return Execute(trainData, numberOfCVFolds, columnInformation, preFeaturizer, progressHandler);
        }

        private IEnumerable<RunDetails<TMetrics>> ExecuteTrainValidate(IDataView trainData,
            ColumnInformation columnInfo,
            IDataView validationData,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<RunDetails<TMetrics>> progressHandler)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData, columnInfo, validationData);

            // Apply pre-featurizer
            ITransformer preprocessorTransform = null;
            if (preFeaturizer != null)
            {
                preprocessorTransform = preFeaturizer.Fit(trainData);
                trainData = preprocessorTransform.Transform(trainData);
                validationData = preprocessorTransform.Transform(validationData);
            }

            var runner = new TrainValidateRunner<TMetrics>(Context, trainData, validationData, columnInfo.LabelColumnName, _metricsAgent,
                preFeaturizer, preprocessorTransform, _settings.DebugLogger);
            var columns = DatasetColumnInfoUtil.GetDatasetColumnInfo(Context, trainData, columnInfo);
            return Execute(columnInfo, columns, preFeaturizer, progressHandler, runner);
        }

        private IEnumerable<CrossValidationRunDetails<TMetrics>> ExecuteCrossVal(IDataView[] trainDatasets,
            ColumnInformation columnInfo,
            IDataView[] validationDatasets,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<CrossValidationRunDetails<TMetrics>> progressHandler)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainDatasets[0], columnInfo, validationDatasets[0]);

            // Apply pre-featurizer
            ITransformer[] preprocessorTransforms = null;
            (trainDatasets, validationDatasets, preprocessorTransforms) = ApplyPreFeaturizerCrossVal(trainDatasets, validationDatasets, preFeaturizer);

            var runner = new CrossValRunner<TMetrics>(Context, trainDatasets, validationDatasets, _metricsAgent, preFeaturizer,
                preprocessorTransforms, columnInfo.LabelColumnName, _settings.DebugLogger);
            var columns = DatasetColumnInfoUtil.GetDatasetColumnInfo(Context, trainDatasets[0], columnInfo);
            return Execute(columnInfo, columns, preFeaturizer, progressHandler, runner);
        }

        private IEnumerable<RunDetails<TMetrics>> ExecuteCrossValSummary(IDataView[] trainDatasets,
            ColumnInformation columnInfo,
            IDataView[] validationDatasets,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<RunDetails<TMetrics>> progressHandler)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainDatasets[0], columnInfo, validationDatasets[0]);

            // Apply pre-featurizer
            ITransformer[] preprocessorTransforms = null;
            (trainDatasets, validationDatasets, preprocessorTransforms) = ApplyPreFeaturizerCrossVal(trainDatasets, validationDatasets, preFeaturizer);

            var runner = new CrossValSummaryRunner<TMetrics>(Context, trainDatasets, validationDatasets, _metricsAgent, preFeaturizer,
                preprocessorTransforms, columnInfo.LabelColumnName, _optimizingMetricInfo, _settings.DebugLogger);
            var columns = DatasetColumnInfoUtil.GetDatasetColumnInfo(Context, trainDatasets[0], columnInfo);
            return Execute(columnInfo, columns, preFeaturizer, progressHandler, runner);
        }

        private IEnumerable<TRunDetails> Execute<TRunDetails>(ColumnInformation columnInfo,
            DatasetColumnInfo[] columns,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<TRunDetails> progressHandler,
            IRunner<TRunDetails> runner)
            where TRunDetails : RunDetails
        {
            // Execute experiment & get all pipelines run
            var experiment = new Experiment<TRunDetails, TMetrics>(Context, _task, _optimizingMetricInfo, progressHandler,
                _settings, _metricsAgent, _trainerWhitelist, columns, runner);

            return experiment.Execute();
        }

        private static (IDataView[] trainDatasets, IDataView[] validDatasets, ITransformer[] preprocessorTransforms)
            ApplyPreFeaturizerCrossVal(IDataView[] trainDatasets, IDataView[] validDatasets, IEstimator<ITransformer> preFeaturizer)
        {
            if (preFeaturizer == null)
            {
                return (trainDatasets, validDatasets, null);
            }

            var preprocessorTransforms = new ITransformer[trainDatasets.Length];
            for (var i = 0; i < trainDatasets.Length; i++)
            {
                // Preprocess train and validation data
                preprocessorTransforms[i] = preFeaturizer.Fit(trainDatasets[i]);
                trainDatasets[i] = preprocessorTransforms[i].Transform(trainDatasets[i]);
                validDatasets[i] = preprocessorTransforms[i].Transform(validDatasets[i]);
            }

            return (trainDatasets, validDatasets, preprocessorTransforms);
        }
    }
}
