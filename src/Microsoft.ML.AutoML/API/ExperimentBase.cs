// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// AutoML experiment base class. All task-specific AutoML experiments
    /// (like <see cref="BinaryClassificationExperiment"/>) inherit from this class.
    /// </summary>
    /// <typeparam name="TMetrics">Metrics type used by task-specific AutoML experiments.</typeparam>
    /// <typeparam name="TExperimentSettings">Experiment settings type.</typeparam>
    public abstract class ExperimentBase<TMetrics, TExperimentSettings>
        where TMetrics : class
        where TExperimentSettings : ExperimentSettings
    {
        private protected readonly MLContext Context;
        private protected readonly IMetricsAgent<TMetrics> MetricsAgent;
        private protected readonly OptimizingMetricInfo OptimizingMetricInfo;
        private protected readonly TExperimentSettings Settings;

        private readonly IChannel _logger;
        private readonly TaskKind _task;
        private readonly IEnumerable<TrainerName> _trainerAllowList;

        internal ExperimentBase(MLContext context,
            IMetricsAgent<TMetrics> metricsAgent,
            OptimizingMetricInfo optimizingMetricInfo,
            TExperimentSettings settings,
            TaskKind task,
            IEnumerable<TrainerName> trainerAllowList)
        {
            Context = context;
            MetricsAgent = metricsAgent;
            OptimizingMetricInfo = optimizingMetricInfo;
            Settings = settings;
            _logger = ((IChannelProvider)context).Start("AutoML");
            _task = task;
            _trainerAllowList = trainerAllowList;
        }

        /// <summary>
        /// Executes an AutoML experiment.
        /// </summary>
        /// <param name="trainData">The training data used by the AutoML experiment.</param>
        /// <param name="labelColumnName">The dataset column used as the label.</param>
        /// <param name="samplingKeyColumn">The dataset column used as the sampling key column.
        /// See <see cref="ColumnInformation.SamplingKeyColumnName"/> for more information.</param>
        /// <param name="preFeaturizer">Pre-featurizer that AutoML will apply to the data during an
        /// experiment. (The pre-featurizer will be fit only on the training data split to produce a
        /// trained transform. Then, the trained transform will be applied to both the training
        /// data split and corresponding validation data split.)</param>
        /// <param name="progressHandler">A user-defined object that implements
        /// the <see cref="IProgress{T}"/> interface. AutoML will invoke the method
        /// <see cref="IProgress{T}.Report(T)"/> after each model it produces during the
        /// course of the experiment.
        /// </param>
        /// <returns>The experiment result.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public virtual ExperimentResult<TMetrics> Execute(IDataView trainData, string labelColumnName = DefaultColumnNames.Label,
            string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<TMetrics>> progressHandler = null)
        {
            ColumnInformation columnInformation;
            if (_task == TaskKind.Ranking)
            {
                columnInformation = new ColumnInformation()
                {
                    LabelColumnName = labelColumnName,
                    SamplingKeyColumnName = samplingKeyColumn ?? DefaultColumnNames.GroupId,
                    GroupIdColumnName = samplingKeyColumn ?? DefaultColumnNames.GroupId // For ranking, we want to enforce having the same column as samplingKeyColum and GroupIdColumn
                };
            }
            else
            {
                columnInformation = new ColumnInformation()
                {
                    LabelColumnName = labelColumnName,
                    SamplingKeyColumnName = samplingKeyColumn
                };
            }
            return Execute(trainData, columnInformation, preFeaturizer, progressHandler);
        }

        /// <summary>
        /// Executes an AutoML experiment.
        /// </summary>
        /// <param name="trainData">The training data to be used by the AutoML experiment.</param>
        /// <param name="columnInformation">Column information for the dataset.</param>
        /// <param name="preFeaturizer">Pre-featurizer that AutoML will apply to the data during an
        /// experiment. (The pre-featurizer will be fit only on the training data split to produce a
        /// trained transform. Then, the trained transform will be applied to both the training
        /// data split and corresponding validation data split.)</param>
        /// <param name="progressHandler">A user-defined object that implements
        /// the <see cref="IProgress{T}"/> interface. AutoML will invoke the method
        /// <see cref="IProgress{T}.Report(T)"/> after each model it produces during the
        /// course of the experiment.
        /// </param>
        /// <returns>The experiment result.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public virtual ExperimentResult<TMetrics> Execute(IDataView trainData, ColumnInformation columnInformation,
            IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<TMetrics>> progressHandler = null)
        {
            // Cross val threshold for # of dataset rows --
            // If dataset has < threshold # of rows, use cross val.
            // Else, run experiment using train-validate split.
            const int crossValRowCountThreshold = 15000;

            var rowCount = DatasetDimensionsUtil.CountRows(trainData, crossValRowCountThreshold);
            var samplingKeyColumnName = GetSamplingKey(columnInformation?.GroupIdColumnName, columnInformation?.SamplingKeyColumnName);
            if (rowCount < crossValRowCountThreshold)
            {
                const int numCrossValFolds = 10;
                var splitResult = SplitUtil.CrossValSplit(Context, trainData, numCrossValFolds, samplingKeyColumnName);
                return ExecuteCrossValSummary(splitResult.trainDatasets, columnInformation, splitResult.validationDatasets, preFeaturizer, progressHandler);
            }
            else
            {
                var splitResult = SplitUtil.TrainValidateSplit(Context, trainData, samplingKeyColumnName);
                return ExecuteTrainValidate(splitResult.trainData, columnInformation, splitResult.validationData, preFeaturizer, progressHandler);
            }
        }

        private string GetSamplingKey(string groupIdColumnName, string samplingKeyColumnName)
        {
            UserInputValidationUtil.ValidateSamplingKey(samplingKeyColumnName, groupIdColumnName, _task);
            if (_task == TaskKind.Ranking)
                return groupIdColumnName ?? DefaultColumnNames.GroupId;
            return samplingKeyColumnName;
        }

        /// <summary>
        /// Executes an AutoML experiment.
        /// </summary>
        /// <param name="trainData">The training data to be used by the AutoML experiment.</param>
        /// <param name="validationData">The validation data to be used by the AutoML experiment.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="preFeaturizer">Pre-featurizer that AutoML will apply to the data during an
        /// experiment. (The pre-featurizer will be fit only on the training data split to produce a
        /// trained transform. Then, the trained transform will be applied to both the training
        /// data split and corresponding validation data split.)</param>
        /// <param name="progressHandler">A user-defined object that implements
        /// the <see cref="IProgress{T}"/> interface. AutoML will invoke the method
        /// <see cref="IProgress{T}.Report(T)"/> after each model it produces during the
        /// course of the experiment.
        /// </param>
        /// <returns>The experiment result.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public virtual ExperimentResult<TMetrics> Execute(IDataView trainData, IDataView validationData, string labelColumnName = DefaultColumnNames.Label, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<TMetrics>> progressHandler = null)
        {
            var columnInformation = (_task == TaskKind.Ranking) ?
                new ColumnInformation() { LabelColumnName = labelColumnName, GroupIdColumnName = DefaultColumnNames.GroupId } :
                new ColumnInformation() { LabelColumnName = labelColumnName };

            return Execute(trainData, validationData, columnInformation, preFeaturizer, progressHandler);
        }

        /// <summary>
        /// Executes an AutoML experiment.
        /// </summary>
        /// <param name="trainData">The training data to be used by the AutoML experiment.</param>
        /// <param name="validationData">The validation data to be used by the AutoML experiment.</param>
        /// <param name="columnInformation">Column information for the dataset.</param>
        /// <param name="preFeaturizer">Pre-featurizer that AutoML will apply to the data during an
        /// experiment. (The pre-featurizer will be fit only on the training data split to produce a
        /// trained transform. Then, the trained transform will be applied to both the training
        /// data split and corresponding validation data split.)</param>
        /// <param name="progressHandler">A user-defined object that implements
        /// the <see cref="IProgress{T}"/> interface. AutoML will invoke the method
        /// <see cref="IProgress{T}.Report(T)"/> after each model it produces during the
        /// course of the experiment.
        /// </param>
        /// <returns>The experiment result.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public virtual ExperimentResult<TMetrics> Execute(IDataView trainData, IDataView validationData,
            ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null,
            IProgress<RunDetail<TMetrics>> progressHandler = null)
        {
            if (validationData == null)
            {
                return Execute(trainData, columnInformation, preFeaturizer, progressHandler);
            }
            return ExecuteTrainValidate(trainData, columnInformation, validationData, preFeaturizer, progressHandler);
        }

        /// <summary>
        /// Executes an AutoML experiment.
        /// </summary>
        /// <param name="trainData">The training data to be used by the AutoML experiment.</param>
        /// <param name="numberOfCVFolds">The number of cross validation folds into which the training data should be divided when fitting a model.</param>
        /// <param name="columnInformation">Column information for the dataset.</param>
        /// <param name="preFeaturizer">Pre-featurizer that AutoML will apply to the data during an
        /// experiment. (The pre-featurizer will be fit only on the training data split to produce a
        /// trained transform. Then, the trained transform will be applied to both the training
        /// data split and corresponding validation data split.)</param>
        /// <param name="progressHandler">A user-defined object that implements
        /// the <see cref="IProgress{T}"/> interface. AutoML will invoke the method
        /// <see cref="IProgress{T}.Report(T)"/> after each model it produces during the
        /// course of the experiment.
        /// </param>
        /// <returns>The cross validation experiment result.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public virtual CrossValidationExperimentResult<TMetrics> Execute(IDataView trainData, uint numberOfCVFolds,
            ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizer = null,
            IProgress<CrossValidationRunDetail<TMetrics>> progressHandler = null)
        {
            UserInputValidationUtil.ValidateNumberOfCVFoldsArg(numberOfCVFolds);
            var samplingKeyColumnName = GetSamplingKey(columnInformation?.GroupIdColumnName, columnInformation?.SamplingKeyColumnName);
            var splitResult = SplitUtil.CrossValSplit(Context, trainData, numberOfCVFolds, samplingKeyColumnName);
            return ExecuteCrossVal(splitResult.trainDatasets, columnInformation, splitResult.validationDatasets, preFeaturizer, progressHandler);
        }

        /// <summary>
        /// Executes an AutoML experiment.
        /// </summary>
        /// <param name="trainData">The training data to be used by the AutoML experiment.</param>
        /// <param name="numberOfCVFolds">The number of cross validation folds into which the training data should be divided when fitting a model.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="samplingKeyColumn">The name of the sampling key column.</param>
        /// <param name="preFeaturizer">Pre-featurizer that AutoML will apply to the data during an
        /// experiment. (The pre-featurizer will be fit only on the training data split to produce a
        /// trained transform. Then, the trained transform will be applied to both the training
        /// data split and corresponding validation data split.)</param>
        /// <param name="progressHandler">A user-defined object that implements
        /// the <see cref="IProgress{T}"/> interface. AutoML will invoke the method
        /// <see cref="IProgress{T}.Report(T)"/> after each model it produces during the
        /// course of the experiment.
        /// </param>
        /// <returns>The cross validation experiment result.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public virtual CrossValidationExperimentResult<TMetrics> Execute(IDataView trainData,
            uint numberOfCVFolds, string labelColumnName = DefaultColumnNames.Label,
            string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null,
            IProgress<CrossValidationRunDetail<TMetrics>> progressHandler = null)
        {
            var columnInformation = (_task == TaskKind.Ranking) ?
            new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn ?? DefaultColumnNames.GroupId,
                GroupIdColumnName = samplingKeyColumn ?? DefaultColumnNames.GroupId // For ranking, we want to enforce having the same column as samplingKeyColum and GroupIdColumn
            }
            :
            new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn
            };
            return Execute(trainData, numberOfCVFolds, columnInformation, preFeaturizer, progressHandler);
        }

        private protected abstract CrossValidationRunDetail<TMetrics> GetBestCrossValRun(IEnumerable<CrossValidationRunDetail<TMetrics>> results);

        private protected abstract RunDetail<TMetrics> GetBestRun(IEnumerable<RunDetail<TMetrics>> results);

        private ExperimentResult<TMetrics> ExecuteTrainValidate(IDataView trainData,
            ColumnInformation columnInfo,
            IDataView validationData,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<RunDetail<TMetrics>> progressHandler)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData, columnInfo, validationData, _task);

            // Apply pre-featurizer
            ITransformer preprocessorTransform = null;
            if (preFeaturizer != null)
            {
                preprocessorTransform = preFeaturizer.Fit(trainData);
                trainData = preprocessorTransform.Transform(trainData);
                validationData = preprocessorTransform.Transform(validationData);
            }

            var runner = new TrainValidateRunner<TMetrics>(Context, trainData, validationData, columnInfo.GroupIdColumnName, columnInfo.LabelColumnName, MetricsAgent,
                preFeaturizer, preprocessorTransform, _logger);
            var columns = DatasetColumnInfoUtil.GetDatasetColumnInfo(Context, trainData, columnInfo);
            return Execute(columnInfo, columns, preFeaturizer, progressHandler, runner);
        }

        private CrossValidationExperimentResult<TMetrics> ExecuteCrossVal(IDataView[] trainDatasets,
            ColumnInformation columnInfo,
            IDataView[] validationDatasets,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<CrossValidationRunDetail<TMetrics>> progressHandler)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainDatasets[0], columnInfo, validationDatasets[0], _task);

            // Apply pre-featurizer
            ITransformer[] preprocessorTransforms = null;
            (trainDatasets, validationDatasets, preprocessorTransforms) = ApplyPreFeaturizerCrossVal(trainDatasets, validationDatasets, preFeaturizer);

            var runner = new CrossValRunner<TMetrics>(Context, trainDatasets, validationDatasets, MetricsAgent, preFeaturizer,
                preprocessorTransforms, columnInfo.GroupIdColumnName, columnInfo.LabelColumnName, _logger);
            var columns = DatasetColumnInfoUtil.GetDatasetColumnInfo(Context, trainDatasets[0], columnInfo);

            // Execute experiment & get all pipelines run
            var experiment = new Experiment<CrossValidationRunDetail<TMetrics>, TMetrics>(Context, _task, OptimizingMetricInfo, progressHandler,
                Settings, MetricsAgent, _trainerAllowList, columns, runner, _logger);
            var runDetails = experiment.Execute();

            var bestRun = GetBestCrossValRun(runDetails);
            var experimentResult = new CrossValidationExperimentResult<TMetrics>(runDetails, bestRun);
            return experimentResult;
        }

        private ExperimentResult<TMetrics> ExecuteCrossValSummary(IDataView[] trainDatasets,
            ColumnInformation columnInfo,
            IDataView[] validationDatasets,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<RunDetail<TMetrics>> progressHandler)
        {
            columnInfo = columnInfo ?? new ColumnInformation();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainDatasets[0], columnInfo, validationDatasets[0], _task);

            // Apply pre-featurizer
            ITransformer[] preprocessorTransforms = null;
            (trainDatasets, validationDatasets, preprocessorTransforms) = ApplyPreFeaturizerCrossVal(trainDatasets, validationDatasets, preFeaturizer);

            var runner = new CrossValSummaryRunner<TMetrics>(Context, trainDatasets, validationDatasets, MetricsAgent, preFeaturizer,
                preprocessorTransforms, columnInfo.GroupIdColumnName, columnInfo.LabelColumnName, OptimizingMetricInfo, _logger);
            var columns = DatasetColumnInfoUtil.GetDatasetColumnInfo(Context, trainDatasets[0], columnInfo);
            return Execute(columnInfo, columns, preFeaturizer, progressHandler, runner);
        }

        private ExperimentResult<TMetrics> Execute(ColumnInformation columnInfo,
            DatasetColumnInfo[] columns,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<RunDetail<TMetrics>> progressHandler,
            IRunner<RunDetail<TMetrics>> runner)
        {
            // Execute experiment & get all pipelines run
            var experiment = new Experiment<RunDetail<TMetrics>, TMetrics>(Context, _task, OptimizingMetricInfo, progressHandler,
                Settings, MetricsAgent, _trainerAllowList, columns, runner, _logger);
            var runDetails = experiment.Execute();

            var bestRun = GetBestRun(runDetails);
            var experimentResult = new ExperimentResult<TMetrics>(runDetails, bestRun);
            return experimentResult;
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
