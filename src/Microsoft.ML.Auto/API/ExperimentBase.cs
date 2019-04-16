// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Auto
{
    /// <summary>
    /// AutoML experiment base class. All task-specific AutoML experiments
    /// (like <see cref="BinaryClassificationExperiment"/>) inherit from this class.
    /// </summary>
    /// <typeparam name="TMetrics">Metrics type used by task-specific AutoML experiments.</typeparam>
    public abstract class ExperimentBase<TMetrics> where TMetrics : class
    {
        private protected readonly MLContext Context;

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
        /// <returns>An enumeration of all the runs in an experiment. See <see cref="RunDetail{TMetrics}"/>
        /// for more information on the contents of a run.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public IEnumerable<RunDetail<TMetrics>> Execute(IDataView trainData, string labelColumnName = DefaultColumnNames.Label,
            string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<TMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn
            };
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
        /// <returns>An enumeration of all the runs in an experiment. See <see cref="RunDetail{TMetrics}"/>
        /// for more information on the contents of a run.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public IEnumerable<RunDetail<TMetrics>> Execute(IDataView trainData, ColumnInformation columnInformation, 
            IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<TMetrics>> progressHandler = null)
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
        /// <returns>An enumeration of all the runs in an experiment. See <see cref="RunDetail{TMetrics}"/>
        /// for more information on the contents of a run.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public IEnumerable<RunDetail<TMetrics>> Execute(IDataView trainData, IDataView validationData, string labelColumnName = DefaultColumnNames.Label, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<TMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation() { LabelColumnName = labelColumnName };
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
        /// <returns>An enumeration of all the runs in an experiment. See <see cref="RunDetail{TMetrics}"/>
        /// for more information on the contents of a run.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public IEnumerable<RunDetail<TMetrics>> Execute(IDataView trainData, IDataView validationData, ColumnInformation columnInformation, IEstimator<ITransformer> preFeaturizer = null, IProgress<RunDetail<TMetrics>> progressHandler = null)
        {
            if (validationData == null)
            {
                var splitResult = SplitUtil.TrainValidateSplit(Context, trainData, columnInformation?.SamplingKeyColumnName);
                trainData = splitResult.trainData;
                validationData = splitResult.validationData;
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
        /// <returns>An enumeration of all the runs in an experiment. See <see cref="RunDetail{TMetrics}"/>
        /// for more information on the contents of a run.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public IEnumerable<CrossValidationRunDetail<TMetrics>> Execute(IDataView trainData, uint numberOfCVFolds, ColumnInformation columnInformation = null, IEstimator<ITransformer> preFeaturizer = null, IProgress<CrossValidationRunDetail<TMetrics>> progressHandler = null)
        {
            UserInputValidationUtil.ValidateNumberOfCVFoldsArg(numberOfCVFolds);
            var splitResult = SplitUtil.CrossValSplit(Context, trainData, numberOfCVFolds, columnInformation?.SamplingKeyColumnName);
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
        /// <returns>An enumeration of all the runs in an experiment. See <see cref="RunDetail{TMetrics}"/>
        /// for more information on the contents of a run.</returns>
        /// <remarks>
        /// Depending on the size of your data, the AutoML experiment could take a long time to execute.
        /// </remarks>
        public IEnumerable<CrossValidationRunDetail<TMetrics>> Execute(IDataView trainData, 
            uint numberOfCVFolds, string labelColumnName = DefaultColumnNames.Label,
            string samplingKeyColumn = null, IEstimator<ITransformer> preFeaturizer = null, 
            Progress<CrossValidationRunDetail<TMetrics>> progressHandler = null)
        {
            var columnInformation = new ColumnInformation()
            {
                LabelColumnName = labelColumnName,
                SamplingKeyColumnName = samplingKeyColumn
            };
            return Execute(trainData, numberOfCVFolds, columnInformation, preFeaturizer, progressHandler);
        }

        private IEnumerable<RunDetail<TMetrics>> ExecuteTrainValidate(IDataView trainData,
            ColumnInformation columnInfo,
            IDataView validationData,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<RunDetail<TMetrics>> progressHandler)
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

        private IEnumerable<CrossValidationRunDetail<TMetrics>> ExecuteCrossVal(IDataView[] trainDatasets,
            ColumnInformation columnInfo,
            IDataView[] validationDatasets,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<CrossValidationRunDetail<TMetrics>> progressHandler)
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

        private IEnumerable<RunDetail<TMetrics>> ExecuteCrossValSummary(IDataView[] trainDatasets,
            ColumnInformation columnInfo,
            IDataView[] validationDatasets,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<RunDetail<TMetrics>> progressHandler)
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

        private IEnumerable<TRunDetail> Execute<TRunDetail>(ColumnInformation columnInfo,
            DatasetColumnInfo[] columns,
            IEstimator<ITransformer> preFeaturizer,
            IProgress<TRunDetail> progressHandler,
            IRunner<TRunDetail> runner)
            where TRunDetail : RunDetail
        {
            // Execute experiment & get all pipelines run
            var experiment = new Experiment<TRunDetail, TMetrics>(Context, _task, _optimizingMetricInfo, progressHandler,
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
