// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
#nullable enable

using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Interface for dataset manager. This interface doesn't include any method or property definition and is used by <see cref="AutoMLExperiment"/> and other components to retrieve the instance of the actual
    /// dataset manager from containers.
    /// </summary>
    public interface IDatasetManager
    {
    }

    /// <summary>
    /// Inferface for cross validate dataset manager.
    /// </summary>
    public interface ICrossValidateDatasetManager : IDatasetManager
    {
        /// <summary>
        /// Cross validate fold.
        /// </summary>
        int Fold { get; set; }

        /// <summary>
        /// The dataset to cross validate.
        /// </summary>
        IDataView Dataset { get; set; }

        /// <summary>
        /// The dataset column used for grouping rows.
        /// </summary>
        string? SamplingKeyColumnName { get; set; }
    }

    public interface ITrainValidateDatasetManager : IDatasetManager
    {
        IDataView LoadTrainDataset(MLContext context, TrialSettings? settings);

        IDataView LoadValidateDataset(MLContext context, TrialSettings? settings);
    }

    internal class TrainValidateDatasetManager : IDatasetManager, ITrainValidateDatasetManager
    {
        private ulong _rowCount;
        private readonly IDataView _trainDataset;
        private readonly IDataView _validateDataset;
        private readonly string _subSamplingKey = "TrainValidateDatasetSubsamplingKey";
        private bool _isInitialized = false;
        public TrainValidateDatasetManager(IDataView trainDataset, IDataView validateDataset, string? subSamplingKey = null)
        {
            _trainDataset = trainDataset;
            _validateDataset = validateDataset;
            _subSamplingKey = subSamplingKey ?? _subSamplingKey;
        }

        public string SubSamplingKey => _subSamplingKey;

        /// <summary>
        /// Load Train Dataset. If <see cref="TrialSettings.Parameter"/> contains <see cref="_subSamplingKey"/> then the train dataset will be subsampled.
        /// </summary>
        /// <param name="context">MLContext.</param>
        /// <param name="settings">trial settings. If null, return entire train dataset.</param>
        /// <returns>train dataset.</returns>
        public IDataView LoadTrainDataset(MLContext context, TrialSettings? settings)
        {
            if (!_isInitialized)
            {
                InitializeTrainDataset(context);
                _isInitialized = true;
            }
            var trainTestSplitParameter = settings?.Parameter.ContainsKey(nameof(TrainValidateDatasetManager)) is true ? settings.Parameter[nameof(TrainValidateDatasetManager)] : null;
            if (trainTestSplitParameter is Parameter parameter)
            {
                var subSampleRatio = parameter.ContainsKey(_subSamplingKey) ? parameter[_subSamplingKey].AsType<double>() : 1;
                if (subSampleRatio < 1.0)
                {
                    var count = (long)(subSampleRatio * _rowCount);
                    if (count <= 10)
                    {
                        // fix issue https://github.com/dotnet/machinelearning-modelbuilder/issues/2734
                        // take at least 10 rows to avoid empty dataset
                        count = 10;
                    }

                    var subSampledTrainDataset = context.Data.TakeRows(_trainDataset, count);
                    return subSampledTrainDataset;
                }
            }

            return _trainDataset;
        }

        public IDataView LoadValidateDataset(MLContext context, TrialSettings? settings)
        {
            return _validateDataset;
        }

        private void InitializeTrainDataset(MLContext context)
        {
            _rowCount = DatasetDimensionsUtil.CountRows(_trainDataset, ulong.MaxValue);
        }
    }

    internal class CrossValidateDatasetManager : IDatasetManager, ICrossValidateDatasetManager
    {
        public CrossValidateDatasetManager(IDataView dataset, int fold, string? samplingKeyColumnName = null)
        {
            Dataset = dataset;
            Fold = fold;
            SamplingKeyColumnName = samplingKeyColumnName;
        }

        public IDataView Dataset { get; set; }

        public int Fold { get; set; }

        public string? SamplingKeyColumnName { get; set; }
    }
}
