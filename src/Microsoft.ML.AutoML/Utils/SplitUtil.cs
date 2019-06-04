// Licensed to the .NET Foundation under one or more agreements.	
// The .NET Foundation licenses this file to you under the MIT license.	
// See the LICENSE file in the project root for more information.	

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.AutoML
{
    internal static class SplitUtil
    {
        public static (IDataView[] trainDatasets, IDataView[] validationDatasets) CrossValSplit(MLContext context,
            IDataView trainData, uint numFolds, string samplingKeyColumn, TaskKind taskKind, string labelColumnName)
        {
            (IEnumerable<IDataView> trainDatasets, IEnumerable<IDataView> validationDatasets) = CrossValSplit(context, 
                trainData, numFolds, samplingKeyColumn);

            if (taskKind != TaskKind.BinaryClassification)
            {
                return (trainDatasets.ToArray(), validationDatasets.ToArray());
            }

            // If we're running binary classification, discard splits where there is not at least one 
            // true & one false label in the test set. Otherwise, scoring the dataset crashes because
            // AUC cannot be computed.
            var filteredTrainDatasets = new List<IDataView>();
            var filteredValidationDatasets = new List<IDataView>();
            for (var i = 0; i < trainDatasets.Count(); i++)
            {
                var validationDataset = validationDatasets.ElementAt(i);
                var labelColumn = validationDataset.Schema.First(c => !c.IsHidden && c.Name == labelColumnName);
                if (DatasetDimensionsUtil.ComputeCardinality<bool>(validationDataset, labelColumn, 2) < 2)
                {
                    continue;
                }

                var trainDataset = trainDatasets.ElementAt(i);
                filteredTrainDatasets.Add(trainDataset);
                filteredValidationDatasets.Add(validationDataset);
            }

            if (!filteredTrainDatasets.Any())
            {
                throw new InvalidOperationException("There are too few rows of data, or there are too few rows of data that have " +
                    "one of the two possible label values. Try increasing the total number of rows provided in the training data, or increasing " + 
                    "the number of rows with the label that occurs least frequently. You can also try specifying a lower number of " +
                    "cross validation folds.");
            }

            return (filteredTrainDatasets.ToArray(), filteredValidationDatasets.ToArray());
        }

        private static (IEnumerable<IDataView> trainDatasets, IEnumerable<IDataView> validationDatasets) CrossValSplit(MLContext context,
            IDataView trainData, uint numFolds, string samplingKeyColumn)
        {
            var originalColumnNames = trainData.Schema.Select(c => c.Name);
            var splits = context.Data.CrossValidationSplit(trainData, (int)numFolds, samplingKeyColumnName: samplingKeyColumn);
            var trainDatasets = new List<IDataView>();
            var validationDatasets = new List<IDataView>();
            
            foreach (var split in splits)
            {
                // Discard splits where either train or test set is empty
                if (DatasetDimensionsUtil.IsDataViewEmpty(split.TrainSet) ||
                    DatasetDimensionsUtil.IsDataViewEmpty(split.TestSet))
                {
                    continue;
                }

                // Remove added columns, so they are not featurized by AutoML
                var trainDataset = DropAllColumnsExcept(context, split.TrainSet, originalColumnNames);
                var validationDataset = DropAllColumnsExcept(context, split.TestSet, originalColumnNames);

                trainDatasets.Add(trainDataset);
                validationDatasets.Add(validationDataset);
            }

            if (!trainDatasets.Any())
            {
                throw new InvalidOperationException("All cross validation folds have empty train or test data. " +
                    "Try increasing the number of rows provided in training data, or specifying a lower number of " +
                    "cross validation folds.");
            }

            return (trainDatasets, validationDatasets);
        }

        /// <summary>
        /// Split the data into a single train/test split.
        /// </summary>
        public static (IDataView trainData, IDataView validationData) TrainValidateSplit(MLContext context, IDataView trainData, 
            string samplingKeyColumn)
        {
            var originalColumnNames = trainData.Schema.Select(c => c.Name);
            var splitData = context.Data.TrainTestSplit(trainData, samplingKeyColumnName: samplingKeyColumn);
            trainData = DropAllColumnsExcept(context, splitData.TrainSet, originalColumnNames);
            var validationData = DropAllColumnsExcept(context, splitData.TestSet, originalColumnNames);
            return (trainData, validationData);
        }

        private static IDataView DropAllColumnsExcept(MLContext context, IDataView data, IEnumerable<string> columnsToKeep)
        {
            var allColumns = data.Schema.Select(c => c.Name);
            var columnsToDrop = allColumns.Except(columnsToKeep);
            if (!columnsToDrop.Any())
            {
                return data;
            }
            return context.Transforms.DropColumns(columnsToDrop.ToArray()).Fit(data).Transform(data);
        }
    }
}
