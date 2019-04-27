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
        private const string CrossValEmptyFoldErrorMsg = @"Cross validation split has 0 rows. Perhaps " +
            "try increasing number of rows provided in training data, or lowering specified number of " +
            "cross validation folds.";

        public static (IDataView[] trainDatasets, IDataView[] validationDatasets) CrossValSplit(MLContext context, 
            IDataView trainData, uint numFolds, string samplingKeyColumn)
        {
            var originalColumnNames = trainData.Schema.Select(c => c.Name);
            var splits = context.Data.CrossValidationSplit(trainData, (int)numFolds, samplingKeyColumnName: samplingKeyColumn);
            var trainDatasets = new IDataView[numFolds];
            var validationDatasets = new IDataView[numFolds];
            for (var i = 0; i < numFolds; i++)
            {
                var split = splits[i];
                trainDatasets[i] = DropAllColumnsExcept(context, split.TrainSet, originalColumnNames);
                validationDatasets[i] = DropAllColumnsExcept(context, split.TestSet, originalColumnNames);
                if (DatasetDimensionsUtil.IsDataViewEmpty(trainDatasets[i]) || DatasetDimensionsUtil.IsDataViewEmpty(validationDatasets[i]))
                {
                    throw new InvalidOperationException(CrossValEmptyFoldErrorMsg);
                }
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
