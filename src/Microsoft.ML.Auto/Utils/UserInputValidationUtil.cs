// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class UserInputValidationUtil
    {
        public static void ValidateExperimentExecuteArgs(IDataView trainData, ColumnInformation columnInformation,
            IDataView validationData)
        {
            ValidateTrainData(trainData);
            ValidateColumnInformation(trainData, columnInformation);
            ValidateValidationData(trainData, validationData);
        }

        public static void ValidateInferColumnsArgs(string path, ColumnInformation columnInformation)
        {
            ValidateColumnInformation(columnInformation);
            ValidatePath(path);
        }

        public static void ValidateInferColumnsArgs(string path, string labelColumn)
        {
            ValidateLabelColumn(labelColumn);
            ValidatePath(path);
        }

        public static void ValidateInferColumnsArgs(string path)
        {
            ValidatePath(path);
        }

        private static void ValidateTrainData(IDataView trainData)
        {
            if (trainData == null)
            {
                throw new ArgumentNullException(nameof(trainData), "Training data cannot be null");
            }

            var type = trainData.Schema.GetColumnOrNull(DefaultColumnNames.Features)?.Type.GetItemType();
            if (type != null && type != NumberDataViewType.Single)
            {
                throw new ArgumentException($"{DefaultColumnNames.Features} column must be of data type Single", nameof(trainData));
            }
        }

        private static void ValidateColumnInformation(IDataView trainData, ColumnInformation columnInformation)
        {
            ValidateColumnInformation(columnInformation);
            ValidateTrainDataColumnExists(trainData, columnInformation.LabelColumn);
            ValidateTrainDataColumnExists(trainData, columnInformation.WeightColumn);
            ValidateTrainDataColumnsExist(trainData, columnInformation.CategoricalColumns);
            ValidateTrainDataColumnsExist(trainData, columnInformation.NumericColumns);
            ValidateTrainDataColumnsExist(trainData, columnInformation.TextColumns);
            ValidateTrainDataColumnsExist(trainData, columnInformation.IgnoredColumns);
        }

        private static void ValidateColumnInformation(ColumnInformation columnInformation)
        {
            ValidateLabelColumn(columnInformation.LabelColumn);

            ValidateColumnInfoEnumerationProperty(columnInformation.CategoricalColumns, "categorical");
            ValidateColumnInfoEnumerationProperty(columnInformation.NumericColumns, "numeric");
            ValidateColumnInfoEnumerationProperty(columnInformation.TextColumns, "text");
            ValidateColumnInfoEnumerationProperty(columnInformation.IgnoredColumns, "ignored");

            // keep a list of all columns, to detect duplicates
            var allColumns = new List<string>();
            allColumns.Add(columnInformation.LabelColumn);
            if (columnInformation.WeightColumn != null) { allColumns.Add(columnInformation.WeightColumn); }
            if (columnInformation.CategoricalColumns != null) { allColumns.AddRange(columnInformation.CategoricalColumns); }
            if (columnInformation.NumericColumns != null) { allColumns.AddRange(columnInformation.NumericColumns); }
            if (columnInformation.TextColumns != null) { allColumns.AddRange(columnInformation.TextColumns); }
            if (columnInformation.IgnoredColumns != null) { allColumns.AddRange(columnInformation.IgnoredColumns); }

            var duplicateColName = FindFirstDuplicate(allColumns);
            if (duplicateColName != null)
            {
                throw new ArgumentException($"Duplicate column name {duplicateColName} is present in two or more distinct properties of provided column information", nameof(columnInformation));
            }
        }

        private static void ValidateColumnInfoEnumerationProperty(IEnumerable<string> columns, string propertyName)
        {
            if (columns?.Contains(null) == true)
            {
                throw new ArgumentException($"Null column string was specified as {propertyName} in column information");
            }
        }

        private static void ValidateLabelColumn(string labelColumn)
        {
            if (labelColumn == null)
            {
                throw new ArgumentException("Provided label column cannot be null");
            }
        }

        private static void ValidatePath(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException(nameof(path), "Provided path cannot be null");
            }

            var fileInfo = new FileInfo(path);

            if (!fileInfo.Exists)
            {
                throw new ArgumentException($"File '{path}' does not exist", nameof(path));
            }

            if (fileInfo.Length == 0)
            {
                throw new ArgumentException($"File at path '{path}' cannot be empty", nameof(path));
            }
        }

        private static void ValidateValidationData(IDataView trainData, IDataView validationData)
        {
            if (validationData == null)
            {
                return;
            }

            const string schemaMismatchError = "Training data and validation data schemas do not match.";

            if (trainData.Schema.Count != validationData.Schema.Count)
            {
                throw new ArgumentException($"{schemaMismatchError} Train data has '{trainData.Schema.Count}' columns," +
                    $"and validation data has '{validationData.Schema.Count}' columns.", nameof(validationData));
            }

            foreach (var trainCol in trainData.Schema)
            {
                var validCol = validationData.Schema.GetColumnOrNull(trainCol.Name);
                if (validCol == null)
                {
                    throw new ArgumentException($"{schemaMismatchError} Column '{trainCol.Name}' exsits in train data, but not in validation data.", nameof(validationData));
                }

                if (trainCol.Type != validCol.Value.Type)
                {
                    throw new ArgumentException($"{schemaMismatchError} Column '{trainCol.Name}' is of type {trainCol.Type} in train data, and type " +
                        $"{validCol.Value.Type} in validation data.", nameof(validationData));
                }
            }
        }

        private static void ValidateTrainDataColumnsExist(IDataView trainData, IEnumerable<string> columnNames)
        {
            if (columnNames == null)
            {
                return;
            }

            foreach (var columnName in columnNames)
            {
                ValidateTrainDataColumnExists(trainData, columnName);
            }
        }

        private static void ValidateTrainDataColumnExists(IDataView trainData, string columnName)
        {
            if (columnName != null && trainData.Schema.GetColumnOrNull(columnName) == null)
            {
                throw new ArgumentException($"Provided column '{columnName}' not found in training data.");
            }
        }

        private static string FindFirstDuplicate(IEnumerable<string> values)
        {
            var groups = values.GroupBy(v => v);
            return groups.FirstOrDefault(g => g.Count() > 1)?.Key;
        }
    }
}
