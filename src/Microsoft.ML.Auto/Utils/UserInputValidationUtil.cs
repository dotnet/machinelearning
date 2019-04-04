// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class UserInputValidationUtil
    {
        // column purpose names
        private const string LabelColumnPurposeName = "label";
        private const string WeightColumnPurposeName = "weight";
        private const string NumericColumnPurposeName = "numeric";
        private const string CategoricalColumnPurposeName = "categorical";
        private const string TextColumnPurposeName = "text";
        private const string IgnoredColumnPurposeName = "ignored";
        private const string SamplingKeyColumnPurposeName = "sampling key";

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

        public static void ValidateNumberOfCVFoldsArg(uint numberOfCVFolds)
        {
            if (numberOfCVFolds <= 1)
            {
                throw new ArgumentException($"{nameof(numberOfCVFolds)} must be at least 2", nameof(numberOfCVFolds));
            }
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
            ValidateTrainDataColumn(trainData, columnInformation.LabelColumnName, LabelColumnPurposeName);
            ValidateTrainDataColumn(trainData, columnInformation.ExampleWeightColumnName, WeightColumnPurposeName);
            ValidateTrainDataColumn(trainData, columnInformation.SamplingKeyColumnName, SamplingKeyColumnPurposeName);
            ValidateTrainDataColumns(trainData, columnInformation.CategoricalColumnNames, CategoricalColumnPurposeName,
                new DataViewType[] { NumberDataViewType.Single, TextDataViewType.Instance });
            ValidateTrainDataColumns(trainData, columnInformation.NumericColumnNames, NumericColumnPurposeName,
                new DataViewType[] { NumberDataViewType.Single, BooleanDataViewType.Instance });
            ValidateTrainDataColumns(trainData, columnInformation.TextColumnNames, TextColumnPurposeName,
                new DataViewType[] { TextDataViewType.Instance });
            ValidateTrainDataColumns(trainData, columnInformation.IgnoredColumnNames, IgnoredColumnPurposeName);
        }

        private static void ValidateColumnInformation(ColumnInformation columnInformation)
        {
            ValidateLabelColumn(columnInformation.LabelColumnName);

            ValidateColumnInfoEnumerationProperty(columnInformation.CategoricalColumnNames, CategoricalColumnPurposeName);
            ValidateColumnInfoEnumerationProperty(columnInformation.NumericColumnNames, NumericColumnPurposeName);
            ValidateColumnInfoEnumerationProperty(columnInformation.TextColumnNames, TextColumnPurposeName);
            ValidateColumnInfoEnumerationProperty(columnInformation.IgnoredColumnNames, IgnoredColumnPurposeName);

            // keep a list of all columns, to detect duplicates
            var allColumns = new List<string>();
            allColumns.Add(columnInformation.LabelColumnName);
            if (columnInformation.ExampleWeightColumnName != null) { allColumns.Add(columnInformation.ExampleWeightColumnName); }
            if (columnInformation.CategoricalColumnNames != null) { allColumns.AddRange(columnInformation.CategoricalColumnNames); }
            if (columnInformation.NumericColumnNames != null) { allColumns.AddRange(columnInformation.NumericColumnNames); }
            if (columnInformation.TextColumnNames != null) { allColumns.AddRange(columnInformation.TextColumnNames); }
            if (columnInformation.IgnoredColumnNames != null) { allColumns.AddRange(columnInformation.IgnoredColumnNames); }

            var duplicateColName = FindFirstDuplicate(allColumns);
            if (duplicateColName != null)
            {
                throw new ArgumentException($"Duplicate column name {duplicateColName} is present in two or more distinct properties of provided column information", nameof(columnInformation));
            }
        }

        private static void ValidateColumnInfoEnumerationProperty(IEnumerable<string> columns, string columnPurpose)
        {
            if (columns?.Contains(null) == true)
            {
                throw new ArgumentException($"Null column string was specified as {columnPurpose} in column information");
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

        private static void ValidateTrainDataColumns(IDataView trainData, IEnumerable<string> columnNames, string columnPurpose,
            IEnumerable<DataViewType> allowedTypes = null)
        {
            if (columnNames == null)
            {
                return;
            }

            foreach (var columnName in columnNames)
            {
                ValidateTrainDataColumn(trainData, columnName, columnPurpose, allowedTypes);
            }
        }

        private static void ValidateTrainDataColumn(IDataView trainData, string columnName, string columnPurpose, IEnumerable<DataViewType> allowedTypes = null)
        {
            if (columnName == null)
            {
                return;
            }

            var nullableColumn = trainData.Schema.GetColumnOrNull(columnName);
            if (nullableColumn == null)
            {
                throw new ArgumentException($"Provided {columnPurpose} column '{columnName}' not found in training data.");
            }

            if(allowedTypes == null)
            {
                return;
            }
            var column = nullableColumn.Value;
            var itemType = column.Type.GetItemType();
            if (!allowedTypes.Contains(itemType))
            {
                if (allowedTypes.Count() == 1)
                {
                    throw new ArgumentException($"Provided {columnPurpose} column '{columnName}' was of type {itemType}, " +
                        $"but only type {allowedTypes.First()} is allowed.");
                }
                else
                {
                    throw new ArgumentException($"Provided {columnPurpose} column '{columnName}' was of type {itemType}, " +
                        $"but only types {string.Join(", ", allowedTypes)} are allowed.");
                }
            }
        }

        private static string FindFirstDuplicate(IEnumerable<string> values)
        {
            var groups = values.GroupBy(v => v);
            return groups.FirstOrDefault(g => g.Count() > 1)?.Key;
        }
    }
}
