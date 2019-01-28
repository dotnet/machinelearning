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
        public static void ValidateAutoFitArgs(IDataView trainData, string label, IDataView validationData,
            AutoFitSettings settings, IEnumerable<(string, ColumnPurpose)> purposeOverrides)
        {
            ValidateTrainData(trainData);
            ValidateValidationData(trainData, validationData);
            ValidateLabel(trainData, validationData, label);
            ValidateSettings(settings);
            ValidatePurposeOverrides(trainData, validationData, label, purposeOverrides);
        }

        public static void ValidateInferColumnsArgs(string path, string label)
        {
            ValidateLabel(label);
            ValidatePath(path);
        }

        public static void ValidateAutoReadArgs(string path, string label)
        {
            ValidateLabel(label);
            ValidatePath(path);
        }

        public static void ValidateCreateTextReaderArgs(ColumnInferenceResult columnInferenceResult)
        {
            if(columnInferenceResult == null)
            {
                throw new ArgumentNullException($"Column inference result cannot be null", nameof(columnInferenceResult));
            }

            if (string.IsNullOrEmpty(columnInferenceResult.Separator))
            {
                throw new ArgumentException($"Column inference result cannot have null or empty separator", nameof(columnInferenceResult));
            }

            if (columnInferenceResult.Columns == null || !columnInferenceResult.Columns.Any())
            {
                throw new ArgumentException($"Column inference result must contain at least one column", nameof(columnInferenceResult));
            }
            
            if(columnInferenceResult.Columns.Any(c => c.Item1 == null))
            {
                throw new ArgumentException($"Column inference result cannot contain null columns", nameof(columnInferenceResult));
            }

            if (columnInferenceResult.Columns.Any(c => c.Item1.Name == null || c.Item1.Type == null || c.Item1.Source == null))
            {
                throw new ArgumentException($"Column inference result cannot contain a column that has a null name, type, or source", nameof(columnInferenceResult));
            }
        }

        private static void ValidateTrainData(IDataView trainData)
        {
            if(trainData == null)
            {
                throw new ArgumentNullException("Training data cannot be null", nameof(trainData));
            }
        }

        private static void ValidateLabel(IDataView trainData, IDataView validationData, string label)
        {
            ValidateLabel(label);

            if(trainData.Schema.GetColumnOrNull(label) == null)
            {
                throw new ArgumentException($"Provided label column '{label}' not found in training data.", nameof(label));
            }
        }

        private static void ValidateLabel(string label)
        {
            if (label == null)
            {
                throw new ArgumentNullException("Provided label cannot be null", nameof(label));
            }
        }

        private static void ValidatePath(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("Provided path cannot be null", nameof(path));
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
            if(validationData == null)
            {
                throw new ArgumentNullException("Validation data cannot be null", nameof(validationData));
            }

            const string schemaMismatchError = "Training data and validation data schemas do not match.";

            if (trainData.Schema.Count != validationData.Schema.Count)
            {
                throw new ArgumentException($"{schemaMismatchError} Train data has '{trainData.Schema.Count}' columns," +
                    $"and validation data has '{validationData.Schema.Count}' columns.", nameof(validationData));
            }

            foreach(var trainCol in trainData.Schema)
            {
                var validCol = validationData.Schema.GetColumnOrNull(trainCol.Name);
                if(validCol == null)
                {
                    throw new ArgumentException($"{schemaMismatchError} Column '{trainCol.Name}' exsits in train data, but not in validation data.", nameof(validationData));
                }

                if(trainCol.Type != validCol.Value.Type)
                {
                    throw new ArgumentException($"{schemaMismatchError} Column '{trainCol.Name}' is of type {trainCol.Type} in train data, and type " +
                        $"{validCol.Value.Type} in validation data.", nameof(validationData));
                }
            }
        }

        private static void ValidateSettings(AutoFitSettings settings)
        {
            if(settings?.StoppingCriteria == null)
            {
                return;
            }

            if(settings.StoppingCriteria.MaxIterations <= 0)
            {
                throw new ArgumentOutOfRangeException("Max iterations must be > 0", nameof(settings));
            }
        }

        private static void ValidatePurposeOverrides(IDataView trainData, IDataView validationData,
            string label, IEnumerable<(string, ColumnPurpose)> purposeOverrides)
        {
            if (purposeOverrides == null)
            {
                return;
            }

            foreach (var purposeOverride in purposeOverrides)
            {
                var colName = purposeOverride.Item1;
                var colPurpose = purposeOverride.Item2;

                if (colName == null)
                {
                    throw new ArgumentException("Purpose override column name cannot be null.", nameof(purposeOverrides));
                }

                if (trainData.Schema.GetColumnOrNull(colName) == null)
                {
                    throw new ArgumentException($"Purpose override column name '{colName}' not found in training data.", nameof(purposeOverride));
                }

                // if column w/ purpose = 'Label' found, ensure it matches the passed-in label
                if(colPurpose == ColumnPurpose.Label && colName != label)
                {
                    throw new ArgumentException($"Label column name in provided list of purposes '{colName}' must match " +
                        $"the label column name '{label}'", nameof(purposeOverrides));
                }
            }

            // ensure all column names unique
            var duplicateColName = FindFirstDuplicate(purposeOverrides.Select(p => p.Item1));
            if (duplicateColName != null)
            {
                throw new ArgumentException($"Duplicate column name '{duplicateColName}' in purpose overrides.", nameof(purposeOverrides));
            }
        }

        private static string FindFirstDuplicate(IEnumerable<string> values)
        {
            var groups = values.GroupBy(v => v);
            return groups.FirstOrDefault(g => g.Count() > 1)?.Key;
        }
    }
}