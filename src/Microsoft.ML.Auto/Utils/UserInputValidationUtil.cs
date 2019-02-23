// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

// todo: re-write & test user input validation once final API nailed down.
// Tracked by Github issue: https://github.com/dotnet/machinelearning-automl/issues/159

namespace Microsoft.ML.Auto
{
    /*internal static class UserInputValidationUtil
    {
        public static void ValidateAutoFitArgs(IDataView trainData, string label, IDataView validationData,
            AutoFitSettings settings, IEnumerable<(string, ColumnPurpose)> purposeOverrides)
        {
            ValidateTrainData(trainData);
            ValidateValidationData(trainData, validationData);
            ValidateLabel(trainData, label);
            ValidateSettings(settings);
            ValidatePurposeOverrides(trainData, validationData, label, purposeOverrides);
        }

        public static void ValidateInferColumnsArgs(string path, string label)
        {
            ValidateLabel(label);
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

        private static void ValidateLabel(IDataView trainData, string label)
        {
            ValidateLabel(label);

            if (trainData.Schema.GetColumnOrNull(label) == null)
            {
                throw new ArgumentException($"Provided label column '{label}' not found in training data.", nameof(label));
            }
        }

        private static void ValidateLabel(string label)
        {
            if (label == null)
            {
                throw new ArgumentNullException(nameof(label), "Provided label cannot be null");
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

        private static void ValidateSettings(AutoFitSettings settings)
        {
            if (settings?.StoppingCriteria == null)
            {
                return;
            }

            if (settings.StoppingCriteria.MaxIterations <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(settings), "Max iterations must be > 0");
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
                if (colPurpose == ColumnPurpose.Label && colName != label)
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
    }*/
}
