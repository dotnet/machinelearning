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
            ValidateLabel(trainData, validationData, label);
            ValidateValidationData(trainData, validationData);
            ValidateSettings(settings);
            ValidatePurposeOverrides(trainData, validationData, label, purposeOverrides);
        }

        public static void ValidateInferTransformArgs(IDataView data, string label)
        {
            ValidateTrainData(data);
            ValidateLabel(data, null, label);
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

        public static void ValidateAutoReadArgs(IMultiStreamSource source, string label)
        {
            ValidateLabel(label);

            if(source == null)
            {
                throw new ArgumentNullException(nameof(source), $"Source parameter cannot be null");
            }

            if(source.Count < 0)
            {
                throw new ArgumentException(nameof(source), $"Multistream source cannot be empty");
            }
        }

        public static void ValidateCreateTextReaderArgs(ColumnInferenceResult columnInferenceResult)
        {
            if(columnInferenceResult == null)
            {
                throw new ArgumentNullException(nameof(columnInferenceResult), $"Column inference result cannot be null");
            }

            if(columnInferenceResult.Columns == null || !columnInferenceResult.Columns.Any())
            {
                throw new ArgumentException(nameof(columnInferenceResult), $"Column inference result must contain at least one column");
            }
            
            if(columnInferenceResult.Columns.Any(c => c.Item1 == null))
            {
                throw new ArgumentException(nameof(columnInferenceResult), $"Column inference result cannot contain null columns");
            }
        }

        private static void ValidateTrainData(IDataView trainData)
        {
            if(trainData == null)
            {
                throw new ArgumentNullException(nameof(trainData), "Training data cannot be null");
            }
        }

        private static void ValidateLabel(IDataView trainData, IDataView validationData, string label)
        {
            ValidateLabel(label);

            if(trainData.Schema.GetColumnOrNull(label) == null)
            {
                throw new ArgumentException(nameof(label), $"Provided label column '{label}' not found in training data.");
            }

            if (validationData != null && validationData.Schema.GetColumnOrNull(label) == null)
            {
                throw new ArgumentException(nameof(label), $"Provided label column '{label}' not found in validation data.");
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
                throw new ArgumentException(nameof(path), $"File '{path}' does not exist");
            }

            if (fileInfo.Length == 0)
            {
                throw new ArgumentException(nameof(path), $"File at path '{path}' cannot be empty");
            }
        }

        private static void ValidateValidationData(IDataView trainData, IDataView validationData)
        {
            if(validationData == null)
            {
                return;
            }

            const string schemaMismatchError = "Training data and validation data schemas do not match.";

            if (trainData.Schema.Count != validationData.Schema.Count)
            {
                throw new ArgumentException(nameof(validationData), $"{schemaMismatchError} Train data has '{trainData.Schema.Count}' columns," +
                    $"and validation data has '{validationData.Schema.Count}' columns.");
            }

            foreach(var trainCol in trainData.Schema)
            {
                var validCol = validationData.Schema.GetColumnOrNull(trainCol.Name);
                if(validCol == null)
                {
                    throw new ArgumentException(nameof(validationData), $"{schemaMismatchError} Column '{trainCol.Name}' exsits in train data, but not in validation data.");
                }

                if(trainCol.Type != validCol.Value.Type)
                {
                    throw new ArgumentException(nameof(validationData), $"{schemaMismatchError} Column '{trainCol.Name}' is of type {trainCol.Type} in train data, and type " +
                        $"{validCol.Value.Type} in validation data.");
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
                    throw new ArgumentException(nameof(purposeOverrides), "Purpose override column name cannot be null.");
                }

                if (trainData.Schema.GetColumnOrNull(colName) == null)
                {
                    throw new ArgumentException(nameof(purposeOverride), $"Purpose override column name '{colName}' not found in training data.");
                }

                if(validationData != null && validationData.Schema.GetColumnOrNull(colName) == null)
                {
                    throw new ArgumentException(nameof(purposeOverride), $"Purpose override column name '{colName}' not found in validation data.");
                }

                // if column w/ purpose = 'Label' found, ensure it matches the passed-in label
                if(colPurpose == ColumnPurpose.Label && colName != label)
                {
                    throw new ArgumentException(nameof(purposeOverrides), $"Label column name in provided list of purposes '{colName}' must match " +
                        $"the label column name '{label}'");
                }
            }

            // ensure all column names unique
            var groups = purposeOverrides.GroupBy(p => p.Item1);
            var duplicateColName = groups.FirstOrDefault(g => g.Count() > 1)?.First().Item1;
            if (duplicateColName != null)
            {
                throw new ArgumentException(nameof(purposeOverrides), $"Duplicate column name '{duplicateColName}' in purpose overrides.");
            }
        }
    }
}