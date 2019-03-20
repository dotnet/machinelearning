// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;
using NLog;

namespace Microsoft.ML.CLI.Utilities
{
    internal class Utils
    {
        internal static LogLevel GetVerbosity(string verbosity)
        {
            switch (verbosity)
            {
                case "q":
                    return LogLevel.Warn;
                case "m":
                    return LogLevel.Info;
                case "diag":
                    return LogLevel.Debug;
                default:
                    return LogLevel.Info;
            }
        }


        internal static void SaveModel(ITransformer model, FileInfo modelPath, MLContext mlContext)
        {

            if (!Directory.Exists(modelPath.Directory.FullName))
            {
                Directory.CreateDirectory(modelPath.Directory.FullName);
            }

            using (var fs = System.IO.File.Create(modelPath.FullName))
                model.SaveTo(mlContext, fs);
        }

        internal static string Sanitize(string name)
        {
            return string.Join("", name.Select(x => Char.IsLetterOrDigit(x) ? x : '_'));
        }

        internal static TaskKind GetTaskKind(string mlTask)
        {
            switch (mlTask)
            {
                case "binary-classification":
                    return TaskKind.BinaryClassification;
                case "multiclass-classification":
                    return TaskKind.MulticlassClassification;
                case "regression":
                    return TaskKind.Regression;
                default: // this should never be hit because the validation is done on command-line-api.
                    throw new NotImplementedException($"{Strings.UnsupportedMlTask} : {mlTask}");
            }
        }

        internal static string Normalize(string input)
        {
            //check if first character is int
            if (!string.IsNullOrEmpty(input) && int.TryParse(input.Substring(0, 1), out int val))
            {
                input = "Col" + input;
                return input;
            }
            switch (input)
            {
                case null: throw new ArgumentNullException(nameof(input));
                case "": throw new ArgumentException($"{nameof(input)} cannot be empty", nameof(input));
                default:
                    var sanitizedInput = Sanitize(input);
                    return sanitizedInput.First().ToString().ToUpper() + input.Substring(1);
            }
        }

        internal static Type GetCSharpType(DataKind labelType)
        {
            switch (labelType)
            {
                case Microsoft.ML.Data.DataKind.String:
                    return typeof(string);
                case Microsoft.ML.Data.DataKind.Boolean:
                    return typeof(bool);
                case Microsoft.ML.Data.DataKind.Single:
                    return typeof(float);
                case Microsoft.ML.Data.DataKind.Double:
                    return typeof(double);
                case Microsoft.ML.Data.DataKind.Int32:
                    return typeof(int);
                case Microsoft.ML.Data.DataKind.UInt32:
                    return typeof(uint);
                case Microsoft.ML.Data.DataKind.Int64:
                    return typeof(long);
                case Microsoft.ML.Data.DataKind.UInt64:
                    return typeof(ulong);
                default:
                    throw new ArgumentException($"The data type '{labelType}' is not handled currently.");
            }
        }

        internal static bool? GetCacheSettings(string input)
        {
            switch (input)
            {
                case "on": return true;
                case "off": return false;
                case "auto": return null;
                default:
                    throw new ArgumentException($"{nameof(input)} is invalid", nameof(input));
            }
        }

        internal static ColumnInformation GetSanitizedColumnInformation(ColumnInformation columnInformation)
        {
            var result = new ColumnInformation();

            result.LabelColumn = Sanitize(columnInformation.LabelColumn);

            if (!string.IsNullOrEmpty(columnInformation.WeightColumn))
                result.WeightColumn = Sanitize(columnInformation.WeightColumn);

            if (!string.IsNullOrEmpty(columnInformation.SamplingKeyColumn))
                result.SamplingKeyColumn = Sanitize(columnInformation.SamplingKeyColumn);

            foreach (var value in columnInformation.CategoricalColumns)
            {
                result.CategoricalColumns.Add(Sanitize(value));
            }

            foreach (var value in columnInformation.IgnoredColumns)
            {
                result.IgnoredColumns.Add(Sanitize(value));
            }

            foreach (var value in columnInformation.NumericColumns)
            {
                result.NumericColumns.Add(Sanitize(value));
            }

            foreach (var value in columnInformation.TextColumns)
            {
                result.TextColumns.Add(Sanitize(value));
            }


            return result;
        }

    }
}
