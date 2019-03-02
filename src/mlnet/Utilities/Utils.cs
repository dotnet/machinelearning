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
                    return sanitizedInput;
            }
        }

    }
}
