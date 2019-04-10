// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Formatting;
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
                    return LogLevel.Trace;
                default:
                    return LogLevel.Info;
            }
        }

        internal static void SaveModel(ITransformer model, FileInfo modelPath, MLContext mlContext,
            DataViewSchema modelInputSchema)
        {

            if (!Directory.Exists(modelPath.Directory.FullName))
            {
                Directory.CreateDirectory(modelPath.Directory.FullName);
            }

            using (var fs = File.Create(modelPath.FullName))
                mlContext.Model.Save(model, modelInputSchema, fs);
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

            result.LabelColumnName = Sanitize(columnInformation.LabelColumnName);

            if (!string.IsNullOrEmpty(columnInformation.ExampleWeightColumnName))
                result.ExampleWeightColumnName = Sanitize(columnInformation.ExampleWeightColumnName);

            if (!string.IsNullOrEmpty(columnInformation.SamplingKeyColumnName))
                result.SamplingKeyColumnName = Sanitize(columnInformation.SamplingKeyColumnName);

            foreach (var value in columnInformation.CategoricalColumnNames)
            {
                result.CategoricalColumnNames.Add(Sanitize(value));
            }

            foreach (var value in columnInformation.IgnoredColumnNames)
            {
                result.IgnoredColumnNames.Add(Sanitize(value));
            }

            foreach (var value in columnInformation.NumericColumnNames)
            {
                result.NumericColumnNames.Add(Sanitize(value));
            }

            foreach (var value in columnInformation.TextColumnNames)
            {
                result.TextColumnNames.Add(Sanitize(value));
            }


            return result;
        }

        internal static void WriteOutputToFiles(string fileContent, string fileName, string outPutBaseDir)
        {
            if (!Directory.Exists(outPutBaseDir))
            {
                Directory.CreateDirectory(outPutBaseDir);
            }
            File.WriteAllText($"{outPutBaseDir}/{fileName}", fileContent);
        }

        internal static string FormatCode(string trainProgramCSFileContent)
        {
            //Format
            var tree = CSharpSyntaxTree.ParseText(trainProgramCSFileContent);
            var syntaxNode = tree.GetRoot();
            trainProgramCSFileContent = Formatter.Format(syntaxNode, new AdhocWorkspace()).ToFullString();
            return trainProgramCSFileContent;
        }


        internal static void AddProjectsToSolution(string modelprojectDir,
            string modelProjectName,
            string predictProjectDir,
            string predictProjectName,
            string trainProjectDir,
            string trainProjectName,
            string solutionName)
        {
            var proc2 = new System.Diagnostics.Process();
            proc2.StartInfo.FileName = @"dotnet";

            proc2.StartInfo.Arguments = $"sln {solutionName} add {Path.Combine(trainProjectDir, trainProjectName)} {Path.Combine(predictProjectDir, predictProjectName)} {Path.Combine(modelprojectDir, modelProjectName)}";
            proc2.StartInfo.UseShellExecute = false;
            proc2.StartInfo.RedirectStandardOutput = true;
            proc2.Start();
            string outPut2 = proc2.StandardOutput.ReadToEnd();

            proc2.WaitForExit();
            var exitCode2 = proc2.ExitCode;
            proc2.Close();
        }

        internal static void CreateSolutionFile(string solutionFile, string outputPath)
        {
            var proc = new System.Diagnostics.Process();
            proc.StartInfo.FileName = @"dotnet";

            proc.StartInfo.Arguments = $"new sln --name {solutionFile} --output {outputPath} --force";
            proc.StartInfo.UseShellExecute = false;
            proc.StartInfo.RedirectStandardOutput = true;
            proc.Start();
            string outPut = proc.StandardOutput.ReadToEnd();

            proc.WaitForExit();
            var exitCode = proc.ExitCode;
            proc.Close();
        }
    }
}
