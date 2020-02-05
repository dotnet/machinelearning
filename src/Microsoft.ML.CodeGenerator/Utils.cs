// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Formatting;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.Data;

namespace Microsoft.ML.CodeGenerator.Utilities
{
    internal class Utils
    {
        internal static string Sanitize(string name)
        {
            return string.Join("", name.Select(x => Char.IsLetterOrDigit(x) ? x : '_'));
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
                    return sanitizedInput.First().ToString().ToUpper() + sanitizedInput.Substring(1);
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

        internal static int AddProjectsToSolution(string solutionPath, string[] projects)
        {
            var proc = new System.Diagnostics.Process();
            var projectPaths = projects.Select((name) => $"\"{Path.Combine(Path.GetDirectoryName(solutionPath), name).ToString()}\"");
            try
            {
                proc.StartInfo.FileName = @"dotnet";
                proc.StartInfo.Arguments = $"sln \"{solutionPath}\" add {string.Join(" ", projectPaths)}";
                proc.StartInfo.UseShellExecute = false;
                proc.StartInfo.RedirectStandardOutput = true;
                proc.Start();
                string outPut = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();
                var exitCode = proc.ExitCode;
                return exitCode;
            }
            finally
            {
                proc.Close();
            }
        }

        internal static int AddProjectsToSolution(string modelprojectDir,
            string modelProjectName,
            string consoleAppProjectDir,
            string consoleAppProjectName,
            string solutionPath)
        {
            var proc = new System.Diagnostics.Process();
            try
            {
                proc.StartInfo.FileName = @"dotnet";
                proc.StartInfo.Arguments = $"sln \"{solutionPath}\" add  \"{Path.Combine(consoleAppProjectDir, consoleAppProjectName)}\" \"{Path.Combine(modelprojectDir, modelProjectName)}\"";
                proc.StartInfo.UseShellExecute = false;
                proc.StartInfo.RedirectStandardOutput = true;
                proc.Start();
                string outPut = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();
                var exitCode = proc.ExitCode;
                return exitCode;
            }
            finally
            {
                proc.Close();
            }
        }

        internal static int CreateSolutionFile(string solutionFile, string outputPath)
        {
            var proc = new System.Diagnostics.Process();
            try
            {
                proc.StartInfo.FileName = @"dotnet";
                proc.StartInfo.Arguments = $"new sln --name \"{solutionFile}\" --output \"{outputPath}\" --force";
                proc.StartInfo.UseShellExecute = false;
                proc.StartInfo.RedirectStandardOutput = true;
                proc.Start();
                string outPut = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();
                var exitCode = proc.ExitCode;
                return exitCode;
            }
            finally
            {
                proc.Close();
            }
        }

        internal static IList<string> GenerateClassLabels(ColumnInferenceResults columnInferenceResults, IDictionary<string, CodeGeneratorSettings.ColumnMapping> columnMapping = default)
        {
            IList<string> result = new List<string>();
            foreach (var column in columnInferenceResults.TextLoaderOptions.Columns)
            {
                StringBuilder sb = new StringBuilder();
                int range = (column.Source[0].Max - column.Source[0].Min).Value;
                bool isArray = range > 0;
                sb.Append(Symbols.PublicSymbol);
                sb.Append(Symbols.Space);

                // if column is in columnMapping, use the type and name in that
                DataKind dataKind;
                string columnName;

                if (columnMapping != null && columnMapping.ContainsKey(column.Name))
                {
                    dataKind = columnMapping[column.Name].ColumnType;
                    columnName = columnMapping[column.Name].ColumnName;
                }
                else
                {
                    dataKind = column.DataKind;
                    columnName = column.Name;
                }
                switch (dataKind)
                {
                    case Microsoft.ML.Data.DataKind.String:
                        sb.Append(Symbols.StringSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Boolean:
                        sb.Append(Symbols.BoolSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Single:
                        sb.Append(Symbols.FloatSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Double:
                        sb.Append(Symbols.DoubleSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Int32:
                        sb.Append(Symbols.IntSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.UInt32:
                        sb.Append(Symbols.UIntSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Int64:
                        sb.Append(Symbols.LongSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.UInt64:
                        sb.Append(Symbols.UlongSymbol);
                        break;
                    default:
                        throw new ArgumentException($"The data type '{column.DataKind}' is not handled currently.");

                }

                if (range > 0)
                {
                    result.Add($"[ColumnName(\"{columnName}\"),LoadColumn({column.Source[0].Min}, {column.Source[0].Max}) VectorType({(range + 1)})]");
                    sb.Append("[]");
                }
                else
                {
                    result.Add($"[ColumnName(\"{columnName}\"), LoadColumn({column.Source[0].Min})]");
                }
                sb.Append(" ");
                sb.Append(Utils.Normalize(column.Name));
                sb.Append("{get; set;}");
                result.Add(sb.ToString());
                result.Add("\r\n");
            }
            return result;
        }
    }
}
