// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
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

        /// <summary>
        /// Take the first line of data from inputFile and parse it as a dictionary using schema from columnInference.
        /// </summary>
        /// <param name="inputFile">path to input file.</param>
        /// <param name="columnInference">Column Inferernce Result.</param>
        /// <returns>A dictionary which key is sanitized column name and value is first line of data.</returns>
        internal static IDictionary<string, string> GenerateSampleData(string inputFile, ColumnInferenceResults columnInference)
        {
            try
            {
                var mlContext = new MLContext();
                var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
                var trainData = textLoader.Load(inputFile);
                return Utils.GenerateSampleData(trainData, columnInference);
            }
            catch (Exception)
            {
                return null;
            }
        }

        internal static IDictionary<string, string> GenerateSampleData(IDataView dataView, ColumnInferenceResults columnInference)
        {
            var featureColumns = dataView.Schema.AsEnumerable().Where(col => col.Name != columnInference.ColumnInformation.LabelColumnName && !columnInference.ColumnInformation.IgnoredColumnNames.Contains(col.Name));
            var rowCursor = dataView.GetRowCursor(featureColumns);

            var sampleData = featureColumns.Select(column => new { key = Utils.Normalize(column.Name), val = "null" }).ToDictionary(x => x.key, x => x.val);
            if (rowCursor.MoveNext())
            {
                var getGetGetterMethod = typeof(Utils).GetMethod(nameof(Utils.GetValueFromColumn), BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);

                foreach (var column in featureColumns)
                {
                    var getGeneraicGetGetterMethod = getGetGetterMethod.MakeGenericMethod(column.Type.RawType);
                    string val = getGeneraicGetGetterMethod.Invoke(null, new object[] { rowCursor, column }) as string;
                    sampleData[Utils.Normalize(column.Name)] = val;
                }
            }

            return sampleData;
        }

        internal static string GetValueFromColumn<T>(DataViewRowCursor rowCursor, DataViewSchema.Column column)
        {
            T val = default;
            var getter = rowCursor.GetGetter<T>(column);
            getter(ref val);

            // wrap string in quotes
            if (typeof(T) == typeof(ReadOnlyMemory<Char>))
            {
                return $"@\"{val.ToString().Replace("\"", "\\\"")}\"";
            }

            if (val is null)
            {
                return "\"null\"";
            }

            if (val is float)
            {
                var f = val as float?;
                if (Single.IsNaN(f.GetValueOrDefault()))
                {
                    return "Single.NaN";
                }

                if (Single.IsPositiveInfinity(f.GetValueOrDefault()))
                {
                    return "Single.PositiveInfinity";
                }

                if (Single.IsNegativeInfinity(f.GetValueOrDefault()))
                {
                    return "Single.NegativeInfinity";
                }

                return f?.ToString(CultureInfo.InvariantCulture) + "F";
            }

            if (val is bool)
            {
                var f = val as bool?;
                return f.GetValueOrDefault() ? "true" : "false";
            }

            return val.ToString();
        }

        internal static string Normalize(string input)
        {
            //check if first character is int
            if (!string.IsNullOrEmpty(input) && int.TryParse(input.Substring(0, 1), out int val))
            {
                input = "_" + input;
                return Normalize(input);
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
            List<string> normalizedColumnNames = new List<string>();
            bool duplicateColumnNamesExist = false;
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
                sb.Append(GetSymbolOfDataKind(dataKind));

                // Accomodate VectorType (array) columns
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
                string normalizedColumnName = Utils.Normalize(column.Name);
                // Put placeholder for normalized and unique version of column name
                if (!duplicateColumnNamesExist && normalizedColumnNames.Contains(normalizedColumnName))
                    duplicateColumnNamesExist = true;
                normalizedColumnNames.Add(normalizedColumnName);
                result.Add(sb.ToString());
                result.Add("\r\n");
            }
            for (int i = 1; i < result.Count; i+=3)
            {
                // Get normalized column name for correctly typed class property name
                // If duplicate column names exist, the only way to ensure all generated column names are unique is to add
                // a differentiator depending on the column load order from dataset.
                if (duplicateColumnNamesExist)
                    result[i] += normalizedColumnNames[i/3] + $"_col_{i/3}";
                else
                    result[i] += normalizedColumnNames[i/3];
                result[i] += "{get; set;}";
            }
            return result;
        }

        internal static string GetSymbolOfDataKind(DataKind dataKind)
        {
            switch (dataKind)
            {
                case DataKind.String:
                    return Symbols.StringSymbol;
                case DataKind.Boolean:
                    return Symbols.BoolSymbol;
                case DataKind.Single:
                    return Symbols.FloatSymbol;
                case DataKind.Double:
                    return Symbols.DoubleSymbol;
                case DataKind.Int32:
                    return Symbols.IntSymbol;
                case DataKind.UInt32:
                    return Symbols.UIntSymbol;
                case DataKind.Int64:
                    return Symbols.LongSymbol;
                case DataKind.UInt64:
                    return Symbols.UlongSymbol;
                default:
                    throw new ArgumentException($"The data type '{dataKind}' is not handled currently.");
            }
        }
    }
}
