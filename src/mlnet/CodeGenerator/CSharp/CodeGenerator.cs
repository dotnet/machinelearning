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
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.Templates.Console;
using Microsoft.ML.CLI.Utilities;
using static Microsoft.ML.Data.TextLoader;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
{
    internal class CodeGenerator : IProjectGenerator
    {
        private readonly Pipeline pipeline;
        private readonly CodeGeneratorSettings settings;
        private readonly ColumnInferenceResults columnInferenceResult;

        internal CodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResult, CodeGeneratorSettings settings)
        {
            this.pipeline = pipeline;
            this.columnInferenceResult = columnInferenceResult;
            this.settings = settings;
        }

        public void GenerateOutput()
        {
            // Generate Code
            (string trainScoreCode, string projectSourceCode, string consoleHelperCode) = GenerateCode();

            // Write output to file
            WriteOutputToFiles(trainScoreCode, projectSourceCode, consoleHelperCode);
        }

        internal (string, string, string) GenerateCode()
        {
            // Generate usings
            (string usings, string trainer, List<string> transforms) = GenerateUsings();

            // Generate code for columns
            var columns = this.GenerateColumns();

            // Generate code for prediction Class labels
            var classLabels = this.GenerateClassLabels();

            // Get Namespace
            var namespaceValue = Utils.Normalize(settings.OutputName);

            // Generate code for training and scoring
            var trainFileContent = GenerateTrainCode(usings, trainer, transforms, columns, classLabels, namespaceValue);
            var tree = CSharpSyntaxTree.ParseText(trainFileContent);
            var syntaxNode = tree.GetRoot();
            trainFileContent = Formatter.Format(syntaxNode, new AdhocWorkspace()).ToFullString();

            // Generate csproj
            var projectFileContent = GeneratProjectCode();

            // Generate Helper class
            var consoleHelperFileContent = GenerateConsoleHelper(namespaceValue);

            return (trainFileContent, projectFileContent, consoleHelperFileContent);
        }

        internal void WriteOutputToFiles(string trainScoreCode, string projectSourceCode, string consoleHelperCode)
        {
            if (!Directory.Exists(settings.OutputBaseDir))
            {
                Directory.CreateDirectory(settings.OutputBaseDir);
            }
            File.WriteAllText($"{settings.OutputBaseDir}/Program.cs", trainScoreCode);
            File.WriteAllText($"{settings.OutputBaseDir}/{settings.OutputName}.csproj", projectSourceCode);
            File.WriteAllText($"{settings.OutputBaseDir}/ConsoleHelper.cs", consoleHelperCode);
        }

        internal static string GenerateConsoleHelper(string namespaceValue)
        {
            var consoleHelperCodeGen = new ConsoleHelper() { Namespace = namespaceValue };
            return consoleHelperCodeGen.TransformText();
        }

        internal static string GeneratProjectCode()
        {
            var projectCodeGen = new MLProjectGen();
            return projectCodeGen.TransformText();
        }

        internal string GenerateTrainCode(string usings, string trainer, List<string> transforms, IList<string> columns, IList<string> classLabels, string namespaceValue)
        {
            var trainingAndScoringCodeGen = new MLCodeGen()
            {
                Columns = columns,
                Transforms = transforms,
                HasHeader = columnInferenceResult.TextLoaderArgs.HasHeader,
                Separator = columnInferenceResult.TextLoaderArgs.Separators.FirstOrDefault(),
                AllowQuoting = columnInferenceResult.TextLoaderArgs.AllowQuoting,
                AllowSparse = columnInferenceResult.TextLoaderArgs.AllowSparse,
                TrimWhiteSpace = columnInferenceResult.TextLoaderArgs.TrimWhitespace,
                Trainer = trainer,
                ClassLabels = classLabels,
                GeneratedUsings = usings,
                Path = settings.TrainDataset.FullName,
                TestPath = settings.TestDataset?.FullName,
                TaskType = settings.MlTask.ToString(),
                Namespace = namespaceValue,
                LabelName = settings.LabelName,
                ModelPath = settings.ModelPath.FullName
            };

            return trainingAndScoringCodeGen.TransformText();
        }

        internal (string, string, List<string>) GenerateUsings()
        {
            StringBuilder usingsBuilder = new StringBuilder();
            var usings = new List<string>();
            var trainerAndUsings = this.GenerateTrainerAndUsings();
            var transformsAndUsings = this.GenerateTransformsAndUsings();

            //Get trainer code and its associated usings.
            var trainer = trainerAndUsings.Item1;
            usings.Add(trainerAndUsings.Item2);

            //Get transforms code and its associated (unique) usings.
            var transforms = transformsAndUsings.Select(t => t.Item1).ToList();
            usings.AddRange(transformsAndUsings.Select(t => t.Item2));
            usings = usings.Distinct().ToList();

            //Combine all using statements to actual text.
            usingsBuilder = new StringBuilder();
            usings.ForEach(t =>
            {
                if (t != null)
                    usingsBuilder.Append(t);
            });

            return (usingsBuilder.ToString(), trainer, transforms);
        }

        internal IList<(string, string)> GenerateTransformsAndUsings()
        {
            var nodes = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Transform);
            var results = new List<(string, string)>();
            foreach (var node in nodes)
            {
                ITransformGenerator generator = TransformGeneratorFactory.GetInstance(node);
                results.Add((generator.GenerateTransformer(), generator.GenerateUsings()));
            }

            return results;
        }

        internal (string, string) GenerateTrainerAndUsings()
        {
            ITrainerGenerator generator = TrainerGeneratorFactory.GetInstance(pipeline);
            var trainerString = generator.GenerateTrainer();
            var trainerUsings = generator.GenerateUsings();
            return (trainerString, trainerUsings);
        }

        internal IList<string> GenerateClassLabels()
        {
            IList<string> result = new List<string>();
            var label_column = Utils.Sanitize(columnInferenceResult.ColumnInformation.LabelColumn);
            foreach (var column in columnInferenceResult.TextLoaderArgs.Columns)
            {
                StringBuilder sb = new StringBuilder();
                int range = (column.Source[0].Max - column.Source[0].Min).Value;
                bool isArray = range > 0;
                sb.Append(Symbols.PublicSymbol);
                sb.Append(Symbols.Space);
                switch (column.Type)
                {
                    case Microsoft.ML.Data.DataKind.TX:
                        sb.Append(Symbols.StringSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.BL:
                        sb.Append(Symbols.BoolSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.R4:
                        sb.Append(Symbols.FloatSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.R8:
                        sb.Append(Symbols.DoubleSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.I4:
                        sb.Append(Symbols.IntSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.U4:
                        sb.Append(Symbols.UIntSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.I8:
                        sb.Append(Symbols.LongSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.U8:
                        sb.Append(Symbols.UlongSymbol);
                        break;
                    default:
                        throw new ArgumentException($"The data type '{column.Type}' is not handled currently.");

                }

                if (range > 0)
                {
                    result.Add($"[ColumnName(\"{column.Name}\"),LoadColumn({column.Source[0].Min}, {column.Source[0].Max}) VectorType({(range + 1)})]");
                    sb.Append("[]");
                }
                else
                {
                    result.Add($"[ColumnName(\"{column.Name}\"), LoadColumn({column.Source[0].Min})]");
                }
                sb.Append(" ");
                sb.Append(Utils.Normalize(column.Name));
                sb.Append("{get; set;}");
                result.Add(sb.ToString());
                result.Add("\r\n");
            }
            return result;
        }

        internal IList<string> GenerateColumns()
        {
            var result = new List<string>();
            foreach (var column in columnInferenceResult.TextLoaderArgs.Columns)
            {
                result.Add(ConstructColumnDefinition(column));
            }
            return result;
        }

        private static string ConstructColumnDefinition(Column column)
        {
            Range[] source = column.Source;
            StringBuilder rangeBuilder = new StringBuilder();
            if (source.Length == 1)
            {
                if (source[0].Min == source[0].Max)
                    rangeBuilder.Append($"{source[0].Max}");
                else
                {
                    rangeBuilder.Append("new[]{");
                    rangeBuilder.Append($"new Range({ source[0].Min },{ source[0].Max}),");
                    rangeBuilder.Remove(rangeBuilder.Length - 1, 1);
                    rangeBuilder.Append("}");
                }
            }
            else
            {
                rangeBuilder.Append("new[]{");
                foreach (var range in source)
                {
                    if (range.Min == range.Max)
                    {
                        rangeBuilder.Append($"new Range({range.Min}),");
                    }
                    else
                    {
                        rangeBuilder.Append($"new Range({range.Min},{range.Max}),");
                    }
                }
                rangeBuilder.Remove(rangeBuilder.Length - 1, 1);
                rangeBuilder.Append("}");
            }

            var def = $"new Column(\"{column.Name}\",DataKind.{column.Type},{rangeBuilder.ToString()}),";
            return def;
        }
    }
}
