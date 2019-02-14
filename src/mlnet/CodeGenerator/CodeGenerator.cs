// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Auto;
using mlnet.Templates;
using static Microsoft.ML.Data.TextLoader;

namespace Microsoft.ML.CLI
{
    internal class CodeGenerator
    {
        private readonly Pipeline pipeline;
        private readonly CodeGeneratorOptions options;
        private readonly (Arguments, IEnumerable<(string, ColumnPurpose)>) columnInferenceResult;

        internal CodeGenerator(Pipeline pipeline, (Arguments, IEnumerable<(string, ColumnPurpose)>) columnInferenceResult, CodeGeneratorOptions options)
        {
            this.pipeline = pipeline;
            this.columnInferenceResult = columnInferenceResult;
            this.options = options;
        }

        internal void GenerateOutput()
        {
            var trainerAndUsings = this.GenerateTrainerAndUsings();
            var transformsAndUsings = this.GenerateTransformsAndUsings();

            //Capture all the usings
            var usings = new List<string>();

            //Get trainer code and its associated usings.
            var trainer = trainerAndUsings.Item1;
            usings.Add(trainerAndUsings.Item2);

            //Get transforms code and its associated (unique) usings.
            var transforms = transformsAndUsings.Select(t => t.Item1).ToList();
            usings.AddRange(transformsAndUsings.Select(t => t.Item2));
            usings = usings.Distinct().ToList();

            //Combine all using statements to actual text.
            StringBuilder usingsBuilder = new StringBuilder();
            usings.ForEach(t =>
            {
                if (t != null)
                    usingsBuilder.Append(t);
            });

            //Generate code for columns
            var columns = this.GenerateColumns();

            //Generate code for prediction Class labels
            var classLabels = this.GenerateClassLabels();

            MLCodeGen codeGen = new MLCodeGen()
            {
                Path = options.TrainDataset.FullName,
                TestPath = options.TestDataset?.FullName,
                Columns = columns,
                Transforms = transforms,
                HasHeader = columnInferenceResult.Item1.HasHeader,
                Separator = columnInferenceResult.Item1.Separators.FirstOrDefault(),
                AllowQuoting = columnInferenceResult.Item1.AllowQuoting,
                AllowSparse = columnInferenceResult.Item1.AllowSparse,
                TrimWhiteSpace = columnInferenceResult.Item1.TrimWhitespace,
                Trainer = trainer,
                TaskType = options.MlTask.ToString(),
                ClassLabels = classLabels,
                GeneratedUsings = usingsBuilder.ToString()
            };

            MLProjectGen csProjGenerator = new MLProjectGen();
            ConsoleHelper consoleHelper = new ConsoleHelper();
            var trainScoreCode = codeGen.TransformText();
            var projectSourceCode = csProjGenerator.TransformText();
            var consoleHelperCode = consoleHelper.TransformText();
            var outputFolder = Path.Combine(options.OutputBaseDir, options.OutputName);
            if (!Directory.Exists(outputFolder))
            {
                Directory.CreateDirectory(outputFolder);
            }
            File.WriteAllText($"{outputFolder}/Train.cs", trainScoreCode);
            File.WriteAllText($"{outputFolder}/{options.OutputName}.csproj", projectSourceCode);
            File.WriteAllText($"{outputFolder}/ConsoleHelper.cs", consoleHelperCode);
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
            foreach (var column in columnInferenceResult.Item1.Column)
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
                sb.Append(Normalize(column.Name));
                sb.Append("{get; set;}");
                result.Add(sb.ToString());
                result.Add("\r\n");
            }
            return result;
        }

        internal IList<string> GenerateColumns()
        {
            var result = new List<string>();
            foreach (var column in columnInferenceResult.Item1.Column)
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

        private static string Normalize(string inputColumn)
        {
            //check if first character is int
            if (!string.IsNullOrEmpty(inputColumn) && int.TryParse(inputColumn.Substring(0, 1), out int val))
            {
                inputColumn = "Col" + inputColumn;
                return inputColumn;
            }
            switch (inputColumn)
            {
                case null: throw new ArgumentNullException(nameof(inputColumn));
                case "": throw new ArgumentException($"{nameof(inputColumn)} cannot be empty", nameof(inputColumn));
                default: return inputColumn.First().ToString().ToUpper() + inputColumn.Substring(1);
            }
        }

    }
}
