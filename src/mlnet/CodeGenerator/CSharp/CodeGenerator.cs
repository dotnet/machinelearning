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
            // Generate transforms and trainer strings along with using statements
            var result = GenerateTransformsAndTrainers();

            // Generate code for columns
            var columns = this.GenerateColumns();

            // Generate code for prediction Class labels
            var classLabels = this.GenerateClassLabels();

            // Get the type of the label
            var labelType = columnInferenceResult.TextLoaderOptions.Columns.Where(t => t.Name == columnInferenceResult.ColumnInformation.LabelColumn).First().DataKind;
            Type labelTypeCsharp = Utils.GetCSharpType(labelType);


            // Get Namespace
            var namespaceValue = Utils.Normalize(settings.OutputName);

            // Generate code for training and scoring
            var trainFileContent = GenerateTrainCode(result.Usings, result.Trainer, result.PreTrainerTransforms, result.PostTrainerTransforms, columns, classLabels, namespaceValue, pipeline.CacheBeforeTrainer, labelTypeCsharp.Name);
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

        internal string GenerateTrainCode(string usings, string trainer,
            List<string> preTrainerTransforms,
            List<string> postTrainerTransforms,
            IList<string> columns,
            IList<string> classLabels,
            string namespaceValue,
            bool cacheBeforeTrainer,
            string predictionLabelType)
        {
            var trainingAndScoringCodeGen = new MLCodeGen()
            {
                Columns = columns,
                PreTrainerTransforms = preTrainerTransforms,
                PostTrainerTransforms = postTrainerTransforms,
                HasHeader = columnInferenceResult.TextLoaderOptions.HasHeader,
                Separator = columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                AllowQuoting = columnInferenceResult.TextLoaderOptions.AllowQuoting,
                AllowSparse = columnInferenceResult.TextLoaderOptions.AllowSparse,
                TrimWhiteSpace = columnInferenceResult.TextLoaderOptions.TrimWhitespace,
                Trainer = trainer,
                ClassLabels = classLabels,
                GeneratedUsings = usings,
                Path = settings.TrainDataset,
                TestPath = settings.TestDataset,
                TaskType = settings.MlTask.ToString(),
                Namespace = namespaceValue,
                LabelName = settings.LabelName,
                ModelPath = settings.ModelPath,
                CacheBeforeTrainer = cacheBeforeTrainer,
                PredictionLabelType = predictionLabelType
            };

            return trainingAndScoringCodeGen.TransformText();
        }

        internal (string Usings, string Trainer, List<string> PreTrainerTransforms, List<string> PostTrainerTransforms) GenerateTransformsAndTrainers()
        {
            StringBuilder usingsBuilder = new StringBuilder();
            var usings = new List<string>();
            var trainerAndUsings = this.GenerateTrainerAndUsings();

            // Get pre-trainer transforms
            var nodes = pipeline.Nodes.TakeWhile(t => t.NodeType == PipelineNodeType.Transform);
            var preTrainerTransformsAndUsings = this.GenerateTransformsAndUsings(nodes);

            // Get post trainer transforms
            nodes = pipeline.Nodes.SkipWhile(t => t.NodeType == PipelineNodeType.Transform)
                .SkipWhile(t => t.NodeType == PipelineNodeType.Trainer) //skip the trainer
                .TakeWhile(t => t.NodeType == PipelineNodeType.Transform); //post trainer transforms
            var postTrainerTransformsAndUsings = this.GenerateTransformsAndUsings(nodes);

            //Get trainer code and its associated usings.
            var trainer = trainerAndUsings.Item1;
            usings.Add(trainerAndUsings.Item2);

            //Get transforms code and its associated (unique) usings.
            var preTrainerTransforms = preTrainerTransformsAndUsings.Select(t => t.Item1).ToList();
            var postTrainerTransforms = postTrainerTransformsAndUsings.Select(t => t.Item1).ToList();
            usings.AddRange(preTrainerTransformsAndUsings.Select(t => t.Item2));
            usings.AddRange(postTrainerTransformsAndUsings.Select(t => t.Item2));
            usings = usings.Distinct().ToList();

            //Combine all using statements to actual text.
            usingsBuilder = new StringBuilder();
            usings.ForEach(t =>
            {
                if (t != null)
                    usingsBuilder.Append(t);
            });

            return (usingsBuilder.ToString(), trainer, preTrainerTransforms, postTrainerTransforms);
        }

        internal IList<(string, string)> GenerateTransformsAndUsings(IEnumerable<PipelineNode> nodes)
        {
            //var nodes = pipeline.Nodes.TakeWhile(t => t.NodeType == PipelineNodeType.Transform);
            //var nodes = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Transform);
            var results = new List<(string, string)>();
            foreach (var node in nodes)
            {
                ITransformGenerator generator = TransformGeneratorFactory.GetInstance(node);
                results.Add((generator.GenerateTransformer(), generator.GenerateUsings()));
            }

            return results;
        }

        internal IList<(string, string)> GeneratePostTrainerTransformsAndUsings()
        {
            var nodes = pipeline.Nodes.SkipWhile(t => t.NodeType == PipelineNodeType.Transform)
                .SkipWhile(t => t.NodeType == PipelineNodeType.Trainer) //skip the trainer
                .TakeWhile(t => t.NodeType == PipelineNodeType.Transform); //post trainer transforms

            //var nodes = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Transform);
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
            foreach (var column in columnInferenceResult.TextLoaderOptions.Columns)
            {
                StringBuilder sb = new StringBuilder();
                int range = (column.Source[0].Max - column.Source[0].Min).Value;
                bool isArray = range > 0;
                sb.Append(Symbols.PublicSymbol);
                sb.Append(Symbols.Space);
                switch (column.DataKind)
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
            foreach (var column in columnInferenceResult.TextLoaderOptions.Columns)
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

            var def = $"new Column(\"{column.Name}\",DataKind.{column.DataKind},{rangeBuilder.ToString()}),";
            return def;
        }
    }
}
