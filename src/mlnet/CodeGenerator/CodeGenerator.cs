// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Auto;
using static Microsoft.ML.Data.TextLoader;

namespace Microsoft.ML.CLI
{
    internal class CodeGenerator
    {
        private readonly Pipeline pipeline;
        private readonly ColumnInferenceResult columnInferenceResult;

        public CodeGenerator(Pipeline pipelineToDeconstruct, ColumnInferenceResult columnInferenceResult)
        {
            this.pipeline = pipelineToDeconstruct;
            this.columnInferenceResult = columnInferenceResult;
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
            foreach (var column in columnInferenceResult.Columns)
            {
                StringBuilder sb = new StringBuilder();
                var current = column.Item1;
                int range = (current.Source[0].Max - current.Source[0].Min).Value;
                bool isArray = range > 0;
                sb.Append(Symbols.PublicSymbol);
                sb.Append(Symbols.Space);
                switch (current.Type)
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
                        throw new ArgumentException($"The data type '{current.Type}' is not handled currently.");

                }

                if (range > 0)
                {
                    result.Add("[ColumnName(\"" + current.Name + "\"), VectorType(" + (range + 1) + ")]");
                    sb.Append("[]");
                }
                else
                {
                    result.Add("[ColumnName(\"" + current.Name + "\")]");
                }
                sb.Append(" ");
                sb.Append(Normalize(current.Name));
                sb.Append("{get; set;}");
                result.Add(sb.ToString());
                result.Add("\r\n");
            }
            return result;
        }

        internal IList<string> GenerateColumns()
        {
            var result = new List<string>();
            foreach (var column in columnInferenceResult.Columns)
            {
                result.Add(ConstructColumnDefinition(column.Item1));
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
