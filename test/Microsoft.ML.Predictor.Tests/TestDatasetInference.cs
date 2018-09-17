// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.MLTesting.Inference;

namespace Microsoft.ML.Runtime.RunTests
{
    using Xunit;
    using Xunit.Abstractions;

    public sealed class TestDatasetInference : BaseTestBaseline
    {
        public TestDatasetInference(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact(Skip="Disabled")]
        public void DatasetInferenceTest()
        {
            var datasets = new[]
            {
                GetDataPath(@"..\UCI\adult.train"),
                GetDataPath(@"..\UCI\adult.test"),
                GetDataPath(@"..\UnitTest\breast-cancer.txt"),
            };

            using (var env = new ConsoleEnvironment())
            {
                var h = env.Register("InferDatasetFeatures", seed: 0, verbose: false);

                using (var ch = h.Start("InferDatasetFeatures"))
                {

                    for (int i = 0; i < datasets.Length; i++)
                    {
                        var sample = TextFileSample.CreateFromFullFile(h, datasets[i]);
                        var splitResult = TextFileContents.TrySplitColumns(h, sample, TextFileContents.DefaultSeparators);
                        if (!splitResult.IsSuccess)
                            throw ch.ExceptDecode("Couldn't detect separator.");

                        var typeInfResult = ColumnTypeInference.InferTextFileColumnTypes(Env, sample,
                            new ColumnTypeInference.Arguments
                            {
                                Separator = splitResult.Separator,
                                AllowSparse = splitResult.AllowSparse,
                                AllowQuote = splitResult.AllowQuote,
                                ColumnCount = splitResult.ColumnCount
                            });

                        if (!typeInfResult.IsSuccess)
                            return;

                        ColumnGroupingInference.GroupingColumn[] columns = null;
                        bool hasHeader = false;
                        columns = InferenceUtils.InferColumnPurposes(ch, h, sample, splitResult, out hasHeader);
                        Guid id = new Guid("60C77F4E-DB62-4351-8311-9B392A12968E");
                        var commandArgs = new DatasetFeatureInference.Arguments(typeInfResult.Data,
                            columns.Select(
                                col =>
                                    new DatasetFeatureInference.Column(col.SuggestedName, col.Purpose, col.ItemKind,
                                        col.ColumnRangeSelector)).ToArray(), sample.FullFileSize, sample.ApproximateRowCount,
                            false, id, true);

                        string jsonString = DatasetFeatureInference.InferDatasetFeatures(env, commandArgs);
                        var outFile = string.Format("dataset-inference-result-{0:00}.txt", i);
                        string dataPath = GetOutputPath(@"..\Common\Inference", outFile);
                        using (var sw = new StreamWriter(File.Create(dataPath)))
                            sw.WriteLine(jsonString);

                        CheckEquality(@"..\Common\Inference", outFile);
                    }
                }
            }
            Done();
        }

        [Fact]
        public void InferSchemaCommandTest()
        {
            var datasets = new[]
            {
                GetDataPath(Path.Combine("..", "data", "wikipedia-detox-250-line-data.tsv"))
            };

            using (var env = new ConsoleEnvironment())
            {
                var h = env.Register("InferSchemaCommandTest", seed: 0, verbose: false);
                using (var ch = h.Start("InferSchemaCommandTest"))
                {
                    for (int i = 0; i < datasets.Length; i++)
                    {
                        var outFile = string.Format("dataset-infer-schema-result-{0:00}.txt", i);
                        string dataPath = GetOutputPath(Path.Combine("..", "Common", "Inference"), outFile);
                        var args = new InferSchemaCommand.Arguments()
                        {
                            DataFile = datasets[i],
                            OutputFile = dataPath,
                        };

                        var cmd = new InferSchemaCommand(Env, args);
                        cmd.Run();

                        CheckEquality(Path.Combine("..", "Common", "Inference"), outFile);
                    }
                }
            }
            Done();
        }

        [Fact]
        public void InferRecipesCommandTest()
        {
            var datasets = new Tuple<string, string>[]
            {
                Tuple.Create(
                    GetDataPath(Path.Combine("..", "data", "wikipedia-detox-250-line-data.tsv")),
                    GetDataPath(Path.Combine("..", "data", "wikipedia-detox-250-line-data-schema.txt")))
            };

            using (var env = new ConsoleEnvironment())
            {
                var h = env.Register("InferRecipesCommandTest", seed: 0, verbose: false);
                using (var ch = h.Start("InferRecipesCommandTest"))
                {
                    for (int i = 0; i < datasets.Length; i++)
                    {
                        var outFile = string.Format("dataset-infer-recipe-result-{0:00}.txt", i);
                        string dataPath = GetOutputPath(Path.Combine("..", "Common", "Inference"), outFile);
                        var args = new InferRecipesCommand.Arguments()
                        {
                            DataFile = datasets[i].Item1,
                            SchemaDefinitionFile = datasets[i].Item2,
                            RspOutputFile = dataPath
                        };
                        var cmd = new InferRecipesCommand(Env, args);
                        cmd.Run();

                        CheckEquality(Path.Combine("..", "Common", "Inference"), outFile);
                    }
                }
            }
            Done();
        }
    }
}
