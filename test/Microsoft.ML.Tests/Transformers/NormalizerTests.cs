// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class NormalizerTests : TestDataPipeBase
    {
        public NormalizerTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void NormalizerWorkout()
        {
            string dataPath = GetDataPath("iris.txt");

            var loader = new TextLoader(Env, new TextLoader.Arguments
            {
                Column = new[] {
                    new TextLoader.Column("float1", DataKind.R4, 1),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("double1", DataKind.R8, 1),
                    new TextLoader.Column("double4", DataKind.R8, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("int1", DataKind.I4, 0),
                    new TextLoader.Column("float0", DataKind.R4, new[]{ new TextLoader.Range { Min = 1, VariableEnd = true } }),
                },
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var est = new Normalizer(Env,
                new Normalizer.MinMaxColumn("float1"),
                new Normalizer.MinMaxColumn("float4"),
                new Normalizer.MinMaxColumn("double1"),
                new Normalizer.MinMaxColumn("double4"),
                new Normalizer.BinningColumn("float1", "float1bin"),
                new Normalizer.BinningColumn("float4", "float4bin"),
                new Normalizer.BinningColumn("double1", "double1bin"),
                new Normalizer.BinningColumn("double4", "double4bin"),
                new Normalizer.MeanVarColumn("float1", "float1mv"),
                new Normalizer.MeanVarColumn("float4", "float4mv"),
                new Normalizer.MeanVarColumn("double1", "double1mv"),
                new Normalizer.MeanVarColumn("double4", "double4mv"),
                new Normalizer.LogMeanVarColumn("float1", "float1lmv"),
                new Normalizer.LogMeanVarColumn("float4", "float4lmv"),
                new Normalizer.LogMeanVarColumn("double1", "double1lmv"),
                new Normalizer.LogMeanVarColumn("double4", "double4lmv"));

            var data = loader.Read(new MultiFileSource(dataPath));

            var badData1 = new CopyColumnsTransform(Env, ("int1", "float1")).Transform(data);
            var badData2 = new CopyColumnsTransform(Env, ("float0", "float4")).Transform(data);

            TestEstimatorCore(est, data, null, badData1);
            TestEstimatorCore(est, data, null, badData2);

            var outputPath = GetOutputPath("Normalizer", "normalized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, new DropColumnsTransform(Env, est.Fit(data).Transform(data), "float0"), fs, keepHidden: true);
            }

            CheckEquality("Normalizer", "normalized.tsv");

            Done();
        }

        [Fact]
        public void SimpleConstructors()
        {
            string dataPath = GetDataPath("iris.txt");

            var loader = new TextLoader(Env, new TextLoader.Arguments
            {
                Column = new[] {
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(1, 4) }),
                }
            });

            var data = loader.Read(new MultiFileSource(dataPath));

            var est1 = new Normalizer(Env, "float4");
            var est2 = new Normalizer(Env, Normalizer.NormalizerMode.MinMax, ("float4", "float4"));
            var est3 = new Normalizer(Env, new Normalizer.MinMaxColumn("float4"));

            var data1 = est1.Fit(data).Transform(data);
            var data2 = est2.Fit(data).Transform(data);
            var data3 = est3.Fit(data).Transform(data);

            CheckSameSchemas(data1.Schema, data2.Schema);
            CheckSameSchemas(data1.Schema, data3.Schema);
            CheckSameValues(data1, data2);
            CheckSameValues(data1, data3);

            Done();
        }
    }
}
