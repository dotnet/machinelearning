// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Normalizers;
using System;
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
                    new TextLoader.Column("double1", DataKind.R8, 1),
                    new TextLoader.Column("double4", DataKind.R8, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("int1", DataKind.I4, 0),
                    new TextLoader.Column("float0", DataKind.R4, new[]{ new TextLoader.Range { Min = 1, VariableEnd = true } }),
                },
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var est = new NormalizingEstimator(Env,
                new NormalizingEstimator.MinMaxColumn("float1"),
                new NormalizingEstimator.MinMaxColumn("float4"),
                new NormalizingEstimator.MinMaxColumn("double1"),
                new NormalizingEstimator.MinMaxColumn("double4"),
                new NormalizingEstimator.BinningColumn("float1", "float1bin"),
                new NormalizingEstimator.BinningColumn("float4", "float4bin"),
                new NormalizingEstimator.BinningColumn("double1", "double1bin"),
                new NormalizingEstimator.BinningColumn("double4", "double4bin"),
                new NormalizingEstimator.MeanVarColumn("float1", "float1mv"),
                new NormalizingEstimator.MeanVarColumn("float4", "float4mv"),
                new NormalizingEstimator.MeanVarColumn("double1", "double1mv"),
                new NormalizingEstimator.MeanVarColumn("double4", "double4mv"),
                new NormalizingEstimator.LogMeanVarColumn("float1", "float1lmv"),
                new NormalizingEstimator.LogMeanVarColumn("float4", "float4lmv"),
                new NormalizingEstimator.LogMeanVarColumn("double1", "double1lmv"),
                new NormalizingEstimator.LogMeanVarColumn("double4", "double4lmv"));

            var data = loader.Read(dataPath);

            var badData1 = new CopyColumnsTransform(Env, ("int1", "float1")).Transform(data);
            var badData2 = new CopyColumnsTransform(Env, ("float0", "float4")).Transform(data);

            TestEstimatorCore(est, data, null, badData1);
            TestEstimatorCore(est, data, null, badData2);

            var outputPath = GetOutputPath("NormalizerEstimator", "normalized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                using (var fs = File.Create(outputPath))
                {
                    var dataView = SelectColumnsTransform.CreateDrop(Env, est.Fit(data).Transform(data), true, "float0");
                    DataSaverUtils.SaveDataView(ch, saver, dataView, fs, keepHidden: true);
                }
            }

            CheckEquality("NormalizerEstimator", "normalized.tsv");

            Done();
        }

        [Fact]
        public void NormalizerParameters()
        {
            string dataPath = GetDataPath("iris.txt");

            var loader = new TextLoader(Env, new TextLoader.Arguments
            {
                Column = new[] {
                    new TextLoader.Column("float1", DataKind.R4, 1),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("double1", DataKind.R8, 1),
                    new TextLoader.Column("double4", DataKind.R8, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("int1", DataKind.I4, 0),
                    new TextLoader.Column("float0", DataKind.R4, new[]{ new TextLoader.Range { Min = 1, VariableEnd = true } }),
                },
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var est = new NormalizingEstimator(Env,
                new NormalizingEstimator.MinMaxColumn("float1"), // 0
                new NormalizingEstimator.MinMaxColumn("float4"),
                new NormalizingEstimator.MinMaxColumn("double1"), // 2
                new NormalizingEstimator.MinMaxColumn("double4"),
                new NormalizingEstimator.BinningColumn("float1", "float1bin"),//4
                new NormalizingEstimator.BinningColumn("float4", "float4bin"), 
                new NormalizingEstimator.BinningColumn("double1", "double1bin"),//6
                new NormalizingEstimator.BinningColumn("double4", "double4bin"),
                new NormalizingEstimator.MeanVarColumn("float1", "float1mv"),//8
                new NormalizingEstimator.MeanVarColumn("float4", "float4mv"),
                new NormalizingEstimator.MeanVarColumn("double1", "double1mv"),//10
                new NormalizingEstimator.MeanVarColumn("double4", "double4mv"),
                new NormalizingEstimator.LogMeanVarColumn("float1", "float1lmv"),//12
                new NormalizingEstimator.LogMeanVarColumn("float4", "float4lmv"),
                new NormalizingEstimator.LogMeanVarColumn("double1", "double1lmv"),//14
                new NormalizingEstimator.LogMeanVarColumn("double4", "double4lmv"));

            var data = loader.Read(dataPath);

            var transformer = est.Fit(data);

            var floatAffineData = transformer.ColumnFunctions[0] as NormalizerTransformer.IAffineData<float>;
            var doubleAffineData = transformer.ColumnFunctions[2] as NormalizerTransformer.IAffineData<double>;

            var floatBinData = transformer.ColumnFunctions[4] as NormalizerTransformer.IBinData<float>;
            var doubleBinData = transformer.ColumnFunctions[6] as NormalizerTransformer.IBinData<double>;

            var floatCdfMeanData = transformer.ColumnFunctions[8] as NormalizerTransformer.ICdfData<float>;
            var doubleCdfMeanData = transformer.ColumnFunctions[10] as NormalizerTransformer.ICdfData<double>;

            var floatCdfLogMeanData = transformer.ColumnFunctions[12] as NormalizerTransformer.ICdfData<float>;
            var doubleCdfLogMeanData = transformer.ColumnFunctions[14] as NormalizerTransformer.ICdfData<double>;

            Done();
        }

        [Fact]
        public void SimpleConstructorsAndExtensions()
        {
            string dataPath = GetDataPath("iris.txt");

            var loader = new TextLoader(Env, new TextLoader.Arguments
            {
                Column = new[] {
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(1, 4) }),
                }
            });

            var data = loader.Read(dataPath);

            var est1 = new NormalizingEstimator(Env, "float4");
            var est2 = new NormalizingEstimator(Env, NormalizingEstimator.NormalizerMode.MinMax, ("float4", "float4"));
            var est3 = new NormalizingEstimator(Env, new NormalizingEstimator.MinMaxColumn("float4"));
            var est4 = ML.Transforms.Normalize(NormalizingEstimator.NormalizerMode.MinMax, ("float4", "float4"));
            var est5 = ML.Transforms.Normalize("float4");

            var data1 = est1.Fit(data).Transform(data);
            var data2 = est2.Fit(data).Transform(data);
            var data3 = est3.Fit(data).Transform(data);
            var data4 = est4.Fit(data).Transform(data);
            var data5 = est5.Fit(data).Transform(data);

            CheckSameSchemas(data1.Schema, data2.Schema);
            CheckSameSchemas(data1.Schema, data3.Schema);
            CheckSameSchemas(data1.Schema, data4.Schema);
            CheckSameSchemas(data1.Schema, data5.Schema);
            CheckSameValues(data1, data2);
            CheckSameValues(data1, data3);
            CheckSameValues(data1, data4);
            CheckSameValues(data1, data5);

            Done();
        }

        [Fact]
        public void LpGcNormAndWhiteningWorkout()
        {
            var env = new ConsoleEnvironment(seed: 0);
            string dataSource = GetDataPath("generated_regression_dataset.csv");
            var data = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var invalidData = TextLoader.CreateReader(env,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var est = new LpNormalizer(env, "features", "lpnorm")
                .Append(new GlobalContrastNormalizer(env, "features", "gcnorm"))
                .Append(new Whitening(env, "features", "whitened"));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "lpnorm_gcnorm_whitened.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = SelectColumnsTransform.CreateKeep(Env, savedData, "lpnorm", "gcnorm", "whitened");

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Text", "lpnorm_gcnorm_whitened.tsv", digitsOfPrecision: 4);
            Done();
        }
    }
}
