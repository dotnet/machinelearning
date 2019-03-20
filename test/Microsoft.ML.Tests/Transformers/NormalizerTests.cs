// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
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
            string dataPath = GetDataPath(TestDatasets.iris.trainFilename);

            var loader = new TextLoader(Env, new TextLoader.Options
            {
                Columns = new[] {
                    new TextLoader.Column("float1", DataKind.Single, 1),
                    new TextLoader.Column("float4", DataKind.Single, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("double1", DataKind.Double, 1),
                    new TextLoader.Column("double4", DataKind.Double, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("int1", DataKind.Int32, 0),
                    new TextLoader.Column("float0", DataKind.Single, new[]{ new TextLoader.Range { Min = 1, VariableEnd = true } }),
                },
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var est = new NormalizingEstimator(Env,
                new NormalizingEstimator.MinMaxColumnOptions("float1"),
                new NormalizingEstimator.MinMaxColumnOptions("float4"),
                new NormalizingEstimator.MinMaxColumnOptions("double1"),
                new NormalizingEstimator.MinMaxColumnOptions("double4"),
                new NormalizingEstimator.BinningColumnOptions("float1bin", "float1"),
                new NormalizingEstimator.BinningColumnOptions("float4bin", "float4"),
                new NormalizingEstimator.BinningColumnOptions("double1bin", "double1"),
                new NormalizingEstimator.BinningColumnOptions("double4bin", "double4"),
                new NormalizingEstimator.SupervisedBinningColumOptions("float1supervisedbin", "float1", labelColumnName: "int1"),
                new NormalizingEstimator.SupervisedBinningColumOptions("float4supervisedbin", "float4", labelColumnName: "int1"),
                new NormalizingEstimator.SupervisedBinningColumOptions("double1supervisedbin", "double1", labelColumnName: "int1"),
                new NormalizingEstimator.SupervisedBinningColumOptions("double4supervisedbin", "double4", labelColumnName: "int1"),
                new NormalizingEstimator.MeanVarianceColumnOptions("float1mv", "float1"),
                new NormalizingEstimator.MeanVarianceColumnOptions("float4mv", "float4"),
                new NormalizingEstimator.MeanVarianceColumnOptions("double1mv", "double1"),
                new NormalizingEstimator.MeanVarianceColumnOptions("double4mv", "double4"),
                new NormalizingEstimator.LogMeanVarianceColumnOptions("float1lmv", "float1"),
                new NormalizingEstimator.LogMeanVarianceColumnOptions("float4lmv", "float4"),
                new NormalizingEstimator.LogMeanVarianceColumnOptions("double1lmv", "double1"),
                new NormalizingEstimator.LogMeanVarianceColumnOptions("double4lmv", "double4"));

            var data = loader.Load(dataPath);

            var badData1 = new ColumnCopyingTransformer(Env, ("float1", "int1")).Transform(data);
            var badData2 = new ColumnCopyingTransformer(Env, ("float4", "float0")).Transform(data);

            TestEstimatorCore(est, data, null, badData1);
            TestEstimatorCore(est, data, null, badData2);

            var outputPath = GetOutputPath("NormalizerEstimator", "normalized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                using (var fs = File.Create(outputPath))
                {
                    var transformedData = est.Fit(data).Transform(data);
                    var dataView = ML.Transforms.DropColumns(new[] { "float0" }).Fit(transformedData).Transform(transformedData);
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

            var loader = new TextLoader(Env, new TextLoader.Options
            {
                Columns = new[] {
                    new TextLoader.Column("float1", DataKind.Single, 1),
                    new TextLoader.Column("float4", DataKind.Single, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("double1", DataKind.Double, 1),
                    new TextLoader.Column("double4", DataKind.Double, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("int1", DataKind.Int32, 0),
                    new TextLoader.Column("float0", DataKind.Single, new[]{ new TextLoader.Range { Min = 1, VariableEnd = true } })
                },
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var est = new NormalizingEstimator(Env,
                new NormalizingEstimator.MinMaxColumnOptions("float1"),
                new NormalizingEstimator.MinMaxColumnOptions("float4"),
                new NormalizingEstimator.MinMaxColumnOptions("double1"),
                new NormalizingEstimator.MinMaxColumnOptions("double4"),
                new NormalizingEstimator.BinningColumnOptions("float1bin", "float1"),
                new NormalizingEstimator.BinningColumnOptions("float4bin", "float4"),
                new NormalizingEstimator.BinningColumnOptions("double1bin", "double1"),
                new NormalizingEstimator.BinningColumnOptions("double4bin", "double4"),
                new NormalizingEstimator.MeanVarianceColumnOptions("float1mv", "float1"),
                new NormalizingEstimator.MeanVarianceColumnOptions("float4mv", "float4"),
                new NormalizingEstimator.MeanVarianceColumnOptions("double1mv", "double1"),
                new NormalizingEstimator.MeanVarianceColumnOptions("double4mv", "double4"),
                new NormalizingEstimator.LogMeanVarianceColumnOptions("float1lmv", "float1"),
                new NormalizingEstimator.LogMeanVarianceColumnOptions("float4lmv", "float4"),
                new NormalizingEstimator.LogMeanVarianceColumnOptions("double1lmv", "double1"),
                new NormalizingEstimator.LogMeanVarianceColumnOptions("double4lmv", "double4"));

            var data = loader.Load(dataPath);

            var transformer = est.Fit(data);

            var floatAffineData = transformer.Columns[0].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<float>;
            Assert.Equal(0.12658228f, floatAffineData.Scale);
            Assert.Equal(0, floatAffineData.Offset);

            var floatAffineDataVec = transformer.Columns[1].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<float>>;
            Assert.Equal(4, floatAffineDataVec.Scale.Length);
            Assert.Empty(floatAffineDataVec.Offset);

            var doubleAffineData = transformer.Columns[2].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<double>;
            Assert.Equal(0.12658227848101264, doubleAffineData.Scale);
            Assert.Equal(0, doubleAffineData.Offset);

            var doubleAffineDataVec = transformer.Columns[3].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleAffineDataVec.Scale.Length);
            Assert.Empty(doubleAffineDataVec.Offset);

            var floatBinData = transformer.Columns[4].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<float>;
            Assert.True(35 == floatBinData.UpperBounds.Length);
            Assert.True(34 == floatBinData.Density);
            Assert.True(0 == floatBinData.Offset);

            var floatBinDataVec = transformer.Columns[5].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<ImmutableArray<float>>;
            Assert.True(4 == floatBinDataVec.UpperBounds.Length);
            Assert.True(35 == floatBinDataVec.UpperBounds[0].Length);
            Assert.True(4 == floatBinDataVec.Density.Length);
            Assert.True(0 == floatBinDataVec.Offset.Length);

            var doubleBinData = transformer.Columns[6].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<double>;
            Assert.Equal(35, doubleBinData.UpperBounds.Length);
            Assert.Equal(34, doubleBinData.Density);
            Assert.Equal(0, doubleBinData.Offset);

            var doubleBinDataVec = transformer.Columns[7].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(35, doubleBinDataVec.UpperBounds[0].Length);
            Assert.Equal(4, doubleBinDataVec.Density.Length);
            Assert.Empty(doubleBinDataVec.Offset);

            var floatCdfMeanData = transformer.Columns[8].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<float>;
            Assert.Equal(0.169309646f, floatCdfMeanData.Scale);
            Assert.Equal(0, floatCdfMeanData.Offset);

            var floatCdfMeanDataVec = transformer.Columns[9].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<float>>;
            Assert.Equal(0.16930964589119f, floatCdfMeanDataVec.Scale[0]);
            Assert.Equal(4, floatCdfMeanDataVec.Scale.Length);
            Assert.Empty(floatCdfMeanDataVec.Offset);

            var doubleCdfMeanData = transformer.Columns[10].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<double>;
            Assert.Equal(0.16930963784387665, doubleCdfMeanData.Scale);
            Assert.Equal(0, doubleCdfMeanData.Offset);

            var doubleCdfMeanDataVec = transformer.Columns[11].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleCdfMeanDataVec.Scale.Length);
            Assert.Empty(doubleCdfMeanDataVec.Offset);

            var floatCdfLogMeanData = transformer.Columns[12].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<float>;
            Assert.Equal(1.75623953f, floatCdfLogMeanData.Mean);
            Assert.True(true == floatCdfLogMeanData.UseLog);
            Assert.Equal(0.140807763f, floatCdfLogMeanData.StandardDeviation);

            var floatCdfLogMeanDataVec = transformer.Columns[13].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<ImmutableArray<float>>;
            Assert.Equal(4, floatCdfLogMeanDataVec.Mean.Length);
            Assert.True(true == floatCdfLogMeanDataVec.UseLog);
            Assert.Equal(4, floatCdfLogMeanDataVec.StandardDeviation.Length);

            var doubleCdfLogMeanData = transformer.Columns[14].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<double>;
            Assert.Equal(1.7562395401953814, doubleCdfLogMeanData.Mean);
            Assert.True(doubleCdfLogMeanData.UseLog);
            Assert.Equal(0.14080776721611848, doubleCdfLogMeanData.StandardDeviation);

            var doubleCdfLogMeanDataVec = transformer.Columns[15].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleCdfLogMeanDataVec.Mean.Length);
            Assert.True(doubleCdfLogMeanDataVec.UseLog);
            Assert.Equal(4, doubleCdfLogMeanDataVec.StandardDeviation.Length);

            Done();
        }

        [Fact]
        public void SimpleConstructorsAndExtensions()
        {
            string dataPath = GetDataPath(TestDatasets.iris.trainFilename);

            var loader = new TextLoader(Env, new TextLoader.Options
            {
                Columns = new[] {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("float4", DataKind.Single, new[]{new TextLoader.Range(1, 4) }),
                }
            });

            var data = loader.Load(dataPath);

            var est1 = new NormalizingEstimator(Env, "float4");
            var est2 = new NormalizingEstimator(Env, NormalizingEstimator.NormalizationMode.MinMax, ("float4", "float4"));
            var est3 = new NormalizingEstimator(Env, new NormalizingEstimator.MinMaxColumnOptions("float4"));
            var est4 = ML.Transforms.Normalize(NormalizingEstimator.NormalizationMode.MinMax, ("float4", "float4"));
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

            // Tests for SupervisedBinning
            var est6 = new NormalizingEstimator(Env, NormalizingEstimator.NormalizationMode.SupervisedBinning, ("float4", "float4"));
            var est7 = new NormalizingEstimator(Env, new NormalizingEstimator.SupervisedBinningColumOptions("float4"));
            var est8 = ML.Transforms.Normalize(NormalizingEstimator.NormalizationMode.SupervisedBinning, ("float4", "float4"));

            var data6 = est6.Fit(data).Transform(data);
            var data7 = est7.Fit(data).Transform(data);
            var data8 = est8.Fit(data).Transform(data);
            CheckSameSchemas(data6.Schema, data7.Schema);
            CheckSameSchemas(data6.Schema, data8.Schema);
            CheckSameValues(data6, data7);
            CheckSameValues(data6, data8);

            Done();
        }

        [Fact]
        public void LpGcNormAndWhiteningWorkout()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var data = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource);

            var invalidData = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource);

            var est = ML.Transforms.NormalizeLpNorm("lpnorm", "features")
                .Append(ML.Transforms.NormalizeGlobalContrast("gcnorm", "features"))
                .Append(new VectorWhiteningEstimator(ML, "whitened", "features"));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("NormalizerEstimator", "lpnorm_gcnorm_whitened.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = ML.Transforms.SelectColumns("lpnorm", "gcnorm", "whitened").Fit(savedData).Transform(savedData);

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("NormalizerEstimator", "lpnorm_gcnorm_whitened.tsv", digitsOfPrecision: 4);
            Done();
        }

        [Fact]
        public void WhiteningWorkout()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var data = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource);

            var invalidData = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource);

            var est = new VectorWhiteningEstimator(ML, "whitened1", "features")
                .Append(new VectorWhiteningEstimator(ML, "whitened2", "features", kind: WhiteningKind.PrincipalComponentAnalysis, rank: 5));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("NormalizerEstimator", "whitened.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = ML.Transforms.SelectColumns("whitened1", "whitened2").Fit(savedData).Transform(savedData);

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("NormalizerEstimator", "whitened.tsv", digitsOfPrecision: 4);
            Done();
        }

        [Fact]
        public void TestWhiteningCommandLine()
        {
            // typeof helps to load the VectorWhiteningTransformer type.
            Type type = typeof(VectorWhiteningTransformer);
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=whitening{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestWhiteningOldSavingAndLoading()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataView = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource).AsDynamic;
            var pipe = new VectorWhiteningEstimator(ML, "whitened", "features");

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(ML, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(ML, dataView, ms);
            }
            Done();
        }

        [Fact]
        public void LpNormWorkout()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var data = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource);

            var invalidData = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource);

            var est = ML.Transforms.NormalizeLpNorm("lpNorm1", "features")
                .Append(ML.Transforms.NormalizeLpNorm("lpNorm2", "features", norm: LpNormNormalizingEstimatorBase.NormFunction.L1, ensureZeroMean: true));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("NormalizerEstimator", "lpNorm.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = ML.Transforms.SelectColumns("lpNorm1", "lpNorm2").Fit(savedData).Transform(savedData);

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("NormalizerEstimator", "lpNorm.tsv");
            Done();
        }

        [Fact]
        public void TestLpNormCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=LpNormNormalizer{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestLpNormOldSavingAndLoading()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataView = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource).AsDynamic;
            var pipe = ML.Transforms.NormalizeLpNorm("whitened", "features");

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(ML, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(ML, dataView, ms);
            }
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void GcnWorkout()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var data = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource);

            var invalidData = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource);

            var est = ML.Transforms.NormalizeGlobalContrast("gcnNorm1", "features")
                .Append(ML.Transforms.NormalizeGlobalContrast("gcnNorm2", "features", ensureZeroMean: false, ensureUnitStandardDeviation: true, scale: 3));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("NormalizerEstimator", "gcnNorm.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = ML.Transforms.SelectColumns("gcnNorm1", "gcnNorm2").Fit(savedData).Transform(savedData);

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("NormalizerEstimator", "gcnNorm.tsv", digitsOfPrecision: 4);
            Done();
        }

        [Fact]
        public void TestGcnNormCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=GcnTransform{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestGcnNormOldSavingAndLoading()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataView = TextLoaderStatic.CreateLoader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Load(dataSource).AsDynamic;
            var pipe = ML.Transforms.NormalizeGlobalContrast("whitened", "features");

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(ML, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(ML, dataView, ms);
            }
        }

        [Fact]
        void TestNormalizeBackCompatibility()
        {
            var dataFile = GetDataPath("breast-cancer.txt");
            var dataView = TextLoader.Create(ML, new TextLoader.Options(), new MultiFileSource(dataFile));
            string chooseModelPath = GetDataPath("backcompat/ap_with_norm.zip");
            using (FileStream fs = File.OpenRead(chooseModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                Assert.Equal(3, result.Schema.Count);
            }
        }
    }
}
