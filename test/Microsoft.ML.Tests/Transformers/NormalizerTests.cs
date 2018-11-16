﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Projections;
using System;
using System.Collections.Immutable;
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
            string dataPath = GetDataPath(TestDatasets.iris.trainFilename);

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

            var badData1 = new ColumnsCopyingTransformer(Env, ("int1", "float1")).Transform(data);
            var badData2 = new ColumnsCopyingTransformer(Env, ("float0", "float4")).Transform(data);

            TestEstimatorCore(est, data, null, badData1);
            TestEstimatorCore(est, data, null, badData2);

            var outputPath = GetOutputPath("NormalizerEstimator", "normalized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                using (var fs = File.Create(outputPath))
                {
                    var dataView = ColumnSelectingTransformer.CreateDrop(Env, est.Fit(data).Transform(data), "float0");
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
                    new TextLoader.Column("float0", DataKind.R4, new[]{ new TextLoader.Range { Min = 1, VariableEnd = true } })
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
            Assert.True(34 ==  floatBinData.Density);
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
            Assert.Equal(0.140807763f, floatCdfLogMeanData.Stddev);

            var floatCdfLogMeanDataVec = transformer.Columns[13].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<ImmutableArray<float>>;
            Assert.Equal(4, floatCdfLogMeanDataVec.Mean.Length);
            Assert.True(true == floatCdfLogMeanDataVec.UseLog);
            Assert.Equal(4, floatCdfLogMeanDataVec.Stddev.Length);

            var doubleCdfLogMeanData = transformer.Columns[14].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<double>;
            Assert.Equal(1.7562395401953814, doubleCdfLogMeanData.Mean);
            Assert.True(doubleCdfLogMeanData.UseLog);
            Assert.Equal(0.14080776721611848, doubleCdfLogMeanData.Stddev);

            var doubleCdfLogMeanDataVec = transformer.Columns[15].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleCdfLogMeanDataVec.Mean.Length);
            Assert.True(doubleCdfLogMeanDataVec.UseLog);
            Assert.Equal(4, doubleCdfLogMeanDataVec.Stddev.Length);

            Done();
        }

        [Fact]
        public void SimpleConstructorsAndExtensions()
        {
            string dataPath = GetDataPath(TestDatasets.iris.trainFilename);

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
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var data = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var invalidData = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var est = new LpNormalizingEstimator(ML, "features", "lpnorm")
                .Append(new GlobalContrastNormalizingEstimator(ML, "features", "gcnorm"))
                .Append(new VectorWhiteningEstimator(ML, "features", "whitened"));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("NormalizerEstimator", "lpnorm_gcnorm_whitened.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(ML, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = ColumnSelectingTransformer.CreateKeep(ML, savedData, new[] { "lpnorm", "gcnorm", "whitened" });

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
            var data = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var invalidData = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var est = new VectorWhiteningEstimator(ML, "features", "whitened1")
                .Append(new VectorWhiteningEstimator(ML, "features", "whitened2", kind: WhiteningKind.Pca, pcaNum: 5));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("NormalizerEstimator", "whitened.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = ColumnSelectingTransformer.CreateKeep(Env, savedData, new[] { "whitened1", "whitened2" });

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("NormalizerEstimator", "whitened.tsv", digitsOfPrecision: 4);
            Done();
        }

        [Fact]
        public void TestWhiteningCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=whitening{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestWhiteningOldSavingAndLoading()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataView = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource).AsDynamic;
            var pipe = new VectorWhiteningEstimator(ML, "features", "whitened");

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
            var data = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var invalidData = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var est = new LpNormalizingEstimator(ML, "features", "lpNorm1")
                .Append(new LpNormalizingEstimator(ML, "features", "lpNorm2", normKind: LpNormalizingEstimatorBase.NormalizerKind.L1Norm, substractMean: true));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("NormalizerEstimator", "lpNorm.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(ML, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = ColumnSelectingTransformer.CreateKeep(Env, savedData, new[] { "lpNorm1", "lpNorm2" });

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
            var dataView = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource).AsDynamic;
            var pipe = new LpNormalizingEstimator(ML, "features", "whitened");

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
        public void GcnWorkout()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var data = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var invalidData = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadText(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource);

            var est = new GlobalContrastNormalizingEstimator(ML, "features", "gcnNorm1")
                .Append(new GlobalContrastNormalizingEstimator(ML, "features", "gcnNorm2", substractMean: false, useStdDev: true, scale: 3));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("NormalizerEstimator", "gcnNorm.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(ML, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = ColumnSelectingTransformer.CreateKeep(ML, savedData, new[] { "gcnNorm1", "gcnNorm2" });

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
            var dataView = TextLoader.CreateReader(ML,
                c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
                separator: ';', hasHeader: true)
                .Read(dataSource).AsDynamic;
            var pipe = new GlobalContrastNormalizingEstimator(ML, "features", "whitened");

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(ML, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(ML, dataView, ms);
            }
        }
    }
}
