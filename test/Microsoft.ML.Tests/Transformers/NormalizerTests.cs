// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Experimental;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;
using static Microsoft.ML.Transforms.NormalizingTransformer;

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
                new NormalizingEstimator.LogMeanVarianceColumnOptions("double4lmv", "double4"),
                new NormalizingEstimator.RobustScalingColumnOptions("float1rb", "float1"),
                new NormalizingEstimator.RobustScalingColumnOptions("float4rb", "float4"),
                new NormalizingEstimator.RobustScalingColumnOptions("double1rb", "double1"),
                new NormalizingEstimator.RobustScalingColumnOptions("double4rb", "double4"));

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
        public void NormalizerParametersMultiColumnApi()
        {
            string dataPath = GetDataPath("iris.txt");
            var context = new MLContext(seed: 0);

            var loader = new TextLoader(context, new TextLoader.Options
            {
                Columns = new[] {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("float1", DataKind.Single, 1),
                    new TextLoader.Column("float4", DataKind.Single, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("double1", DataKind.Double, 1),
                    new TextLoader.Column("double4", DataKind.Double, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("int1", DataKind.Int32, 0),
                    new TextLoader.Column("float0", DataKind.Single, new[]{ new TextLoader.Range { Min = 1, VariableEnd = true } })
                },
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var est = context.Transforms.NormalizeMinMax(
                new[] { new InputOutputColumnPair("float1"), new InputOutputColumnPair("float4"),
                    new InputOutputColumnPair("double1"), new InputOutputColumnPair("double4"), })
                    .Append(context.Transforms.NormalizeBinning(
                                new[] {new InputOutputColumnPair("float1bin", "float1"), new InputOutputColumnPair("float4bin", "float4"),
                                    new InputOutputColumnPair("double1bin", "double1"), new InputOutputColumnPair("double4bin", "double4")}))
                    .Append(context.Transforms.NormalizeMeanVariance(
                                new[] {new InputOutputColumnPair("float1mv", "float1"), new InputOutputColumnPair("float4mv", "float4"),
                                    new InputOutputColumnPair("double1mv", "double1"), new InputOutputColumnPair("double4mv", "double4")}))
                    .Append(context.Transforms.NormalizeLogMeanVariance(
                                new[] {new InputOutputColumnPair("float1lmv", "float1"), new InputOutputColumnPair("float4lmv", "float4"),
                                    new InputOutputColumnPair("double1lmv", "double1"), new InputOutputColumnPair("double4lmv", "double4")}))
                    .Append(context.Transforms.NormalizeSupervisedBinning(
                                new[] {new InputOutputColumnPair("float1nsb", "float1"), new InputOutputColumnPair("float4nsb", "float4"),
                                    new InputOutputColumnPair("double1nsb", "double1"), new InputOutputColumnPair("double4nsb", "double4")}));

            var data = loader.Load(dataPath);

            var transformer = est.Fit(data);
            var transformers = transformer.ToImmutableArray();
            var floatAffineModel = ((NormalizingTransformer)transformers[0]).Columns[0].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<float>;
            Assert.Equal(0.12658228f, floatAffineModel.Scale);
            Assert.Equal(0, floatAffineModel.Offset);

            var floatAffineModelVec = ((NormalizingTransformer)transformers[0]).Columns[1].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<float>>;
            Assert.Equal(4, floatAffineModelVec.Scale.Length);
            Assert.Empty(floatAffineModelVec.Offset);

            var doubleAffineModel = ((NormalizingTransformer)transformers[0]).Columns[2].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<double>;
            Assert.Equal(0.12658227848101264, doubleAffineModel.Scale);
            Assert.Equal(0, doubleAffineModel.Offset);

            var doubleAffineModelVector = ((NormalizingTransformer)transformers[0]).Columns[3].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleAffineModelVector.Scale.Length);
            Assert.Equal(0.12658227848101264, doubleAffineModelVector.Scale[0]);
            Assert.Equal(0.4, doubleAffineModelVector.Scale[3]);
            Assert.Empty(doubleAffineModelVector.Offset);

            var floatBinModel = ((NormalizingTransformer)transformers[1]).Columns[0].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<float>;
            Assert.True(35 == floatBinModel.UpperBounds.Length);
            Assert.True(0.550632954f == floatBinModel.UpperBounds[0]);
            Assert.True(float.PositiveInfinity == floatBinModel.UpperBounds[34]);
            Assert.True(34 == floatBinModel.Density);
            Assert.True(0 == floatBinModel.Offset);

            var floatBinModelVector = ((NormalizingTransformer)transformers[1]).Columns[1].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<ImmutableArray<float>>;
            Assert.True(4 == floatBinModelVector.UpperBounds.Length);
            Assert.True(35 == floatBinModelVector.UpperBounds[0].Length);
            Assert.True(0.550632954f == floatBinModelVector.UpperBounds[0][0]);
            Assert.True(float.PositiveInfinity == floatBinModelVector.UpperBounds[0][floatBinModelVector.UpperBounds[0].Length - 1]);
            Assert.True(0.0600000024f == floatBinModelVector.UpperBounds[3][0]);
            Assert.True(float.PositiveInfinity == floatBinModelVector.UpperBounds[3][floatBinModelVector.UpperBounds[3].Length - 1]);
            Assert.True(4 == floatBinModelVector.Density.Length);
            Assert.True(0 == floatBinModelVector.Offset.Length);

            var doubleBinModel = ((NormalizingTransformer)transformers[1]).Columns[2].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<double>;
            Assert.Equal(35, doubleBinModel.UpperBounds.Length);
            Assert.True(0.550632911392405 == doubleBinModel.UpperBounds[0]);
            Assert.True(double.PositiveInfinity == doubleBinModel.UpperBounds[34]);
            Assert.Equal(34, doubleBinModel.Density);
            Assert.Equal(0, doubleBinModel.Offset);

            var doubleBinModelVector = ((NormalizingTransformer)transformers[1]).Columns[3].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(35, doubleBinModelVector.UpperBounds[0].Length);
            Assert.True(0.550632911392405 == doubleBinModelVector.UpperBounds[0][0]);
            Assert.True(double.PositiveInfinity == doubleBinModelVector.UpperBounds[0][doubleBinModelVector.UpperBounds[0].Length - 1]);
            Assert.True(0.060000000000000012 == doubleBinModelVector.UpperBounds[3][0]);
            Assert.True(double.PositiveInfinity == doubleBinModelVector.UpperBounds[3][doubleBinModelVector.UpperBounds[3].Length - 1]);
            Assert.Equal(4, doubleBinModelVector.Density.Length);
            Assert.Empty(doubleBinModelVector.Offset);

            var floatCdfMeanModel = ((NormalizingTransformer)transformers[2]).Columns[0].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<float>;
            Assert.Equal(1.33754611f, floatCdfMeanModel.Scale);
            Assert.Equal(0, floatCdfMeanModel.Offset);

            var floatCdfMeanModelVector = ((NormalizingTransformer)transformers[2]).Columns[1].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<float>>;
            Assert.Equal(1.33754611f, floatCdfMeanModelVector.Scale[0]);
            Assert.Equal(1.75526536f, floatCdfMeanModelVector.Scale[3]);
            Assert.Equal(4, floatCdfMeanModelVector.Scale.Length);
            Assert.Empty(floatCdfMeanModelVector.Offset);

            var doubleCdfMeanModel = ((NormalizingTransformer)transformers[2]).Columns[2].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<double>;
            Assert.Equal(1.3375461389666252, doubleCdfMeanModel.Scale);
            Assert.Equal(0, doubleCdfMeanModel.Offset);

            var doubleCdfMeanModelVector = ((NormalizingTransformer)transformers[2]).Columns[3].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleCdfMeanModelVector.Scale.Length);
            Assert.True(1.3375461389666252 == doubleCdfMeanModelVector.Scale[0]);
            Assert.True(1.7552654477786787 == doubleCdfMeanModelVector.Scale[3]);
            Assert.Empty(doubleCdfMeanModelVector.Offset);

            var floatCdfLogMeanModel = ((NormalizingTransformer)transformers[3]).Columns[0].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<float>;
            Assert.Equal(-0.310623198747635f, floatCdfLogMeanModel.Mean);
            Assert.True(true == floatCdfLogMeanModel.UseLog);
            Assert.Equal(0.140807763f, floatCdfLogMeanModel.StandardDeviation);

            var floatCdfLogMeanModelVector = ((NormalizingTransformer)transformers[3]).Columns[1].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<ImmutableArray<float>>;
            Assert.Equal(4, floatCdfLogMeanModelVector.Mean.Length);
            Assert.True(-0.3106232f == floatCdfLogMeanModelVector.Mean[0]);
            Assert.True(-1.08362031f == floatCdfLogMeanModelVector.Mean[3]);
            Assert.True(true == floatCdfLogMeanModelVector.UseLog);
            Assert.Equal(4, floatCdfLogMeanModelVector.StandardDeviation.Length);
            Assert.True(0.140807763f == floatCdfLogMeanModelVector.StandardDeviation[0]);
            Assert.True(0.9843767f == floatCdfLogMeanModelVector.StandardDeviation[3]);

            var doubleCdfLogMeanModel = ((NormalizingTransformer)transformers[3]).Columns[2].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<double>;
            Assert.Equal(-0.31062321927759518, doubleCdfLogMeanModel.Mean);
            Assert.True(doubleCdfLogMeanModel.UseLog);
            Assert.Equal(0.14080776721611871, doubleCdfLogMeanModel.StandardDeviation);

            var doubleCdfLogMeanModelVector = ((NormalizingTransformer)transformers[3]).Columns[3].ModelParameters as NormalizingTransformer.CdfNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleCdfLogMeanModelVector.Mean.Length);
            Assert.True(-0.31062321927759518 == doubleCdfLogMeanModelVector.Mean[0]);
            Assert.True(-1.0836203140680853 == doubleCdfLogMeanModelVector.Mean[3]);
            Assert.True(doubleCdfLogMeanModelVector.UseLog);
            Assert.Equal(4, doubleCdfLogMeanModelVector.StandardDeviation.Length);
            Assert.True(0.14080776721611871 == doubleCdfLogMeanModelVector.StandardDeviation[0]);
            Assert.True(0.98437679839698122 == doubleCdfLogMeanModelVector.StandardDeviation[3]);

            floatBinModel = ((NormalizingTransformer)transformers[4]).Columns[0].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<float>;
            Assert.True(4 == floatBinModel.UpperBounds.Length);
            Assert.True(0.6139241f == floatBinModel.UpperBounds[0]);
            Assert.True(float.PositiveInfinity == floatBinModel.UpperBounds[3]);
            Assert.True(3 == floatBinModel.Density);
            Assert.True(0 == floatBinModel.Offset);

            floatBinModelVector = ((NormalizingTransformer)transformers[4]).Columns[1].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<ImmutableArray<float>>;
            Assert.True(4 == floatBinModelVector.UpperBounds.Length);
            Assert.True(4 == floatBinModelVector.UpperBounds[0].Length);
            Assert.True(0.6139241f == floatBinModelVector.UpperBounds[0][0]);
            Assert.True(float.PositiveInfinity == floatBinModelVector.UpperBounds[0][floatBinModelVector.UpperBounds[0].Length - 1]);
            Assert.True(0.32f == floatBinModelVector.UpperBounds[3][0]);
            Assert.True(float.PositiveInfinity == floatBinModelVector.UpperBounds[3][floatBinModelVector.UpperBounds[3].Length - 1]);
            Assert.True(4 == floatBinModelVector.Density.Length);
            Assert.True(0 == floatBinModelVector.Offset.Length);

            doubleBinModel = ((NormalizingTransformer)transformers[4]).Columns[2].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<double>;
            Assert.Equal(4, doubleBinModel.UpperBounds.Length);
            Assert.True(0.61392405063291133 == doubleBinModel.UpperBounds[0]);
            Assert.True(float.PositiveInfinity == doubleBinModel.UpperBounds[3]);
            Assert.Equal(3, doubleBinModel.Density);
            Assert.Equal(0, doubleBinModel.Offset);

            doubleBinModelVector = ((NormalizingTransformer)transformers[4]).Columns[3].ModelParameters as NormalizingTransformer.BinNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleBinModelVector.UpperBounds[0].Length);
            Assert.True(0.6139240506329113335 == doubleBinModelVector.UpperBounds[0][0]);
            Assert.True(double.PositiveInfinity == doubleBinModelVector.UpperBounds[0][doubleBinModelVector.UpperBounds[0].Length - 1]);
            Assert.True(0.32 == doubleBinModelVector.UpperBounds[3][0]);
            Assert.True(double.PositiveInfinity == doubleBinModelVector.UpperBounds[3][doubleBinModelVector.UpperBounds[3].Length - 1]);
            Assert.Equal(4, doubleBinModelVector.Density.Length);
            Assert.Empty(doubleBinModelVector.Offset);

            // Robust scaler
            var robustScalerEstimator = context.Transforms.NormalizeRobustScaling(
                                new[] {new InputOutputColumnPair("float1rbs", "float1"), new InputOutputColumnPair("float4rbs", "float4"),
                                    new InputOutputColumnPair("double1rbs", "double1"), new InputOutputColumnPair("double4rbs", "double4")});

            var robustScalerTransformer = robustScalerEstimator.Fit(data);

            floatAffineModel = ((NormalizingTransformer)robustScalerTransformer).Columns[0].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<float>;
            Assert.Equal(1 / 1.8, floatAffineModel.Scale, 2);
            Assert.Equal(5.8f, floatAffineModel.Offset, 2);

            floatAffineModelVec = ((NormalizingTransformer)robustScalerTransformer).Columns[1].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<float>>;
            Assert.Equal(4, floatAffineModelVec.Scale.Length);
            Assert.Equal(.5555556, floatAffineModelVec.Scale[0], 2);
            Assert.Equal(.8333333, floatAffineModelVec.Scale[1], 2);
            Assert.Equal(.3389830, floatAffineModelVec.Scale[2], 2);
            Assert.Equal(.8333333, floatAffineModelVec.Scale[3], 2);

            Assert.Equal(5.8, floatAffineModelVec.Offset[0], 2);
            Assert.Equal(3, floatAffineModelVec.Offset[1], 2);
            Assert.Equal(4.4, floatAffineModelVec.Offset[2], 2);
            Assert.Equal(1.3, floatAffineModelVec.Offset[3], 2);

            doubleAffineModel = ((NormalizingTransformer)robustScalerTransformer).Columns[2].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<double>;
            Assert.Equal(1 / 1.8, doubleAffineModel.Scale, 2);
            Assert.Equal(5.8, doubleAffineModel.Offset, 2);

            doubleAffineModelVector = ((NormalizingTransformer)robustScalerTransformer).Columns[3].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleAffineModelVector.Scale.Length);
            Assert.Equal(.5555556, doubleAffineModelVector.Scale[0], 2);
            Assert.Equal(.8333333, doubleAffineModelVector.Scale[1], 2);
            Assert.Equal(.3389830, doubleAffineModelVector.Scale[2], 2);
            Assert.Equal(.8333333, doubleAffineModelVector.Scale[3], 2);

            Assert.Equal(5.8, doubleAffineModelVector.Offset[0], 2);
            Assert.Equal(3, doubleAffineModelVector.Offset[1], 2);
            Assert.Equal(4.4, doubleAffineModelVector.Offset[2], 2);
            Assert.Equal(1.3, doubleAffineModelVector.Offset[3], 2);

            // Robust scaler no offset
            robustScalerEstimator = context.Transforms.NormalizeRobustScaling(
                                new[] {new InputOutputColumnPair("float1rbs", "float1"), new InputOutputColumnPair("float4rbs", "float4"),
                                    new InputOutputColumnPair("double1rbs", "double1"), new InputOutputColumnPair("double4rbs", "double4")}
                                , centerData: false);

            robustScalerTransformer = robustScalerEstimator.Fit(data);

            floatAffineModel = ((NormalizingTransformer)robustScalerTransformer).Columns[0].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<float>;
            Assert.Equal(1 / 1.8, floatAffineModel.Scale, 2);
            Assert.Equal(0, floatAffineModel.Offset, 2);

            floatAffineModelVec = ((NormalizingTransformer)robustScalerTransformer).Columns[1].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<float>>;
            Assert.Equal(4, floatAffineModelVec.Scale.Length);
            Assert.Equal(.5555556, floatAffineModelVec.Scale[0], 2);
            Assert.Equal(.8333333, floatAffineModelVec.Scale[1], 2);
            Assert.Equal(.3389830, floatAffineModelVec.Scale[2], 2);
            Assert.Equal(.8333333, floatAffineModelVec.Scale[3], 2);

            Assert.Empty(floatAffineModelVec.Offset);

            doubleAffineModel = ((NormalizingTransformer)robustScalerTransformer).Columns[2].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<double>;
            Assert.Equal(1 / 1.8, doubleAffineModel.Scale, 2);
            Assert.Equal(0, doubleAffineModel.Offset, 2);

            doubleAffineModelVector = ((NormalizingTransformer)robustScalerTransformer).Columns[3].ModelParameters as NormalizingTransformer.AffineNormalizerModelParameters<ImmutableArray<double>>;
            Assert.Equal(4, doubleAffineModelVector.Scale.Length);
            Assert.Equal(.5555556, doubleAffineModelVector.Scale[0], 2);
            Assert.Equal(.8333333, doubleAffineModelVector.Scale[1], 2);
            Assert.Equal(.3389830, doubleAffineModelVector.Scale[2], 2);
            Assert.Equal(.8333333, doubleAffineModelVector.Scale[3], 2);

            Assert.Empty(doubleAffineModelVector.Offset);

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
            var est4 = ML.Transforms.NormalizeMinMax("float4", "float4");
            var est5 = ML.Transforms.NormalizeMinMax("float4");

            var data1 = est1.Fit(data).Transform(data);
            var data2 = est2.Fit(data).Transform(data);
            var data3 = est3.Fit(data).Transform(data);
            var data4 = est4.Fit(data).Transform(data);
            var data5 = est5.Fit(data).Transform(data);

            TestCommon.CheckSameSchemas(data1.Schema, data2.Schema);
            TestCommon.CheckSameSchemas(data1.Schema, data3.Schema);
            TestCommon.CheckSameSchemas(data1.Schema, data4.Schema);
            TestCommon.CheckSameSchemas(data1.Schema, data5.Schema);
            CheckSameValues(data1, data2);
            CheckSameValues(data1, data3);
            CheckSameValues(data1, data4);
            CheckSameValues(data1, data5);

            // Tests for MeanVariance
            var est6 = new NormalizingEstimator(Env, NormalizingEstimator.NormalizationMode.MeanVariance, ("float4", "float4"));
            var est7 = new NormalizingEstimator(Env, new NormalizingEstimator.MeanVarianceColumnOptions("float4"));
            var est8 = ML.Transforms.NormalizeMeanVariance("float4", "float4");

            var data6 = est6.Fit(data).Transform(data);
            var data7 = est7.Fit(data).Transform(data);
            var data8 = est8.Fit(data).Transform(data);
            TestCommon.CheckSameSchemas(data6.Schema, data7.Schema);
            TestCommon.CheckSameSchemas(data6.Schema, data8.Schema);
            CheckSameValues(data6, data7);
            CheckSameValues(data6, data8);

            // Tests for LogMeanVariance
            var est9 = new NormalizingEstimator(Env, NormalizingEstimator.NormalizationMode.LogMeanVariance, ("float4", "float4"));
            var est10 = new NormalizingEstimator(Env, new NormalizingEstimator.LogMeanVarianceColumnOptions("float4"));
            var est11 = ML.Transforms.NormalizeLogMeanVariance("float4", "float4");

            var data9 = est9.Fit(data).Transform(data);
            var data10 = est10.Fit(data).Transform(data);
            var data11 = est11.Fit(data).Transform(data);
            TestCommon.CheckSameSchemas(data9.Schema, data10.Schema);
            TestCommon.CheckSameSchemas(data9.Schema, data11.Schema);
            CheckSameValues(data9, data10);
            CheckSameValues(data9, data11);

            // Tests for Binning
            var est12 = new NormalizingEstimator(Env, NormalizingEstimator.NormalizationMode.Binning, ("float4", "float4"));
            var est13 = new NormalizingEstimator(Env, new NormalizingEstimator.BinningColumnOptions("float4"));
            var est14 = ML.Transforms.NormalizeBinning("float4", "float4");

            var data12 = est12.Fit(data).Transform(data);
            var data13 = est13.Fit(data).Transform(data);
            var data14 = est14.Fit(data).Transform(data);
            TestCommon.CheckSameSchemas(data12.Schema, data13.Schema);
            TestCommon.CheckSameSchemas(data12.Schema, data14.Schema);
            CheckSameValues(data12, data13);
            CheckSameValues(data12, data14);

            // Tests for SupervisedBinning
            var est15 = new NormalizingEstimator(Env, NormalizingEstimator.NormalizationMode.SupervisedBinning, ("float4", "float4"));
            var est16 = new NormalizingEstimator(Env, new NormalizingEstimator.SupervisedBinningColumOptions("float4"));
            var est17 = ML.Transforms.NormalizeSupervisedBinning("float4", "float4");

            var data15 = est15.Fit(data).Transform(data);
            var data16 = est16.Fit(data).Transform(data);
            var data17 = est17.Fit(data).Transform(data);
            TestCommon.CheckSameSchemas(data15.Schema, data16.Schema);
            TestCommon.CheckSameSchemas(data15.Schema, data17.Schema);
            CheckSameValues(data15, data16);
            CheckSameValues(data15, data17);

            // Tests for RobustScaler
            var est18 = new NormalizingEstimator(Env, NormalizingEstimator.NormalizationMode.RobustScaling, ("float4", "float4"));
            var est19 = new NormalizingEstimator(Env, new NormalizingEstimator.RobustScalingColumnOptions("float4"));
            var est20 = ML.Transforms.NormalizeRobustScaling("float4", "float4");

            var data18 = est18.Fit(data).Transform(data);
            var data19 = est19.Fit(data).Transform(data);
            var data20 = est20.Fit(data).Transform(data);
            TestCommon.CheckSameSchemas(data18.Schema, data19.Schema);
            TestCommon.CheckSameSchemas(data18.Schema, data20.Schema);
            CheckSameValues(data18, data19);
            CheckSameValues(data18, data20);

            Done();
        }

        [Fact]
        public void NormalizerExperimentalExtensions()
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

            // Normalizer Extensions
            var est1 = ML.Transforms.NormalizeMinMax("float4", "float4");
            var est2 = ML.Transforms.NormalizeMeanVariance("float4", "float4");
            var est3 = ML.Transforms.NormalizeLogMeanVariance("float4", "float4");
            var est4 = ML.Transforms.NormalizeBinning("float4", "float4");
            var est5 = ML.Transforms.NormalizeSupervisedBinning("float4", "float4");

            // Normalizer Extensions (Experimental)
            var est6 = ML.Transforms.NormalizeMinMax("float4", "float4");
            var est7 = ML.Transforms.NormalizeMeanVariance("float4", "float4");
            var est8 = ML.Transforms.NormalizeLogMeanVariance("float4", "float4");
            var est9 = ML.Transforms.NormalizeBinning("float4", "float4");
            var est10 = ML.Transforms.NormalizeSupervisedBinning("float4", "float4");

            // Fit and Transpose
            var data1 = est1.Fit(data).Transform(data);
            var data2 = est2.Fit(data).Transform(data);
            var data3 = est3.Fit(data).Transform(data);
            var data4 = est4.Fit(data).Transform(data);
            var data5 = est5.Fit(data).Transform(data);
            var data6 = est6.Fit(data).Transform(data);
            var data7 = est7.Fit(data).Transform(data);
            var data8 = est8.Fit(data).Transform(data);
            var data9 = est9.Fit(data).Transform(data);
            var data10 = est10.Fit(data).Transform(data);

            // Schema Checks
            TestCommon.CheckSameSchemas(data1.Schema, data6.Schema);
            TestCommon.CheckSameSchemas(data2.Schema, data7.Schema);
            TestCommon.CheckSameSchemas(data3.Schema, data8.Schema);
            TestCommon.CheckSameSchemas(data4.Schema, data9.Schema);
            TestCommon.CheckSameSchemas(data5.Schema, data10.Schema);

            // Value Checks
            CheckSameValues(data1, data6);
            CheckSameValues(data2, data7);
            CheckSameValues(data3, data8);
            CheckSameValues(data4, data9);
            CheckSameValues(data5, data10);

            Done();
        }

        [Fact]
        public void NormalizerExperimentalExtensionGetColumnPairs()
        {
            string dataPath = GetDataPath(TestDatasets.iris.trainFilename);

            var loader = new TextLoader(Env, new TextLoader.Options
            {
                Columns = new[] {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("input", DataKind.Single, new[]{new TextLoader.Range(1, 4) }),
                }
            });

            var data = loader.Load(dataPath);
            var est = ML.Transforms.NormalizeMinMax("output", "input");
            var t = est.Fit(data);

            Assert.Single(t.GetColumnPairs());
            Assert.Equal("output", t.GetColumnPairs()[0].OutputColumnName);
            Assert.Equal("input", t.GetColumnPairs()[0].InputColumnName);

            Done();
        }

        [NativeDependencyFact("MklImports")]
        public void LpGcNormAndWhiteningWorkout()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var data = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            var invalidData = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.String, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            var est = ML.Transforms.NormalizeLpNorm("lpnorm", "features")
                .Append(ML.Transforms.NormalizeGlobalContrast("gcnorm", "features"))
                .Append(new VectorWhiteningEstimator(ML, "whitened", "features"));
            TestEstimatorCore(est, data, invalidInput: invalidData);

            var outputPath = GetOutputPath("NormalizerEstimator", "lpnorm_gcnorm_whitened.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
                savedData = ML.Transforms.SelectColumns("lpnorm", "gcnorm", "whitened").Fit(savedData).Transform(savedData);

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("NormalizerEstimator", "lpnorm_gcnorm_whitened.tsv", digitsOfPrecision: 4);
            Done();
        }

        [NativeDependencyFact("MklImports")]
        public void WhiteningWorkout()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var data = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            var invalidData = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.String, 0, 10)
            }, hasHeader: true, separatorChar: ';');


            var est = new VectorWhiteningEstimator(ML, "whitened1", "features")
                .Append(new VectorWhiteningEstimator(ML, "whitened2", "features", kind: WhiteningKind.PrincipalComponentAnalysis, rank: 5));
            TestEstimatorCore(est, data, invalidInput: invalidData);

            var outputPath = GetOutputPath("NormalizerEstimator", "whitened.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
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

        [NativeDependencyFact("MklImports")]
        public void TestWhiteningOldSavingAndLoading()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var dataView = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

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
            var data = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            var invalidData = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.String, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            var est = ML.Transforms.NormalizeLpNorm("lpNorm1", "features")
                .Append(ML.Transforms.NormalizeLpNorm("lpNorm2", "features", norm: LpNormNormalizingEstimatorBase.NormFunction.L1, ensureZeroMean: true));
            TestEstimatorCore(est, data, invalidInput: invalidData);

            var outputPath = GetOutputPath("NormalizerEstimator", "lpNorm.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
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
            var dataView = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

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

        [Fact]
        public void GcnWorkout()
        {
            string dataSource = GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename);
            var data = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            var invalidData = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.String, 0, 10)
            }, hasHeader: true, separatorChar: ';');

            var est = ML.Transforms.NormalizeGlobalContrast("gcnNorm1", "features")
                .Append(ML.Transforms.NormalizeGlobalContrast("gcnNorm2", "features", ensureZeroMean: false, ensureUnitStandardDeviation: true, scale: 3));
            TestEstimatorCore(est, data, invalidInput: invalidData);

            var outputPath = GetOutputPath("NormalizerEstimator", "gcnNorm.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
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
            var dataView = ML.Data.LoadFromTextFile(dataSource, new[] {
                new TextLoader.Column("label", DataKind.Single, 11),
                new TextLoader.Column("features", DataKind.Single, 0, 10)
            }, hasHeader: true, separatorChar: ';');

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
        public void TestNormalizeBackCompatibility()
        {
            var dataFile = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var dataView = TextLoader.Create(ML, new TextLoader.Options(), new MultiFileSource(dataFile));
            string chooseModelPath = GetDataPath("backcompat/ap_with_norm.zip");
            using (FileStream fs = File.OpenRead(chooseModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, dataView, fs);
                Assert.Equal(3, result.Schema.Count);
            }
        }

        private sealed class DataPointVec
        {
            [VectorType(5)]
            public float[] Features { get; set; }
        }

        private sealed class DataPointOne
        {
            public float Features { get; set; }
        }

        [Fact]
        public void TestNormalizeLogMeanVarianceFixZeroOne()
        {
            var samples = new List<DataPointOne>()
            {
                new DataPointOne(){ Features = 1f },
                new DataPointOne(){ Features = 2f },
                new DataPointOne(){ Features = 0f },
                new DataPointOne(){ Features = -1 }
            };
            // Convert training data to IDataView, the general data type used in ML.NET.
            var data = ML.Data.LoadFromEnumerable(samples);
            // NormalizeLogMeanVariance normalizes the data based on the computed mean and variance of the logarithm of the data.
            // Uses Cumulative distribution function as output.
            var normalize = ML.Transforms.NormalizeLogMeanVariance("Features", true, useCdf: true);

            // NormalizeLogMeanVariance normalizes the data based on the computed mean and variance of the logarithm of the data.
            var normalizeNoCdf = ML.Transforms.NormalizeLogMeanVariance("Features", true, useCdf: false);

            // Now we can transform the data and look at the output to confirm the behavior of the estimator.
            var normalizeTransform = normalize.Fit(data);
            var transformedData = normalizeTransform.Transform(data);
            var normalizeNoCdfTransform = normalizeNoCdf.Fit(data);
            var noCdfData = normalizeNoCdfTransform.Transform(data);

            var transformParams = normalizeTransform.GetNormalizerModelParameters(0) as CdfNormalizerModelParameters<float>;
            var noCdfParams = normalizeNoCdfTransform.GetNormalizerModelParameters(0) as AffineNormalizerModelParameters<float>;

            // Standard deviation and offset should not be zero for the given data even when FixZero is set to true.
            Assert.NotEqual(0f, transformParams.Mean);
            Assert.NotEqual(0f, transformParams.StandardDeviation);

            // Offset should be zero when FixZero is set to true but not the scale (on this data).
            Assert.Equal(0f, noCdfParams.Offset);
            Assert.NotEqual(0f, noCdfParams.Scale);

            var transformedDataArray = ML.Data.CreateEnumerable<DataPointOne>(noCdfData, false).ToImmutableArray();
            // Without the Cdf and fixing zero, any 0 should stay 0.
            Assert.Equal(0f, transformedDataArray[2].Features);
        }

        [Fact]
        public void TestNormalizeLogMeanVarianceFixZeroVec()
        {
            var samples = new List<DataPointVec>()
            {
                new DataPointVec(){ Features = new float[5] { 1, 1, 3, 0, float.MaxValue } },
                new DataPointVec(){ Features = new float[5] { 2, 2, 2, 0, float.MinValue } },
                new DataPointVec(){ Features = new float[5] { 0, 0, 1, 0.5f, 0} },
                new DataPointVec(){ Features = new float[5] {-1,-1,-1, 1, 1} }
            };
            // Convert training data to IDataView, the general data type used in ML.NET.
            var data = ML.Data.LoadFromEnumerable(samples);
            // NormalizeLogMeanVariance normalizes the data based on the computed mean and variance of the logarithm of the data.
            // Uses Cumulative distribution function as output.
            var normalize = ML.Transforms.NormalizeLogMeanVariance("Features", true, useCdf: true);

            // NormalizeLogMeanVariance normalizes the data based on the computed mean and variance of the logarithm of the data.
            var normalizeNoCdf = ML.Transforms.NormalizeLogMeanVariance("Features", true, useCdf: false);

            // Now we can transform the data and look at the output to confirm the behavior of the estimator.
            var normalizeTransform = normalize.Fit(data);
            var transformedData = normalizeTransform.Transform(data);
            var normalizeNoCdfTransform = normalizeNoCdf.Fit(data);
            var noCdfData = normalizeNoCdfTransform.Transform(data);

            var transformParams = normalizeTransform.GetNormalizerModelParameters(0) as CdfNormalizerModelParameters<ImmutableArray<float>>;
            var noCdfParams = normalizeNoCdfTransform.GetNormalizerModelParameters(0) as AffineNormalizerModelParameters<ImmutableArray<float>>;

            for (int i = 0; i < 5; i++)
            {
                // Standard deviation and offset should not be zero for the given data even when FixZero is set to true.
                Assert.NotEqual(0f, transformParams.Mean[i]);
                Assert.NotEqual(0f, transformParams.StandardDeviation[i]);

                // Offset should be zero when FixZero is set to true but not the scale (on this data).
                Assert.Empty(noCdfParams.Offset);
                Assert.NotEqual(0f, noCdfParams.Scale[i]);
            }

            var transformedDataArray = ML.Data.CreateEnumerable<DataPointVec>(noCdfData, false).ToImmutableArray();
            // Without the Cdf and fixing zero, any 0 should stay 0.
            Assert.Equal(0f, transformedDataArray[0].Features[3]);
            Assert.Equal(0f, transformedDataArray[1].Features[3]);
            Assert.Equal(0f, transformedDataArray[2].Features[0]);
            Assert.Equal(0f, transformedDataArray[2].Features[1]);
            Assert.Equal(0f, transformedDataArray[2].Features[4]);
        }

        [Fact]
        public void TestNormalizeBackCompatibility2()
        {
            // Tests backward compatibility with a normalizing transformer
            // whose version is "verWrittenCur: 0x00010001"

            string dataPath = GetDataPath(TestDatasets.iris.trainFilename);

            var loader = new TextLoader(Env, new TextLoader.Options
            {
                Columns = new[] {
                    new TextLoader.Column("float1", DataKind.Single, 1),
                    new TextLoader.Column("float4", DataKind.Single, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("double1", DataKind.Double, 1),
                    new TextLoader.Column("double4", DataKind.Double, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("int1", DataKind.Int32, 0),
                },
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var data = loader.Load(dataPath);

            var modelPath = Path.Combine("TestModels", "normalizer_verwrit-00010001.zip");
            var normalizer = ML.Model.Load(modelPath, out var schema);

            var outputPath = GetOutputPath("NormalizerEstimator", "normalized2.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                using (var fs = File.Create(outputPath))
                {
                    var transformedData = normalizer.Transform(data);
                    DataSaverUtils.SaveDataView(ch, saver, transformedData, fs, keepHidden: true);
                }
            }

            CheckEquality("NormalizerEstimator", "normalized2.tsv", "normalizedBackwardsCompat.tsv");

            Done();
        }

        public class TensorData
        {
            private const int Dim1 = 2;
            private const int Dim2 = 3;
            private const int Dim3 = 4;
            private const int Size = Dim1 * Dim2 * Dim3;

            [VectorType(Dim1, Dim2, Dim3)]
            public float[] input { get; set; }

            public static TensorData[] GetTensorData()
            {
                var tensor1 = Enumerable.Range(0, Size).Select(
                x => (float)x).ToArray();

                var tensor2 = Enumerable.Range(0, Size).Select(
                x => (float)(x + 10000)).ToArray();

                return new TensorData[]
                {
                    new TensorData() { input = tensor1},
                    new TensorData() { input = tensor2}
                };
            }
        }

        [Fact]
        public void TestSavingNormalizerWithMultidimensionalVectorInput()
        {
            var samples = TensorData.GetTensorData();
            var data = ML.Data.LoadFromEnumerable(samples);
            var model = ML.Transforms.NormalizeMinMax("output", "input").Fit(data);
            var transformedData = model.Transform(data);

            var modelAndSchemaPath = GetOutputPath("TestSavingNormalizerWithMultidimensionalVectorInput.zip");
            ML.Model.Save(model, data.Schema, modelAndSchemaPath);
            var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
            var transformedData2 = loadedModel.Transform(data);

            var dimensions1 = (transformedData.Schema["output"].Type as VectorDataViewType).Dimensions;
            var dimensions2 = (transformedData2.Schema["output"].Type as VectorDataViewType).Dimensions;

            Assert.True(dimensions1.SequenceEqual(dimensions2));
        }

        [Fact]
        public void TestHeapMedianAlgorithm()
        {
            // Generate 100,000 random numbers
            var numberOfItems = 100000;
            var numbers = GenerateRandomFloats(numberOfItems);

            // Allocate memory for median ahead of time. 
            float linqMedian = default;
            float heapMedian = default;

            // Find the median using LINQ so we can compare it to the heap approach.
            int numberCount = numbers.Count();
            int halfIndex = numbers.Count() / 2;
            var sortedNumbers = numbers.OrderBy(n => n);

            if ((numberCount % 2) == 0)
            {
                linqMedian = ((sortedNumbers.ElementAt(halfIndex) +
                    sortedNumbers.ElementAt((halfIndex - 1))) / 2);
            }
            else
            {
                linqMedian = sortedNumbers.ElementAt(halfIndex);
            }

            // Find the median using the heap approach.
            // Create the heaps
            var minHeap = new MedianAggregatorUtils.MinHeap<float>((numberOfItems / 2) + 1);
            var maxHeap = new MedianAggregatorUtils.MaxHeap<float>((numberOfItems / 2) + 1);

            foreach (var num in numbers)
            {
                MedianAggregatorUtils.GetMedianSoFar(num, ref heapMedian, ref maxHeap, ref minHeap);
            }

            // Compare the medians, they should be equal.
            Assert.Equal(linqMedian, heapMedian);
        }

        static List<float> GenerateRandomFloats(int num, int min = int.MinValue, int max = int.MaxValue, int seed = 0)
        {
            var rand = new Random(seed);

            var list = new List<float>(num);

            for (int i = 0; i < num; i++)
            {
                list.Add(rand.Next(min, max));
            }

            return list;
        }

        [Fact]
        public void TestMinHeapForMedianNormalizer()
        {
            // Simple test with all values in order.
            MedianAggregatorUtils.MinHeap<float> heap = new MedianAggregatorUtils.MinHeap<float>(10);
            heap.Add(-1);
            heap.Add(-2);
            heap.Add(-3);
            heap.Add(-4);
            heap.Add(-5);

            var min = heap.Peek();
            Assert.Equal(-5, min);

            // Test with duplicate values.
            heap = new MedianAggregatorUtils.MinHeap<float>(10);
            heap.Add(-5);
            heap.Add(-2);
            heap.Add(-3);
            heap.Add(-4);
            heap.Add(-5);

            min = heap.Peek();
            Assert.Equal(-5, min);

            // Test with values in reverse order.
            heap = new MedianAggregatorUtils.MinHeap<float>(10);
            heap.Add(-5);
            heap.Add(-4);
            heap.Add(-3);
            heap.Add(-2);
            heap.Add(-1);

            min = heap.Peek();
            Assert.Equal(-5, min);

            // Test with repeated duplicated values.
            heap = new MedianAggregatorUtils.MinHeap<float>(10);
            heap.Add(-5);
            heap.Add(-5);
            heap.Add(-5);
            heap.Add(-5);
            heap.Add(-5);
            min = heap.Peek();
            Assert.Equal(-5, min);

            // Test with positive and negative numbers.
            heap = new MedianAggregatorUtils.MinHeap<float>(10);
            heap.Add(1);
            heap.Add(-2);
            heap.Add(-10);
            heap.Add(-4);
            heap.Add(10);

            min = heap.Peek();
            Assert.Equal(-10, min);

            // Large heap test to make sure correct min is chosen.
            heap = new MedianAggregatorUtils.MinHeap<float>(10000);
            Random rand = new Random(0);
            min = float.MaxValue;
            float temp = default;

            for (int i = 0; i < 10000; i++)
            {
                temp = rand.Next(int.MinValue, int.MaxValue);
                min = temp < min ? temp : min;
                heap.Add(temp);
            }

            Assert.Equal(min, heap.Peek());
        }

        [Fact]
        public void TestMaxHeapForMedianNormalizer()
        {
            // Simple test with all values in order.
            MedianAggregatorUtils.MaxHeap<float> heap = new MedianAggregatorUtils.MaxHeap<float>(10);
            heap.Add(1);
            heap.Add(2);
            heap.Add(3);
            heap.Add(4);
            heap.Add(5);

            var max = heap.Peek();
            Assert.Equal(5, max);

            // Test with duplicate values.
            heap = new MedianAggregatorUtils.MaxHeap<float>(10);
            heap.Add(5);
            heap.Add(2);
            heap.Add(3);
            heap.Add(4);
            heap.Add(5);

            max = heap.Peek();
            Assert.Equal(5, max);

            // Test with values in reverse order.
            heap = new MedianAggregatorUtils.MaxHeap<float>(10);
            heap.Add(5);
            heap.Add(4);
            heap.Add(3);
            heap.Add(2);
            heap.Add(1);

            max = heap.Peek();
            Assert.Equal(5, max);

            // Test with repeated duplicated values.
            heap = new MedianAggregatorUtils.MaxHeap<float>(10);
            heap.Add(5);
            heap.Add(5);
            heap.Add(5);
            heap.Add(5);
            heap.Add(5);

            max = heap.Peek();
            Assert.Equal(5, max);

            // Test with positive and negative numbers.
            heap = new MedianAggregatorUtils.MaxHeap<float>(10);
            heap.Add(-1);
            heap.Add(2);
            heap.Add(10);
            heap.Add(4);
            heap.Add(-10);

            max = heap.Peek();
            Assert.Equal(10, max);

            // Large heap test to make sure correct min is chosen.
            heap = new MedianAggregatorUtils.MaxHeap<float>(10000);
            Random rand = new Random(0);
            max = float.MinValue;
            float temp = default;

            for (int i = 0; i < 10000; i++)
            {
                temp = rand.Next(int.MinValue, int.MaxValue);
                max = temp > max ? temp : max;
                heap.Add(temp);
            }

            Assert.Equal(max, heap.Peek());
        }
    }
}
