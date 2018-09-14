// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class NAReplaceTests : TestDataPipeBase
    {
        private class TestClass
        {
            public float A;
            public string B;
            public double C;
            [VectorType(2)]
            public float[] D;
            [VectorType(2)]
            public double[] E;
        }

        public NAReplaceTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void NAReplaceWorkout()
        {
            var data = new[] {
                new TestClass() { A = 1, B = "A", C = 3, D= new float[2]{ 1, 2 } , E = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN, B = null, C = double.NaN, D= new float[2]{ float.NaN, float.NaN } , E = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = null, C = double.NegativeInfinity,D= new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , E = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity, B = null, C = double.PositiveInfinity,D= new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , E = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = "B", C = 1 ,D= new float[2]{ 3, 4 } , E = new double[2]{ 5,6}},
            };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new NAReplaceEstimator(Env,
                new NAReplaceTransform.ColumnInfo("A", "NAA", NAReplaceTransform.ReplacementKind.Mean),
                new NAReplaceTransform.ColumnInfo("B", "NAB", NAReplaceTransform.ReplacementKind.Default),
                new NAReplaceTransform.ColumnInfo("C", "NAC", NAReplaceTransform.ReplacementKind.Mean),
                new NAReplaceTransform.ColumnInfo("D", "NAD", NAReplaceTransform.ReplacementKind.Mean),
                new NAReplaceTransform.ColumnInfo("E", "NAE", NAReplaceTransform.ReplacementKind.Mean));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestCommandLine()
        {
            using (var env = new TlcEnvironment())
            {
                Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0}  xf=NAReplace{col=C:A} in=f:\2.txt" }), (int)0);
            }
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] {
                new TestClass() { A = 1, B = "A", C = 3, D= new float[2]{ 1, 2 } , E = new double[2]{ 3,4} },
                new TestClass() { A = float.NaN, B = null, C = double.NaN, D= new float[2]{ float.NaN, float.NaN } , E = new double[2]{ double.NaN,double.NaN}},
                new TestClass() { A = float.NegativeInfinity, B = null, C = double.NegativeInfinity,D= new float[2]{ float.NegativeInfinity, float.NegativeInfinity } , E = new double[2]{ double.NegativeInfinity, double.NegativeInfinity}},
                new TestClass() { A = float.PositiveInfinity, B = null, C = double.PositiveInfinity,D= new float[2]{ float.PositiveInfinity, float.PositiveInfinity, } , E = new double[2]{  double.PositiveInfinity, double.PositiveInfinity}},
                new TestClass() { A = 2, B = "B", C = 1 ,D= new float[2]{ 3, 4 } , E = new double[2]{ 5,6}},
            };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new NAReplaceEstimator(Env,
                new NAReplaceTransform.ColumnInfo("A", "NAA", NAReplaceTransform.ReplacementKind.Mean),
                new NAReplaceTransform.ColumnInfo("B", "NAB", NAReplaceTransform.ReplacementKind.Default),
                new NAReplaceTransform.ColumnInfo("C", "NAC", NAReplaceTransform.ReplacementKind.Mean),
                new NAReplaceTransform.ColumnInfo("D", "NAD", NAReplaceTransform.ReplacementKind.Mean),
                new NAReplaceTransform.ColumnInfo("E", "NAE", NAReplaceTransform.ReplacementKind.Mean));

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
            }
        }
    }
}
