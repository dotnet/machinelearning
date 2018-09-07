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
    public class KeyToVectorEstimatorTest : TestDataPipeBase
    {
        public KeyToVectorEstimatorTest(ITestOutputHelper output) : base(output)
        {
        }
        class TestClass
        {
            public int A;
            public int B;
            public int C;
        }
        class TestMeta
        {
            [VectorType(2)]
            public string[] A;
            public string B;
            [VectorType(2)]
            public int[] C;
            public int D;
            [VectorType(2)]
            public float[] E;
            public float F;
        }

        [Fact]
        public void KeyToVectorWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            dataView = new TermEstimator(Env, new[]{
                    new TermTransform.ColumnInfo("A", "TermA"),
                    new TermTransform.ColumnInfo("B", "TermB"),
                    new TermTransform.ColumnInfo("C", "TermC", textKeyValues:true)
                }).Fit(dataView).Transform(dataView);

            var pipe = new KeyToVectorEstimator(Env, new KeyToVectorTransform.ColumnInfo("TermA", "CatA", false),
                new KeyToVectorTransform.ColumnInfo("TermB", "CatB", true));
            TestEstimatorCore(pipe, dataView);
        }

        [Fact]
        void TestMetadataCopy()
        {
            var data = new[] { new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6, E=new float[2] { 2.0f,4.0f}, F = 1.0f },
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 5,3}, D= 1, E=new float[2] { 4.0f,2.0f}, F = -1.0f },
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6, E=new float[2] { 2.0f,4.0f}, F = 1.0f } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var termEst = new TermEstimator(Env,
                new TermTransform.ColumnInfo("A", "TA"),
                new TermTransform.ColumnInfo("B", "TB"),
                new TermTransform.ColumnInfo("C", "TC"),
                new TermTransform.ColumnInfo("D", "TD"),
                new TermTransform.ColumnInfo("E", "TE"),
                new TermTransform.ColumnInfo("F", "TF"));
            var termTransformer = termEst.Fit(dataView);
            dataView = termTransformer.Transform(dataView);

            var pipe = new KeyToVectorEstimator(Env,
                 new KeyToVectorTransform.ColumnInfo("TA", "CatA", false),
                 new KeyToVectorTransform.ColumnInfo("TB", "CatB", false),
                 new KeyToVectorTransform.ColumnInfo("TC", "CatC", true),
                 new KeyToVectorTransform.ColumnInfo("TD", "CatD", false),
                 new KeyToVectorTransform.ColumnInfo("TE", "CatE", false),
                 new KeyToVectorTransform.ColumnInfo("TF", "CatF", true)
                 );

            var result = pipe.Fit(dataView).Transform(dataView);
        }


        [Fact]
        void TestCommandLine()
        {
            using (var env = new TlcEnvironment())
            {
                Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Term{col=B:A} xf=KeyToVector{col=C:B} in=f:\2.txt" }), (int)0);
            }
        }

        [Fact]
        void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = new TermEstimator(Env, new[]{
                    new TermTransform.ColumnInfo("A", "TermA"),
                    new TermTransform.ColumnInfo("B", "TermB"),
                    new TermTransform.ColumnInfo("C", "TermC")
                });
            var transformer = est.Fit(dataView);
            dataView = transformer.Transform(dataView);
            var pipe = new KeyToVectorEstimator(Env, new KeyToVectorTransform.ColumnInfo("TermA", "CatA", false),
    new KeyToVectorTransform.ColumnInfo("TermB", "CatB", true));
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
