// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.StaticPipe;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;
using System;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class KeyToBinaryVectorEstimatorTest : TestDataPipeBase
    {
        public KeyToBinaryVectorEstimatorTest(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public int A;
            public int B;
            public int C;
        }

        private class TestMeta
        {
            [VectorType(2)]
            public string[] A;
            public string B;
            [VectorType(2)]
            public int[] C;
            public int D;
        }

        [Fact]
        public void KeyToBinaryVectorWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            dataView = new ValueToKeyMappingEstimator(Env, new[]{
                    new TermTransform.ColumnInfo("A", "TermA"),
                    new TermTransform.ColumnInfo("B", "TermB"),
                    new TermTransform.ColumnInfo("C", "TermC", textKeyValues:true)
                }).Fit(dataView).Transform(dataView);

            var pipe = new KeyToBinaryVectorMappingEstimator(Env, new KeyToBinaryVectorTransform.ColumnInfo("TermA", "CatA"),
                new KeyToBinaryVectorTransform.ColumnInfo("TermC", "CatC"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void KeyToBinaryVectorStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(Env, ctx => (
                ScalarString: ctx.LoadText(1),
                VectorString: ctx.LoadText(1, 4)
            ));

            var data = reader.Read(dataPath);

            // Non-pigsty Term.
            var dynamicData = new ValueToKeyMappingEstimator(Env, new[] {
                new TermTransform.ColumnInfo("ScalarString", "A"),
                new TermTransform.ColumnInfo("VectorString", "B") })
                .Fit(data.AsDynamic).Transform(data.AsDynamic);

            var data2 = dynamicData.AssertStatic(Env, ctx => (
                A: ctx.KeyU4.TextValues.Scalar,
                B: ctx.KeyU4.TextValues.Vector));

            var est = data2.MakeNewEstimator()
                .Append(row => (
                ScalarString: row.A.ToBinaryVector(),
                VectorString: row.B.ToBinaryVector()));

            TestEstimatorCore(est.AsDynamic, data2.AsDynamic, invalidInput: data.AsDynamic);

            Done();
        }

        [Fact]
        public void TestMetadataPropagation()
        {
            var data = new[] {
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 5,3}, D= 1},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6} };


            var dataView = ComponentCreation.CreateDataView(Env, data);
            var termEst = new ValueToKeyMappingEstimator(Env, new[] {
                new TermTransform.ColumnInfo("A", "TA", textKeyValues: true),
                new TermTransform.ColumnInfo("B", "TB", textKeyValues: true),
                new TermTransform.ColumnInfo("C", "TC"),
                new TermTransform.ColumnInfo("D", "TD") });
            var termTransformer = termEst.Fit(dataView);
            dataView = termTransformer.Transform(dataView);

            var pipe = new KeyToBinaryVectorMappingEstimator(Env,
                 new KeyToBinaryVectorTransform.ColumnInfo("TA", "CatA"),
                 new KeyToBinaryVectorTransform.ColumnInfo("TB", "CatB"),
                 new KeyToBinaryVectorTransform.ColumnInfo("TC", "CatC"),
                 new KeyToBinaryVectorTransform.ColumnInfo("TD", "CatD"));

            var result = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(result);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {
            Assert.True(result.Schema.TryGetColumnIndex("CatA", out int colA));
            Assert.True(result.Schema.TryGetColumnIndex("CatB", out int colB));
            Assert.True(result.Schema.TryGetColumnIndex("CatC", out int colC));
            Assert.True(result.Schema.TryGetColumnIndex("CatD", out int colD));
            var types = result.Schema.GetMetadataTypes(colA);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            VBuffer<ReadOnlyMemory<char>> slots = default;
            bool normalized = default;
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colA, ref slots);
            Assert.True(slots.Length == 6);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit2", "[1].Bit1", "[1].Bit0" });

            types = result.Schema.GetMetadataTypes(colB);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colB, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "Bit1", "Bit0" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colB, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colC);
            Assert.Equal(types.Select(x => x.Key), new string[0]);

            types = result.Schema.GetMetadataTypes(colD);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colD, ref normalized);
            Assert.True(normalized);
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(0, Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Term{col=B:A} xf=KeyToBinary{col=C:B} in=f:\2.txt" }));
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var est = new ValueToKeyMappingEstimator(Env, new[]{
                    new TermTransform.ColumnInfo("A", "TermA"),
                    new TermTransform.ColumnInfo("B", "TermB", textKeyValues:true),
                    new TermTransform.ColumnInfo("C", "TermC")
            });
            var transformer = est.Fit(dataView);
            dataView = transformer.Transform(dataView);
            var pipe = new KeyToBinaryVectorMappingEstimator(Env,
                new KeyToBinaryVectorTransform.ColumnInfo("TermA", "CatA"),
                new KeyToBinaryVectorTransform.ColumnInfo("TermB", "CatB"),
                new KeyToBinaryVectorTransform.ColumnInfo("TermC", "CatC")
            );
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
