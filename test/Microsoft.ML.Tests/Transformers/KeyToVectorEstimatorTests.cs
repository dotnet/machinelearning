// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class KeyToVectorEstimatorTest : TestDataPipeBase
    {
        public KeyToVectorEstimatorTest(ITestOutputHelper output) : base(output)
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
            [VectorType(2)]
            public float[] E;
            public float F;
            [VectorType(2)]
            public string[] G;
            public string H;
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
                new KeyToVectorTransform.ColumnInfo("TermB", "CatB", true),
                new KeyToVectorTransform.ColumnInfo("TermC", "CatC", true),
                new KeyToVectorTransform.ColumnInfo("TermC", "CatCNonBag", false));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestMetadataPropagation()
        {
            var data = new[] {
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6, E= new float[2] { 1.0f,2.0f}, F = 1.0f , G= new string[2]{ "A","D"}, H="D"},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 5,3}, D= 1, E=new float[2] { 3.0f,4.0f}, F = -1.0f ,G= new string[2]{"E", "A"}, H="E"},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6, E=new float[2] { 5.0f,6.0f}, F = 1.0f ,G= new string[2]{ "D", "E"}, H="D"} };


            var dataView = ComponentCreation.CreateDataView(Env, data);
            var termEst = new TermEstimator(Env,
                new TermTransform.ColumnInfo("A", "TA", textKeyValues: true),
                new TermTransform.ColumnInfo("B", "TB"),
                new TermTransform.ColumnInfo("C", "TC", textKeyValues: true),
                new TermTransform.ColumnInfo("D", "TD", textKeyValues: true),
                new TermTransform.ColumnInfo("E", "TE"),
                new TermTransform.ColumnInfo("F", "TF"),
                new TermTransform.ColumnInfo("G", "TG"),
                new TermTransform.ColumnInfo("H", "TH", textKeyValues: true));
            var termTransformer = termEst.Fit(dataView);
            dataView = termTransformer.Transform(dataView);

            var pipe = new KeyToVectorEstimator(Env,
                 new KeyToVectorTransform.ColumnInfo("TA", "CatA", true),
                 new KeyToVectorTransform.ColumnInfo("TB", "CatB", false),
                 new KeyToVectorTransform.ColumnInfo("TC", "CatC", false),
                 new KeyToVectorTransform.ColumnInfo("TD", "CatD", true),
                 new KeyToVectorTransform.ColumnInfo("TE", "CatE", false),
                 new KeyToVectorTransform.ColumnInfo("TF", "CatF", true),
                 new KeyToVectorTransform.ColumnInfo("TG", "CatG", true),
                 new KeyToVectorTransform.ColumnInfo("TH", "CatH", false)
                 );

            var result = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(result);
        }

        private void ValidateMetadata(IDataView result)
        {
            Assert.True(result.Schema.TryGetColumnIndex("CatA", out int colA));
            Assert.True(result.Schema.TryGetColumnIndex("CatB", out int colB));
            Assert.True(result.Schema.TryGetColumnIndex("CatC", out int colC));
            Assert.True(result.Schema.TryGetColumnIndex("CatD", out int colD));
            Assert.True(result.Schema.TryGetColumnIndex("CatE", out int colE));
            Assert.True(result.Schema.TryGetColumnIndex("CatF", out int colF));
            Assert.True(result.Schema.TryGetColumnIndex("CatE", out int colG));
            Assert.True(result.Schema.TryGetColumnIndex("CatF", out int colH));
            var types = result.Schema.GetMetadataTypes(colA);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            VBuffer<DvText> slots = default;
            VBuffer<DvInt4> slotRanges = default;
            DvBool normalized = default;
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colA, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Values.Select(x => x.ToString()), new string[2] { "A", "B" });

            types = result.Schema.GetMetadataTypes(colB);
            Assert.Equal(types.Select(x => x.Key), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colB, ref slots);
            Assert.True(slots.Length == 1);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colB, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.RawValue), new int[2] { 0, 0 });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colB, ref normalized);
            Assert.True(normalized.IsTrue);

            types = result.Schema.GetMetadataTypes(colC);
            Assert.Equal(types.Select(x => x.Key), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colC, ref slots);
            Assert.True(slots.Length == 4);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[4] { "[0].3", "[0].5", "[1].3", "[1].5" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colC, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.RawValue), new int[4] { 0, 1, 2, 3 });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colC, ref normalized);
            Assert.True(normalized.IsTrue);

            types = result.Schema.GetMetadataTypes(colD);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colD, ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "6", "1" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colD, ref normalized);
            Assert.True(normalized.IsTrue);


            types = result.Schema.GetMetadataTypes(colE);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colE, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.RawValue), new int[4] { 0, 5, 6, 11 });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colE, ref normalized);
            Assert.True(normalized.IsTrue);

            types = result.Schema.GetMetadataTypes(colF);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colF, ref normalized);
            Assert.True(normalized.IsTrue);

            types = result.Schema.GetMetadataTypes(colG);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colG, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.RawValue), new int[4] { 0, 5, 6, 11 });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colF, ref normalized);
            Assert.True(normalized.IsTrue);

            types = result.Schema.GetMetadataTypes(colH);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colF, ref normalized);
            Assert.True(normalized.IsTrue);
        }

        [Fact]
        public void TestCommandLine()
        {
            using (var env = new TlcEnvironment())
            {
                Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Term{col=B:A} xf=KeyToVector{col=C:B col={name=D source=B bag+}} in=f:\2.txt" }), (int)0);
            }
        }

        [Fact]
        public void TestOldSavingAndLoading()
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
            var pipe = new KeyToVectorEstimator(Env,
                new KeyToVectorTransform.ColumnInfo("TermA", "CatA", false),
                new KeyToVectorTransform.ColumnInfo("TermB", "CatB", true)
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
