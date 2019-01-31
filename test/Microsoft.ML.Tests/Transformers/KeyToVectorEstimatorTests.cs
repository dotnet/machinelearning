// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms.Conversions;
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

            var dataView = ML.Data.ReadFromEnumerable(data);
            dataView = new ValueToKeyMappingEstimator(Env, new[]{
                    new ValueToKeyMappingTransformer.ColumnInfo("A", "TermA"),
                    new ValueToKeyMappingTransformer.ColumnInfo("B", "TermB"),
                    new ValueToKeyMappingTransformer.ColumnInfo("C", "TermC", textKeyValues:true)
                }).Fit(dataView).Transform(dataView);

            var pipe = new KeyToVectorMappingEstimator(Env, new KeyToVectorMappingTransformer.ColumnInfo("TermA", "CatA", false),
                new KeyToVectorMappingTransformer.ColumnInfo("TermB", "CatB", true),
                new KeyToVectorMappingTransformer.ColumnInfo("TermC", "CatC", true),
                new KeyToVectorMappingTransformer.ColumnInfo("TermC", "CatCNonBag", false));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void KeyToVectorStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateReader(Env, ctx => (
                ScalarString: ctx.LoadText(1),
                VectorString: ctx.LoadText(1, 4)
            ));

            var data = reader.Read(dataPath);

            // Non-pigsty Term.
            var dynamicData = new ValueToKeyMappingEstimator(Env, new[] {
                new ValueToKeyMappingTransformer.ColumnInfo("ScalarString", "A"),
                new ValueToKeyMappingTransformer.ColumnInfo("VectorString", "B") })
                .Fit(data.AsDynamic).Transform(data.AsDynamic);

            var data2 = dynamicData.AssertStatic(Env, ctx => (
                A: ctx.KeyU4.TextValues.Scalar,
                B: ctx.KeyU4.TextValues.Vector));

            var est = data2.MakeNewEstimator()
                .Append(row => (
                ScalarString: row.A.ToVector(),
                VectorString: row.B.ToVector(),
                VectorBaggedString: row.B.ToBaggedVector()
                ));

            TestEstimatorCore(est.AsDynamic, data2.AsDynamic, invalidInput: data.AsDynamic);

            Done();
        }

        [Fact]
        public void TestMetadataPropagation()
        {
            var data = new[] {
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6, E= new float[2] { 1.0f,2.0f}, F = 1.0f , G= new string[2]{ "A","D"}, H="D"},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 5,3}, D= 1, E=new float[2] { 3.0f,4.0f}, F = -1.0f ,G= new string[2]{"E", "A"}, H="E"},
                new TestMeta() { A=new string[2] { "A", "B"}, B="C", C=new int[2] { 3,5}, D= 6, E=new float[2] { 5.0f,6.0f}, F = 1.0f ,G= new string[2]{ "D", "E"}, H="D"} };


            var dataView = ML.Data.ReadFromEnumerable(data);
            var termEst = new ValueToKeyMappingEstimator(Env, new[] {
                new ValueToKeyMappingTransformer.ColumnInfo("A", "TA", textKeyValues: true),
                new ValueToKeyMappingTransformer.ColumnInfo("B", "TB"),
                new ValueToKeyMappingTransformer.ColumnInfo("C", "TC", textKeyValues: true),
                new ValueToKeyMappingTransformer.ColumnInfo("D", "TD", textKeyValues: true),
                new ValueToKeyMappingTransformer.ColumnInfo("E", "TE"),
                new ValueToKeyMappingTransformer.ColumnInfo("F", "TF"),
                new ValueToKeyMappingTransformer.ColumnInfo("G", "TG"),
                new ValueToKeyMappingTransformer.ColumnInfo("H", "TH", textKeyValues: true) });
            var termTransformer = termEst.Fit(dataView);
            dataView = termTransformer.Transform(dataView);

            var pipe = new KeyToVectorMappingEstimator(Env,
                 new KeyToVectorMappingTransformer.ColumnInfo("TA", "CatA", true),
                 new KeyToVectorMappingTransformer.ColumnInfo("TB", "CatB", false),
                 new KeyToVectorMappingTransformer.ColumnInfo("TC", "CatC", false),
                 new KeyToVectorMappingTransformer.ColumnInfo("TD", "CatD", true),
                 new KeyToVectorMappingTransformer.ColumnInfo("TE", "CatE", false),
                 new KeyToVectorMappingTransformer.ColumnInfo("TF", "CatF", true),
                 new KeyToVectorMappingTransformer.ColumnInfo("TG", "CatG", true),
                 new KeyToVectorMappingTransformer.ColumnInfo("TH", "CatH", false)
                 );

            var result = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(result);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;
            VBuffer<int> slotRanges = default;

            var column = result.Schema["CatA"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "A", "B" });

            column = result.Schema["CatB"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 1);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });
            column.Metadata.GetValue(MetadataUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value), new int[2] { 0, 0 });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatC"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 4);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[4] { "[0].3", "[0].5", "[1].3", "[1].5" });
            column.Metadata.GetValue(MetadataUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value), new int[4] { 0, 1, 2, 3 });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatD"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "6", "1" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatE"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            column.Metadata.GetValue(MetadataUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value), new int[4] { 0, 5, 6, 11 });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatF"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.IsNormalized });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatG"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 3);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[3] { "A", "D", "E" });

            column = result.Schema["CatH"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "D", "E" });
            column.Metadata.GetValue(MetadataUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value), new int[2] { 0, 1 });
            Assert.True(column.IsNormalized());
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(0, Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Term{col=B:A} xf=KeyToVector{col=C:B col={name=D source=B bag+}} in=f:\2.txt" }));
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ML.Data.ReadFromEnumerable(data);
            var est = new ValueToKeyMappingEstimator(Env, new[]{
                    new ValueToKeyMappingTransformer.ColumnInfo("A", "TermA"),
                    new ValueToKeyMappingTransformer.ColumnInfo("B", "TermB"),
                    new ValueToKeyMappingTransformer.ColumnInfo("C", "TermC")
            });
            var transformer = est.Fit(dataView);
            dataView = transformer.Transform(dataView);
            var pipe = new KeyToVectorMappingEstimator(Env,
                new KeyToVectorMappingTransformer.ColumnInfo("TermA", "CatA", false),
                new KeyToVectorMappingTransformer.ColumnInfo("TermB", "CatB", true)
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
