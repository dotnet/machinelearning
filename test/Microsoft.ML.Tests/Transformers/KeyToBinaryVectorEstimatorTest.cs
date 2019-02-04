// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
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

            var dataView = ML.Data.ReadFromEnumerable(data);
            dataView = new ValueToKeyMappingEstimator(Env, new[]{
                    new ValueToKeyMappingEstimator.ColumnInfo("TermA", "A"),
                    new ValueToKeyMappingEstimator.ColumnInfo("TermB", "B"),
                    new ValueToKeyMappingEstimator.ColumnInfo("TermC", "C", textKeyValues:true)
                }).Fit(dataView).Transform(dataView);

            var pipe = new KeyToBinaryVectorMappingEstimator(Env, new KeyToBinaryVectorMappingTransformer.ColumnInfo("CatA", "TermA"),
                new KeyToBinaryVectorMappingTransformer.ColumnInfo("CatC", "TermC"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void KeyToBinaryVectorStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateReader(Env, ctx => (
                ScalarString: ctx.LoadText(1),
                VectorString: ctx.LoadText(1, 4)
            ));

            var data = reader.Read(dataPath);

            // Non-pigsty Term.
            var dynamicData = new ValueToKeyMappingEstimator(Env, new[] {
                new ValueToKeyMappingEstimator.ColumnInfo("A", "ScalarString"),
                new ValueToKeyMappingEstimator.ColumnInfo("B", "VectorString") })
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


            var dataView = ML.Data.ReadFromEnumerable(data);
            var termEst = new ValueToKeyMappingEstimator(Env, new[] {
                new ValueToKeyMappingEstimator.ColumnInfo("TA", "A", textKeyValues: true),
                new ValueToKeyMappingEstimator.ColumnInfo("TB", "B", textKeyValues: true),
                new ValueToKeyMappingEstimator.ColumnInfo("TC", "C"),
                new ValueToKeyMappingEstimator.ColumnInfo("TD", "D") });
            var termTransformer = termEst.Fit(dataView);
            dataView = termTransformer.Transform(dataView);

            var pipe = new KeyToBinaryVectorMappingEstimator(Env,
                 new KeyToBinaryVectorMappingTransformer.ColumnInfo("CatA", "TA"),
                 new KeyToBinaryVectorMappingTransformer.ColumnInfo("CatB", "TB"),
                 new KeyToBinaryVectorMappingTransformer.ColumnInfo("CatC", "TC"),
                 new KeyToBinaryVectorMappingTransformer.ColumnInfo("CatD", "TD"));

            var result = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(result);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;

            var column = result.Schema["CatA"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 6);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "[0].Bit2", "[0].Bit1", "[0].Bit0", "[1].Bit2", "[1].Bit1", "[1].Bit0" });

            column = result.Schema["CatB"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 2);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "Bit1", "Bit0" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatC"];
            Assert.Empty(column.Metadata.Schema);

            column = result.Schema["CatD"];
            Assert.Equal(column.Metadata.Schema.Single().Name, MetadataUtils.Kinds.IsNormalized);
            Assert.True(column.IsNormalized());
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
            var dataView = ML.Data.ReadFromEnumerable(data);
            var est = new ValueToKeyMappingEstimator(Env, new[]{
                    new ValueToKeyMappingEstimator.ColumnInfo("TermA", "A"),
                    new ValueToKeyMappingEstimator.ColumnInfo("TermB", "B", textKeyValues:true),
                    new ValueToKeyMappingEstimator.ColumnInfo("TermC", "C")
            });
            var transformer = est.Fit(dataView);
            dataView = transformer.Transform(dataView);
            var pipe = new KeyToBinaryVectorMappingEstimator(Env,
                new KeyToBinaryVectorMappingTransformer.ColumnInfo("CatA", "TermA"),
                new KeyToBinaryVectorMappingTransformer.ColumnInfo("CatB", "TermB"),
                new KeyToBinaryVectorMappingTransformer.ColumnInfo("CatC", "TermC")
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
