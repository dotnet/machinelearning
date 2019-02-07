﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class CategoricalHashTests : TestDataPipeBase
    {
        public CategoricalHashTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public string A;
            public string B;
            public string C;
        }

        private class TestMeta
        {
            [VectorType(2)]
            public string[] A;
            public string B;
            [VectorType(2)]
            public float[] C;
            public float D;
            [VectorType(2)]
            public string[] E;
            public string F;
        }

        [Fact]
        public void CategoricalHashWorkout()
        {
            var data = new[] { new TestClass() { A = "1", B = "2", C = "3", }, new TestClass() { A = "4", B = "5", C = "6" } };

            var dataView = ML.Data.ReadFromEnumerable(data);
            var pipe = ML.Transforms.Categorical.OneHotHashEncoding(new[]{
                    new OneHotHashEncodingEstimator.ColumnInfo("CatA", "A", OneHotEncodingTransformer.OutputKind.Bag),
                    new OneHotHashEncodingEstimator.ColumnInfo("CatB", "A", OneHotEncodingTransformer.OutputKind.Bin),
                    new OneHotHashEncodingEstimator.ColumnInfo("CatC", "A", OneHotEncodingTransformer.OutputKind.Ind),
                    new OneHotHashEncodingEstimator.ColumnInfo("CatD", "A", OneHotEncodingTransformer.OutputKind.Key),
                });

            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void CategoricalHashStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateReader(ML, ctx => (
                ScalarString: ctx.LoadText(1),
                VectorString: ctx.LoadText(1, 4)));
            var data = reader.Read(dataPath);
            var wrongCollection = new[] { new TestClass() { A = "1", B = "2", C = "3", }, new TestClass() { A = "4", B = "5", C = "6" } };

            var invalidData = ML.Data.ReadFromEnumerable(wrongCollection);
            var est = data.MakeNewEstimator().
                  Append(row => (
                      row.ScalarString,
                      row.VectorString,
                      // Create a VarVector column
                      VarVectorString: row.ScalarString.TokenizeText())).
                  Append(row => (
                      A: row.ScalarString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashScalarOutputKind.Ind),
                      B: row.VectorString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashVectorOutputKind.Ind),
                      C: row.VectorString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashVectorOutputKind.Bag),
                      D: row.ScalarString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashScalarOutputKind.Bin),
                      E: row.VectorString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashVectorOutputKind.Bin),
                      F: row.VarVectorString.OneHotHashEncoding()
                  ));

            TestEstimatorCore(est.AsDynamic, data.AsDynamic, invalidInput: invalidData);

            var outputPath = GetOutputPath("CategoricalHash", "featurized.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data).AsDynamic, 4);
            var view = ML.Transforms.SelectColumns("A", "B", "C", "D", "E", "F").Fit(savedData).Transform(savedData);
            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(view, fs, headerRow: true, keepHidden: true);

            CheckEquality("CategoricalHash", "featurized.tsv");
            Done();
        }

        [Fact]
        public void TestMetadataPropagation()
        {
            var data = new[] {
                new TestMeta() { A = new string[2] { "A", "B"}, B = "C", C= new float[2] { 1.0f,2.0f}, D = 1.0f , E= new string[2]{"A","D"}, F="D"},
                new TestMeta() { A = new string[2] { "A", "B"}, B = "C", C =new float[2] { 3.0f,4.0f}, D = -1.0f, E= new string[2]{"E","A"}, F="E"},
                new TestMeta() { A = new string[2] { "A", "B"}, B = "C", C =new float[2] { 5.0f,6.0f}, D = 1.0f , E= new string[2]{"D","E"}, F="D"} };

            var dataView = ML.Data.ReadFromEnumerable(data);
            var bagPipe = ML.Transforms.Categorical.OneHotHashEncoding(
                new OneHotHashEncodingEstimator.ColumnInfo("CatA", "A", OneHotEncodingTransformer.OutputKind.Bag, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("CatB", "B", OneHotEncodingTransformer.OutputKind.Bag, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("CatC", "C", OneHotEncodingTransformer.OutputKind.Bag, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("CatD", "D", OneHotEncodingTransformer.OutputKind.Bag, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("CatE", "E", OneHotEncodingTransformer.OutputKind.Ind, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("CatF", "F", OneHotEncodingTransformer.OutputKind.Ind, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("CatG", "A", OneHotEncodingTransformer.OutputKind.Key, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("CatH", "B", OneHotEncodingTransformer.OutputKind.Key, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("CatI", "A", OneHotEncodingTransformer.OutputKind.Bin, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("CatJ", "B", OneHotEncodingTransformer.OutputKind.Bin, invertHash: -1));

            var bagResult = bagPipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(bagResult);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;
            VBuffer<int> slotRanges = default;

            var column = result.Schema["CatA"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "0:A", "1:B" });

            column = result.Schema["CatB"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatC"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "1:6", "1:2", "0:1", "0:3", "0:5", "1:4" });

            column = result.Schema["CatD"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "-1", "1" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatE"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 131072);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()).Distinct(), new string[14] { "[0].", "[0].0:E", "[0].0:D", "[0].1:E", "[0].1:D", "[0].0:A", "[0].1:A", "[1].", "[1].0:E", "[1].0:D", "[1].1:E", "[1].1:D", "[1].0:A", "[1].1:A" });
            column.Metadata.GetValue(MetadataUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[4] { "0", "65535", "65536", "131071" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatF"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "E", "D" });
            column.Metadata.GetValue(MetadataUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[2] { "0", "65535" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatG"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.KeyValues });
            column.GetKeyValues(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "0:A", "1:B" });

            column = result.Schema["CatH"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.KeyValues });
            column.GetKeyValues(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });

            column = result.Schema["CatI"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[1] { MetadataUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 36);

            column = result.Schema["CatJ"];
            Assert.Equal(column.Metadata.Schema.Select(x => x.Name), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 18);
            Assert.True(column.IsNormalized());
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=CatHash{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = "1", B = "2", C = "3", }, new TestClass() { A = "4", B = "5", C = "6" } };
            var dataView = ML.Data.ReadFromEnumerable(data);
            var pipe = ML.Transforms.Categorical.OneHotHashEncoding(new[]{
                    new OneHotHashEncodingEstimator.ColumnInfo("CatHashA", "A"),
                    new OneHotHashEncodingEstimator.ColumnInfo("CatHashB", "B"),
                    new OneHotHashEncodingEstimator.ColumnInfo("CatHashC", "C")
            });
            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ch = Env.Start("saving"))
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, ch, ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
            }
        }

    }
}
