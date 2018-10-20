// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System;
using System.IO;
using System.Linq;
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

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new OneHotHashEncodingEstimator(Env, new[]{
                    new OneHotHashEncodingEstimator.ColumnInfo("A", "CatA", CategoricalTransform.OutputKind.Bag),
                    new OneHotHashEncodingEstimator.ColumnInfo("A", "CatB", CategoricalTransform.OutputKind.Bin),
                    new OneHotHashEncodingEstimator.ColumnInfo("A", "CatC", CategoricalTransform.OutputKind.Ind),
                    new OneHotHashEncodingEstimator.ColumnInfo("A", "CatD", CategoricalTransform.OutputKind.Key),
                });

            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void CategoricalHashStatic()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(Env, ctx => (
                ScalarString: ctx.LoadText(1),
                VectorString: ctx.LoadText(1, 4)));
            var data = reader.Read(new MultiFileSource(dataPath));
            var wrongCollection = new[] { new TestClass() { A = "1", B = "2", C = "3", }, new TestClass() { A = "4", B = "5", C = "6" } };

            var invalidData = ComponentCreation.CreateDataView(Env, wrongCollection);
            var est = data.MakeNewEstimator().
                  Append(row => (
                  A: row.ScalarString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashScalarOutputKind.Ind),
                  B: row.VectorString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashVectorOutputKind.Ind),
                  C: row.VectorString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashVectorOutputKind.Bag),
                  D: row.ScalarString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashScalarOutputKind.Bin),
                  E: row.VectorString.OneHotHashEncoding(outputKind: CategoricalHashStaticExtensions.OneHotHashVectorOutputKind.Bin)
                  ));

            TestEstimatorCore(est.AsDynamic, data.AsDynamic, invalidInput: invalidData);

            var outputPath = GetOutputPath("CategoricalHash", "featurized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                var savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data).AsDynamic, 4);
                savedData = new ChooseColumnsTransform(Env, savedData, "A", "B", "C", "D", "E");
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

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

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var bagPipe = new OneHotHashEncodingEstimator(Env,
                new OneHotHashEncodingEstimator.ColumnInfo("A", "CatA", CategoricalTransform.OutputKind.Bag, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("B", "CatB", CategoricalTransform.OutputKind.Bag, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("C", "CatC", CategoricalTransform.OutputKind.Bag, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("D", "CatD", CategoricalTransform.OutputKind.Bag, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("E", "CatE", CategoricalTransform.OutputKind.Ind, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("F", "CatF", CategoricalTransform.OutputKind.Ind, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("A", "CatG", CategoricalTransform.OutputKind.Key, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("B", "CatH", CategoricalTransform.OutputKind.Key, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("A", "CatI", CategoricalTransform.OutputKind.Bin, invertHash: -1),
                new OneHotHashEncodingEstimator.ColumnInfo("B", "CatJ", CategoricalTransform.OutputKind.Bin, invertHash: -1));

            var bagResult = bagPipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(bagResult);
            Done();
        }


        private void ValidateMetadata(IDataView result)
        {
            Assert.True(result.Schema.TryGetColumnIndex("CatA", out int colA));
            Assert.True(result.Schema.TryGetColumnIndex("CatB", out int colB));
            Assert.True(result.Schema.TryGetColumnIndex("CatC", out int colC));
            Assert.True(result.Schema.TryGetColumnIndex("CatD", out int colD));
            Assert.True(result.Schema.TryGetColumnIndex("CatE", out int colE));
            Assert.True(result.Schema.TryGetColumnIndex("CatF", out int colF));
            Assert.True(result.Schema.TryGetColumnIndex("CatG", out int colG));
            Assert.True(result.Schema.TryGetColumnIndex("CatH", out int colH));
            Assert.True(result.Schema.TryGetColumnIndex("CatI", out int colI));
            Assert.True(result.Schema.TryGetColumnIndex("CatJ", out int colJ));
            var types = result.Schema.GetMetadataTypes(colA);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            VBuffer<ReadOnlyMemory<char>> slots = default;
            VBuffer<int> slotRanges = default;
            bool normalized = default;
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colA, ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "0:A", "1:B" });

            types = result.Schema.GetMetadataTypes(colB);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colB, ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colB, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colC);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colC, ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "1:6", "1:2", "0:1", "0:3", "0:5", "1:4" });

            types = result.Schema.GetMetadataTypes(colD);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colD, ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "-1", "1" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colD, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colE);
            Assert.Equal(types.Select(x => x.Key), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colE, ref slots);
            Assert.True(slots.Length == 131072);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()).Distinct(), new string[14] { "[0].", "[0].0:E", "[0].0:D", "[0].1:E", "[0].1:D", "[0].0:A", "[0].1:A", "[1].", "[1].0:E", "[1].0:D", "[1].1:E", "[1].1:D", "[1].0:A", "[1].1:A" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colE, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[4] { "0", "65535", "65536", "131071" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colE, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colF);
            Assert.Equal(types.Select(x => x.Key), new string[3] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.CategoricalSlotRanges, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colF, ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "E", "D" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colF, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[2] { "0", "65535" });
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colF, ref normalized);
            Assert.True(normalized);

            types = result.Schema.GetMetadataTypes(colG);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.KeyValues });
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, colG, ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "0:A", "1:B" });

            types = result.Schema.GetMetadataTypes(colH);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.KeyValues });
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, colH, ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });

            types = result.Schema.GetMetadataTypes(colI);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.SlotNames });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colI, ref slots);
            Assert.True(slots.Length == 36);

            types = result.Schema.GetMetadataTypes(colJ);
            Assert.Equal(types.Select(x => x.Key), new string[2] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.IsNormalized });
            result.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colJ, ref slots);
            Assert.True(slots.Length == 18);
            result.Schema.GetMetadata(MetadataUtils.Kinds.IsNormalized, colJ, ref normalized);
            Assert.True(normalized);
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
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new OneHotHashEncodingEstimator(Env, new[]{
                    new OneHotHashEncodingEstimator.ColumnInfo("A", "CatHashA"),
                    new OneHotHashEncodingEstimator.ColumnInfo("B", "CatHashB"),
                    new OneHotHashEncodingEstimator.ColumnInfo("C", "CatHashC")
            });
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
