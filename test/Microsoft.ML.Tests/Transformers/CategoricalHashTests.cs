// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
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
            [VectorType(2)]
            public string[] B;
            public string[] C;
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
            var data = new[] { new TestClass() { A = "1", B = new[] { "2", "3" }, C = new[] { "2", "3", "4" } }, new TestClass() { A = "4", B = new[] { "4", "5" }, C = new[] { "3", "4", "5" } } };

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Categorical.OneHotHashEncoding(new[]{
                    new OneHotHashEncodingEstimator.ColumnOptions("CatA", "A",  OneHotEncodingEstimator.OutputKind.Bag),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatB", "A", OneHotEncodingEstimator.OutputKind.Binary),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatC", "A", OneHotEncodingEstimator.OutputKind.Indicator),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatD", "A", OneHotEncodingEstimator.OutputKind.Key),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatVA", "B", OneHotEncodingEstimator.OutputKind.Bag),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatVB", "B", OneHotEncodingEstimator.OutputKind.Binary),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatVC", "B", OneHotEncodingEstimator.OutputKind.Indicator),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatVD", "B", OneHotEncodingEstimator.OutputKind.Key),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatVVA", "C", OneHotEncodingEstimator.OutputKind.Bag),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatVVB", "C", OneHotEncodingEstimator.OutputKind.Binary),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatVVC", "C", OneHotEncodingEstimator.OutputKind.Indicator),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatVVD", "C", OneHotEncodingEstimator.OutputKind.Key),
                });

            TestEstimatorCore(pipe, dataView);
            var outputPath = GetOutputPath("CategoricalHash", "oneHotHash.tsv");
            var savedData = pipe.Fit(dataView).Transform(dataView);

            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);
            CheckEquality("CategoricalHash", "oneHotHash.tsv");
            Done();
        }

        [Fact]
        public void CategoricalHash()
        {
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarString", DataKind.String, 1),
                new TextLoader.Column("VectorString", DataKind.String, 1, 4),
                new TextLoader.Column("SingleVectorString", DataKind.String, new[] { new TextLoader.Range(1, 1) })
            });
            var wrongCollection = new[] { new TestClass() { A = "1", B = new[] { "2", "3" }, C = new[] { "2", "3", "4" } }, new TestClass() { A = "4", B = new[] { "4", "5" }, C = new[] { "3", "4", "5" } } };

            var invalidData = ML.Data.LoadFromEnumerable(wrongCollection);
            var est = ML.Transforms.Text.TokenizeIntoWords("VarVectorString", "ScalarString")
                .Append(ML.Transforms.Categorical.OneHotHashEncoding("A", "ScalarString", outputKind: OneHotEncodingEstimator.OutputKind.Indicator))
                .Append(ML.Transforms.Categorical.OneHotHashEncoding("B", "VectorString", outputKind: OneHotEncodingEstimator.OutputKind.Indicator))
                .Append(ML.Transforms.Categorical.OneHotHashEncoding("C", "VectorString", outputKind: OneHotEncodingEstimator.OutputKind.Bag))
                .Append(ML.Transforms.Categorical.OneHotHashEncoding("D", "ScalarString", outputKind: OneHotEncodingEstimator.OutputKind.Binary))
                .Append(ML.Transforms.Categorical.OneHotHashEncoding("E", "VectorString", outputKind: OneHotEncodingEstimator.OutputKind.Binary))
                .Append(ML.Transforms.Categorical.OneHotHashEncoding("F", "VarVectorString", outputKind: OneHotEncodingEstimator.OutputKind.Bag))
                // The following column and SingleVectorString are meant to test the special case of a vector that happens to be of length 1.
                .Append(ML.Transforms.Categorical.OneHotHashEncoding("G", "SingleVectorString", outputKind: OneHotEncodingEstimator.OutputKind.Bag));

            TestEstimatorCore(est, data, invalidInput: invalidData);

            var outputPath = GetOutputPath("CategoricalHash", "featurized.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
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

            var dataView = ML.Data.LoadFromEnumerable(data);
            var bagPipe = ML.Transforms.Categorical.OneHotHashEncoding(
                new OneHotHashEncodingEstimator.ColumnOptions("CatA", "A", OneHotEncodingEstimator.OutputKind.Bag, maximumNumberOfInverts: -1),
                new OneHotHashEncodingEstimator.ColumnOptions("CatB", "B", OneHotEncodingEstimator.OutputKind.Bag, maximumNumberOfInverts: -1),
                new OneHotHashEncodingEstimator.ColumnOptions("CatC", "C", OneHotEncodingEstimator.OutputKind.Bag, maximumNumberOfInverts: -1),
                new OneHotHashEncodingEstimator.ColumnOptions("CatD", "D", OneHotEncodingEstimator.OutputKind.Bag, maximumNumberOfInverts: -1),
                new OneHotHashEncodingEstimator.ColumnOptions("CatE", "E", OneHotEncodingEstimator.OutputKind.Indicator, maximumNumberOfInverts: -1),
                new OneHotHashEncodingEstimator.ColumnOptions("CatF", "F", OneHotEncodingEstimator.OutputKind.Indicator, maximumNumberOfInverts: -1),
                new OneHotHashEncodingEstimator.ColumnOptions("CatG", "A", OneHotEncodingEstimator.OutputKind.Key, maximumNumberOfInverts: -1),
                new OneHotHashEncodingEstimator.ColumnOptions("CatH", "B", OneHotEncodingEstimator.OutputKind.Key, maximumNumberOfInverts: -1),
                new OneHotHashEncodingEstimator.ColumnOptions("CatI", "A", OneHotEncodingEstimator.OutputKind.Binary, maximumNumberOfInverts: -1),
                new OneHotHashEncodingEstimator.ColumnOptions("CatJ", "B", OneHotEncodingEstimator.OutputKind.Binary, maximumNumberOfInverts: -1));

            var bagResult = bagPipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(bagResult);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {
            VBuffer<ReadOnlyMemory<char>> slots = default;
            VBuffer<int> slotRanges = default;

            var column = result.Schema["CatA"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "0:A", "1:B" });

            column = result.Schema["CatB"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[2] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatC"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[6] { "1:2", "0:5", "1:4", "1:6", "0:3", "0:1" });

            column = result.Schema["CatD"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[2] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "-1", "1" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatE"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[3] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.CategoricalSlotRanges, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 131072);
            System.Diagnostics.Trace.WriteLine(slots.Items().Select(x => x.Value.ToString()).Distinct());
            var temp = slots.Items().Select(x => x.Value.ToString()).Distinct();
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()).Distinct(), new string[14] { "[0].", "[0].0:A", "[0].0:E", "[0].0:D", "[0].1:A", "[0].1:E", "[0].1:D", "[1].", "[1].0:A", "[1].0:E", "[1].0:D", "[1].1:A", "[1].1:E", "[1].1:D" });
            column.Annotations.GetValue(AnnotationUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 4);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[4] { "0", "65535", "65536", "131071" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatF"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[3] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.CategoricalSlotRanges, AnnotationUtils.Kinds.IsNormalized });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "E", "D" });
            column.Annotations.GetValue(AnnotationUtils.Kinds.CategoricalSlotRanges, ref slotRanges);
            Assert.True(slotRanges.Length == 2);
            Assert.Equal(slotRanges.Items().Select(x => x.Value.ToString()), new string[2] { "0", "65535" });
            Assert.True(column.IsNormalized());

            column = result.Schema["CatG"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.KeyValues });
            column.GetKeyValues(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[2] { "0:A", "1:B" });

            column = result.Schema["CatH"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.KeyValues });
            column.GetKeyValues(ref slots);
            Assert.True(slots.Length == 65536);
            Assert.Equal(slots.Items().Select(x => x.Value.ToString()), new string[1] { "C" });

            column = result.Schema["CatI"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[1] { AnnotationUtils.Kinds.SlotNames });
            column.GetSlotNames(ref slots);
            Assert.True(slots.Length == 36);

            column = result.Schema["CatJ"];
            Assert.Equal(column.Annotations.Schema.Select(x => x.Name), new string[2] { AnnotationUtils.Kinds.SlotNames, AnnotationUtils.Kinds.IsNormalized });
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
            var data = new[] { new TestClass() { A = "1", B = new[] { "2", "3" }, C = new[] { "2", "3", "4" } }, new TestClass() { A = "4", B = new[] { "4", "5" }, C = new[] { "3", "4", "5" } } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = ML.Transforms.Categorical.OneHotHashEncoding(new[]{
                    new OneHotHashEncodingEstimator.ColumnOptions("CatHashA", "A"),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatHashB", "B"),
                    new OneHotHashEncodingEstimator.ColumnOptions("CatHashC", "C"),
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
