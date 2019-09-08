// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public sealed class SvmLightTests : TestDataPipeBase
    {
        public SvmLightTests(ITestOutputHelper output) : base(output)
        {
        }

#pragma warning disable 0649 // Disable warnings about unused members. They are used through reflection.
        private sealed class SvmLightOutput
        {
            public float Label;
            public float Weight;
            [KeyType(ulong.MaxValue - 1)]
            public ulong GroupId = ulong.MaxValue;
            public ReadOnlyMemory<char> Comment;
            public VBuffer<float> Features;
        }
#pragma warning restore 0649

        [Fact]
        public void TestSvmLightLoaderAndSaver()
        {
            // Test with a specified size parameter. The "6" feature should be omitted.
            // Also the blank and completely fully commented lines should be omitted,
            // and the feature 2:3 that appears in the comment should not appear.
            var path = GetOutputPath("DataA.txt");
            File.WriteAllLines(path, new string[] {
                "1\t1:3\t4:6",
                "  -1 cost:5\t2:4 \t4:7\t6:-1   ",
                "",
                "1\t5:-2 # A comment! 2:3",
                "# What a nice full line comment",
                "1 cost:0.5\t2:3.14159",
            });

            var data = ML.Data.LoadFromSvmLightFile(path, inputSize: 5);
            Assert.True(data.Schema["Features"].Type.GetValueCount() == 5);

            var schemaDef = SchemaDefinition.Create(typeof(SvmLightOutput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 5);
            schemaDef["Features"].AddAnnotation(
                AnnotationUtils.Kinds.SlotNames, new VBuffer<ReadOnlyMemory<char>>(3, new[] { "aurora".AsMemory(), "beachwood".AsMemory(), "chagrin".AsMemory() }),
                new VectorDataViewType(TextDataViewType.Instance, 3));

            var expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(5, 2, new[] { 3f, 6f }, new[] { 0, 3 }) },
                new SvmLightOutput() { Label = -1, Weight = 5, Features = new VBuffer<float>(5, 2, new[] { 4f, 7f }, new[] { 1, 3 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(5, 1, new[] { -2f }, new[] { 4 }), Comment = " A comment! 2:3".AsMemory() },
                new SvmLightOutput() { Label = 1, Weight = 0.5f, Features = new VBuffer<float>(5, 1, new[] { 3.14159f }, new[] { 1 }) },
            }, schemaDef);
            CheckSameValues(data, expectedData, checkId: false);

            // Save, reload and compare dataviews again.
            var outputPath = DeleteOutputPath("DataA-output.txt");
            using (var stream = File.Create(outputPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream, exampleWeightColumnName: "Weight");
            data = ML.Data.LoadFromSvmLightFile(outputPath, inputSize: 5);
            CheckSameValues(ColumnSelectingTransformer.CreateDrop(Env, data, "Comment"),
                ColumnSelectingTransformer.CreateDrop(Env, expectedData, "Comment"), checkId: false);

            // If we specify the size parameter, and zero-based feature indices, both indices 5 and 6 should
            // not appear.
            data = ML.Data.LoadFromSvmLightFile(path, inputSize: 5, zeroBased: true);
            expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(5, 2, new[] { 3f, 6f }, new[] { 1, 4 }) },
                new SvmLightOutput() { Label = -1, Weight = 5, Features = new VBuffer<float>(5, 2, new[] { 4f, 7f }, new[] { 2, 4 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(5, 0, new float[0], new int[0]), Comment = " A comment! 2:3".AsMemory() },
                new SvmLightOutput() { Label = 1, Weight = 0.5f, Features = new VBuffer<float>(5, 1, new[] { 3.14159f }, new[] { 2 }) },
            }, schemaDef);
            CheckSameValues(data, expectedData, checkId: false);

            // Save, reload and compare dataviews again.
            outputPath = DeleteOutputPath("DataA-output-1.txt");
            using (var stream = File.Create(outputPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream, zeroBasedIndexing: true, exampleWeightColumnName: "Weight");
            data = ML.Data.LoadFromSvmLightFile(outputPath, inputSize: 5, zeroBased: true);
            CheckSameValues(ColumnSelectingTransformer.CreateDrop(Env, data, "Comment"),
                ColumnSelectingTransformer.CreateDrop(Env, expectedData, "Comment"), checkId: false);

            // Test with autodetermined sizes. The the "6" feature should be included,
            // and the feature vector should have length 6.
            data = ML.Data.LoadFromSvmLightFile(path);
            Assert.True(data.Schema["Features"].Type.GetValueCount() == 6);

            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 6);
            expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 2, new[] { 3f, 6f }, new[] { 0, 3 }) },
                new SvmLightOutput() { Label = -1, Weight = 5, Features = new VBuffer<float>(6, 3, new[] { 4f, 7f, -1f }, new[] { 1, 3, 5 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 1, new[] { -2f }, new[] { 4 }), Comment = " A comment! 2:3".AsMemory() },
                new SvmLightOutput() { Label = 1, Weight = 0.5f, Features = new VBuffer<float>(6, 1, new[] { 3.14159f }, new[] { 1 }) },
            }, schemaDef);
            CheckSameValues(data, expectedData, checkId: false);

            // Save, reload and compare dataviews again.
            outputPath = DeleteOutputPath("DataA-output-2.txt");
            using (var stream = File.Create(outputPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream, exampleWeightColumnName: "Weight");
            data = ML.Data.LoadFromSvmLightFile(outputPath);
            CheckSameValues(ColumnSelectingTransformer.CreateDrop(Env, data, "Comment"),
                ColumnSelectingTransformer.CreateDrop(Env, expectedData, "Comment"), checkId: false);

            // Test with a term mapping, instead of the actual SVM^light format that
            // requires positive integers. ALso check that qid works here.
            path = GetOutputPath("DataB.txt");
            File.WriteAllLines(path, new string[] {
                "1 qid:1 aurora:3.14159 beachwood:123",
                "-1 qid:5 beachwood:345 chagrin:-21",
            });
            var model = ML.Data.CreateSvmLightLoaderWithFeatureNames(dataSample: new MultiFileSource(path));
            data = model.Load(path);
            Assert.True(data.Schema["Features"].Type.GetValueCount() == 3);

            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 3);
            expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, GroupId = 1, Features = new VBuffer<float>(3, 2, new[] { 3.14159f, 123f }, new[] { 0, 1 }) },
                new SvmLightOutput() { Label = -1, Weight = 1, GroupId = 5, Features = new VBuffer<float>(3, 2, new[] { 345f, -21f }, new[] { 1, 2 }) },
            }, schemaDef);
            CheckSameValues(data, expectedData, checkId: false);
            CheckSameSchemas(data.Schema, expectedData.Schema);

            // Save, reload and compare dataviews again.
            outputPath = DeleteOutputPath("DataA-output-3.txt");
            using (var stream = File.Create(outputPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream, zeroBasedIndexing: true, rowGroupColumnName: "GroupId");
            data = ML.Data.LoadFromSvmLightFile(outputPath, zeroBased: true);
            CheckSameValues(data, expectedData, checkId: false);

            // We reload the model, but on a new set of data. The "euclid" key should be
            // ignored as it would not have been detected by the term transform.
            path = GetOutputPath("DataC.txt");
            File.WriteAllLines(path, new string[] {
                "-1 aurora:1 chagrin:2",
                "1 chagrin:3"
            });
            data = model.Load(path);
            Assert.True(data.Schema["Features"].Type.GetValueCount() == 3);

            expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = -1, Weight = 1, Features = new VBuffer<float>(3, 2, new[] { 1f, 2f }, new[] { 0, 2 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(3, 1, new[] { 3f }, new[] { 2 }) },
            }, schemaDef);
            CheckSameValues(data, expectedData, checkId: false);

            // Save, reload and compare dataviews again.
            outputPath = DeleteOutputPath("DataA-output-4.txt");
            using (var stream = File.Create(outputPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream);
            data = ML.Data.LoadFromSvmLightFile(outputPath);
            CheckSameValues(data, expectedData, checkId: false);

            path = GetOutputPath("DataD.txt");
            File.WriteAllLines(path, new string[] { "1 aurora:2 :3" });
            data = ML.Data.LoadFromSvmLightFileWithFeatureNames(path);
            Assert.True(data.Schema["Features"].Type.GetValueCount() == 1);

            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 1);
            expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(1, 1, new[] { 2f }, new[] { 0 }) },
            }, schemaDef);
            CheckSameValues(data, expectedData, checkId: false);

            // Save, reload and compare dataviews again.
            outputPath = DeleteOutputPath("DataA-output-5.txt");
            using (var stream = File.Create(outputPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream, zeroBasedIndexing: true);
            data = ML.Data.LoadFromSvmLightFile(outputPath, zeroBased: true);
            CheckSameValues(data, expectedData, checkId: false);

            Done();
        }
    }
}
