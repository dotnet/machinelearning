// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
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

        private string CreateDataset(string name, string[] data)
        {
            var path = DeleteOutputPath(TestName + name);
            File.WriteAllLines(path, data);
            return path;
        }

        private void TestSvmLight(string path, string savingPath, int inputSize, int expectedInputSize, bool zeroBased, IDataView expectedData, long? numberOfRows = null)
        {
            var data = ML.Data.LoadFromSvmLightFile(path, inputSize: inputSize, zeroBased: zeroBased, numberOfRows: numberOfRows);
            Assert.True(data.Schema["Features"].Type.GetValueCount() == expectedInputSize);

            CheckSameValues(data, expectedData, checkId: false);

            // Save, reload and compare dataviews again.
            using (var stream = File.Create(savingPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream, zeroBasedIndexing: zeroBased, exampleWeightColumnName: "Weight");
            data = ML.Data.LoadFromSvmLightFile(savingPath, inputSize: inputSize, zeroBased: zeroBased);
            CheckSameValues(ColumnSelectingTransformer.CreateDrop(Env, data, "Comment"),
                ColumnSelectingTransformer.CreateDrop(Env, expectedData, "Comment"), checkId: false);

            Done();
        }

        [Fact]
        public void TestSvmLightLoaderAndSaverWithSpecifiedInputSize()
        {
            // Test with a specified size parameter. The "6" feature should be omitted.
            // Also the blank and completely fully commented lines should be omitted,
            // and the feature 2:3 that appears in the comment should not appear.

            var path = CreateDataset("-data.txt", new string[] {
                "1\t1:3\t4:6",
                "  -1 cost:5\t2:4 \t4:7\t6:-1   ",
                "",
                "1\t5:-2 # A comment! 2:3",
                "# What a nice full line comment",
                "1 cost:0.5\t2:3.14159",
            });

            var schemaDef = SchemaDefinition.Create(typeof(SvmLightOutput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 5);

            var expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(5, 2, new[] { 3f, 6f }, new[] { 0, 3 }) },
                new SvmLightOutput() { Label = -1, Weight = 5, Features = new VBuffer<float>(5, 2, new[] { 4f, 7f }, new[] { 1, 3 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(5, 1, new[] { -2f }, new[] { 4 }), Comment = " A comment! 2:3".AsMemory() },
                new SvmLightOutput() { Label = 1, Weight = 0.5f, Features = new VBuffer<float>(5, 1, new[] { 3.14159f }, new[] { 1 }) },
            }, schemaDef);
            var savingPath = DeleteOutputPath(TestName + "-saved-data.txt");
            TestSvmLight(path, savingPath, 5, 5, false, expectedData);
        }

        [Fact]
        public void TestSvmLightLoaderAndSaverWithSpecifiedInputSizeZeroBased()
        {
            // If we specify the size parameter, and zero-based feature indices, both indices 5 and 6 should
            // not appear.

            var path = CreateDataset("-data.txt", new string[] {
                "1\t1:3\t4:6",
                "  -1 cost:5\t2:4 \t4:7\t6:-1   ",
                "",
                "1\t5:-2 # A comment! 2:3",
                "# What a nice full line comment",
                "1 cost:0.5\t2:3.14159",
            });

            var schemaDef = SchemaDefinition.Create(typeof(SvmLightOutput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 5);
            var expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(5, 2, new[] { 3f, 6f }, new[] { 1, 4 }) },
                new SvmLightOutput() { Label = -1, Weight = 5, Features = new VBuffer<float>(5, 2, new[] { 4f, 7f }, new[] { 2, 4 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(5, 0, new float[0], new int[0]), Comment = " A comment! 2:3".AsMemory() },
                new SvmLightOutput() { Label = 1, Weight = 0.5f, Features = new VBuffer<float>(5, 1, new[] { 3.14159f }, new[] { 2 }) },
            }, schemaDef);
            var savingPath = DeleteOutputPath(TestName + "-saved-data.txt");
            TestSvmLight(path, savingPath, 5, 5, true, expectedData);
        }

        [Fact]
        public void TestSvmLightLoaderAndSaverAutoDetectInputSize()
        {
            // Test with autodetermined sizes. The the "6" feature should be included,
            // and the feature vector should have length 6.

            var path = CreateDataset("-data.txt", new string[] {
                "1\t1:3\t4:6",
                "  -1 cost:5\t2:4 \t4:7\t6:-1   ",
                "",
                "1\t5:-2 # A comment! 2:3",
                "# What a nice full line comment",
                "1 cost:0.5\t2:3.14159",
            });

            var schemaDef = SchemaDefinition.Create(typeof(SvmLightOutput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 6);

            var expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 2, new[] { 3f, 6f }, new[] { 0, 3 }) },
                new SvmLightOutput() { Label = -1, Weight = 5, Features = new VBuffer<float>(6, 3, new[] { 4f, 7f, -1f }, new[] { 1, 3, 5 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 1, new[] { -2f }, new[] { 4 }), Comment = " A comment! 2:3".AsMemory() },
                new SvmLightOutput() { Label = 1, Weight = 0.5f, Features = new VBuffer<float>(6, 1, new[] { 3.14159f }, new[] { 1 }) },
            }, schemaDef);
            var savingPath = DeleteOutputPath(TestName + "-saved-data.txt");
            TestSvmLight(path, savingPath, 0, 6, false, expectedData);
        }

        [Fact]
        public void TestSvmLightLoaderAndSaverWithTermMapping()
        {
            // Test with a term mapping, instead of the actual SVM^light format that
            // requires positive integers. ALso check that qid works here.
            var path = CreateDataset("-data.txt", new string[] {
                "1 qid:1 aurora:3.14159 beachwood:123",
                "-1 qid:5 beachwood:345 chagrin:-21",
            });

            var model = ML.Data.CreateSvmLightLoaderWithFeatureNames(dataSample: new MultiFileSource(path));
            var data = model.Load(path);
            Assert.True(data.Schema["Features"].Type.GetValueCount() == 3);

            var schemaDef = SchemaDefinition.Create(typeof(SvmLightOutput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 3);
            schemaDef["Features"].AddAnnotation(
                AnnotationUtils.Kinds.SlotNames, new VBuffer<ReadOnlyMemory<char>>(3, new[] { "aurora".AsMemory(), "beachwood".AsMemory(), "chagrin".AsMemory() }),
                new VectorDataViewType(TextDataViewType.Instance, 3));
            var expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, GroupId = 1, Features = new VBuffer<float>(3, 2, new[] { 3.14159f, 123f }, new[] { 0, 1 }) },
                new SvmLightOutput() { Label = -1, Weight = 1, GroupId = 5, Features = new VBuffer<float>(3, 2, new[] { 345f, -21f }, new[] { 1, 2 }) },
            }, schemaDef);
            CheckSameValues(data, expectedData, checkId: false);
            TestCommon.CheckSameSchemas(data.Schema, expectedData.Schema);

            // Save, reload and compare dataviews again.
            var outputPath = DeleteOutputPath(TestName + "-saved-data.txt");
            using (var stream = File.Create(outputPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream, zeroBasedIndexing: true, rowGroupColumnName: "GroupId");
            data = ML.Data.LoadFromSvmLightFile(outputPath, zeroBased: true);
            CheckSameValues(data, expectedData, checkId: false);

            // We reload the model, but on a new set of data. The "euclid" key should be
            // ignored as it would not have been detected by the term transform.
            path = CreateDataset("-data2.txt", new string[] {
                "-1 aurora:1 chagrin:2",
                "1 chagrin:3 euclid:4"
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
            outputPath = DeleteOutputPath(TestName + "-saved-data2.txt");
            using (var stream = File.Create(outputPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream);
            data = ML.Data.LoadFromSvmLightFile(outputPath);
            CheckSameValues(data, expectedData, checkId: false);
        }

        [Fact]
        public void TestSvmLightLoaderAndSaverWithTermMappingWithEmptyName()
        {
            var path = CreateDataset("-data.txt", new string[] { "1 aurora:2 :3" });
            var data = ML.Data.LoadFromSvmLightFileWithFeatureNames(path);
            Assert.True(data.Schema["Features"].Type.GetValueCount() == 1);

            var schemaDef = SchemaDefinition.Create(typeof(SvmLightOutput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 1);
            var expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(1, 1, new[] { 2f }, new[] { 0 }) },
            }, schemaDef);
            CheckSameValues(data, expectedData, checkId: false);

            // Save, reload and compare dataviews again.
            var outputPath = DeleteOutputPath("reloaded-output.txt");
            using (var stream = File.Create(outputPath))
                ML.Data.SaveInSvmLightFormat(expectedData, stream, zeroBasedIndexing: true);
            data = ML.Data.LoadFromSvmLightFile(outputPath, zeroBased: true);
            CheckSameValues(data, expectedData, checkId: false);

            Done();
        }

        [Fact]
        public void TestSvmLightLoaderNoDuplicateKeys()
        {
            var path = CreateDataset("-data.txt", new string[] {
                "-1 aurora:1 chagrin:2",
                "1 chagrin:3 euclid:4 chagrin:5"
            });

            var ex = Assert.Throws<InvalidOperationException>(() =>
            {
                var view = ML.Data.LoadFromSvmLightFileWithFeatureNames(path);
                using (var curs = view.GetRowCursor(view.Schema))
                {
                    var featuresGetter = curs.GetGetter<VBuffer<float>>(view.Schema["Features"]);
                    VBuffer<float> buffer = default;
                    while (curs.MoveNext())
                        featuresGetter(ref buffer);
                }
            });
            Assert.Contains("Duplicate keys found in dataset", ex.InnerException.Message);
        }

        [Fact]
        public void TestSvmLightLoaderBadLabel()
        {
            var path = CreateDataset("-data.txt", new string[] {
                "q\t1:3\t4:6",
                "  -1a cost:5\t2:4 \t4:7\t6:-1   ",
            });

            var data = ML.Data.LoadFromSvmLightFile(path);
            using (var curs = data.GetRowCursor(data.Schema["Label"]))
            {
                var getter = curs.GetGetter<float>(data.Schema["Label"]);
                float label = default;
                while (curs.MoveNext())
                {
                    getter(ref label);
                    Assert.True(float.IsNaN(label));
                }
            }
        }

        [Fact]
        public void TestSvmLightLoaderMissingGroupId()
        {
            var path = CreateDataset("-data.txt", new string[] {
                "1\tqid:-3\t1:3\t4:6",
            });
            var data = ML.Data.LoadFromSvmLightFile(path);
            using (var curs = data.GetRowCursor(data.Schema["GroupId"]))
            {
                var getter = curs.GetGetter<ulong>(data.Schema["GroupId"]);
                ulong group = default;
                while (curs.MoveNext())
                {
                    getter(ref group);
                    Assert.True(group == 0);
                }
            }
        }

        [Fact]
        public void TestSvmLightLoaderBadFeature()
        {
            // Test with a dataset that has a feature that cannot be parsed. The loader should ignore the value.
            var path = CreateDataset("-data.txt", new string[] {
                "1\t1:3\t4:6",
                "  -1 cost:5\t2:4 \t4:7\t6:-1   ",
                "",
                "1\t5:-2 # A comment! 2:3",
                "# What a nice full line comment",
                "1 cost:0.5\t2:3.14159",
                "-1 3:2 4:hello"
            });

            var schemaDef = SchemaDefinition.Create(typeof(SvmLightOutput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 6);

            var expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 2, new[] { 3f, 6f }, new[] { 0, 3 }) },
                new SvmLightOutput() { Label = -1, Weight = 5, Features = new VBuffer<float>(6, 3, new[] { 4f, 7f, -1f }, new[] { 1, 3, 5 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 1, new[] { -2f }, new[] { 4 }), Comment = " A comment! 2:3".AsMemory() },
                new SvmLightOutput() { Label = 1, Weight = 0.5f, Features = new VBuffer<float>(6, 1, new[] { 3.14159f }, new[] { 1 }) },
                new SvmLightOutput() { Label = -1, Weight = 1, Features = new VBuffer<float>(6, 1, new[] { 2f }, new[] { 2 }) },
            }, schemaDef);
            var savingPath = DeleteOutputPath(TestName + "-saved-data.txt");
            TestSvmLight(path, savingPath, 0, 6, false, expectedData);
        }

        [Fact]
        public void TestSvmLightLoaderNoColon()
        {
            var path = CreateDataset("-data.txt", new string[] {
                "1\t1;3\t4:6",
            });
            var data = ML.Data.LoadFromSvmLightFile(path);
            using (var curs = data.GetRowCursor(data.Schema["Features"]))
            {
                var getter = curs.GetGetter<VBuffer<float>>(data.Schema["Features"]);
                VBuffer<float> features = default;
                while (curs.MoveNext())
                {
                    getter(ref features);
                    Assert.True(features.Length == 4);
                    Assert.True(features.GetValues().Length == 1);
                    Assert.True(features.GetIndices().Length == 1);
                    Assert.True(features.GetValues()[0] == 6);
                    Assert.True(features.GetIndices()[0] == 3);
                }
            }
        }

        [Fact]
        public void TestSvmLightLoaderBadIndex()
        {
            // 0 index in 1-based parsing.
            var path = CreateDataset("-data.txt", new string[] {
                "1\t0:3\t4:6",
            });
            var data = ML.Data.LoadFromSvmLightFile(path);
            var ex = Assert.Throws<InvalidOperationException>(() =>
            {
                using (var curs = data.GetRowCursor(data.Schema["Features"]))
                {
                    var getter = curs.GetGetter<VBuffer<float>>(data.Schema["Features"]);
                    VBuffer<float> features = default;
                    while (curs.MoveNext())
                    {
                        getter(ref features);
                    }
                }
            });
            Assert.Contains("Encountered 0 index while parsing a 1-based dataset", ex.InnerException.Message);

            // negative index in 0-based parsing.
            path = CreateDataset("-data1.txt", new string[] {
                "1\t-1:3\t4:6",
            });
            data = ML.Data.LoadFromSvmLightFile(path);
            ex = Assert.Throws<InvalidOperationException>(() =>
            {
                using (var curs = data.GetRowCursor(data.Schema["Features"]))
                {
                    var getter = curs.GetGetter<VBuffer<float>>(data.Schema["Features"]);
                    VBuffer<float> features = default;
                    while (curs.MoveNext())
                    {
                        getter(ref features);
                    }
                }
            });
            Assert.Contains("Encountered non-parsable index '-1' while parsing dataset", ex.InnerException.Message);

            // non-parsable index.
            path = CreateDataset("-data2.txt", new string[] {
                "1\ta:3\t4:6",
            });
            data = ML.Data.LoadFromSvmLightFile(path);
            ex = Assert.Throws<InvalidOperationException>(() =>
            {
                using (var curs = data.GetRowCursor(data.Schema["Features"]))
                {
                    var getter = curs.GetGetter<VBuffer<float>>(data.Schema["Features"]);
                    VBuffer<float> features = default;
                    while (curs.MoveNext())
                    {
                        getter(ref features);
                    }
                }
            });
            Assert.Contains("Encountered non-parsable index 'a' while parsing dataset", ex.InnerException.Message);

            // Only non-parsable indices.
            path = CreateDataset("-data3.txt", new string[] {
                "1\ta:3\tb:6",
            });
            ex = Assert.Throws<InvalidOperationException>(() => ML.Data.LoadFromSvmLightFile(path));
            Assert.Contains("No int parsable keys found during key transform inference", ex.Message);
        }

        [Fact]
        public void TestSvmLightLoaderMultiStreamSourceSpecialCases()
        {
            var path1 = CreateDataset("-data1.txt", new string[] {
                "1\t1:3\t4:6",
            });
            var path2 = CreateDataset("-data2.txt", new string[] {
                "1\t1:3\t4:6",
            });
            var loader = ML.Data.CreateSvmLightLoader(inputSize: 4);
            var data = loader.Load(new MultiFileSource(path1, path2));
            using (var curs = data.GetRowCursor(data.Schema["Features"]))
            {
                var getter = curs.GetGetter<VBuffer<float>>(data.Schema["Features"]);
                VBuffer<float> features = default;
                curs.MoveNext();
                getter(ref features);
                curs.MoveNext();
                getter(ref features);
                Assert.False(curs.MoveNext());
            }

            loader = ML.Data.CreateSvmLightLoader(inputSize: 3);
            data = loader.Load(new MultiFileSource(null));
            using (var curs = data.GetRowCursor())
            {
                Assert.False(curs.MoveNext());
            }
        }

        [Fact]
        public void TestSvmLightLoaderNoDataSample()
        {
            var ex = Assert.Throws<InvalidOperationException>(() => ML.Data.CreateSvmLightLoader());
            Assert.Contains("If the number of features is not specified, a dataset must be provided to infer it.", ex.Message);
            ex = Assert.Throws<InvalidOperationException>(() => ML.Data.CreateSvmLightLoaderWithFeatureNames());
            Assert.Contains("To use the text feature names option, a dataset must be provided", ex.Message);
        }

        [Fact]
        public void TestSvmLightLoaderAndSaverTrainOnSubsetOfRows()
        {
            var path = CreateDataset("-data.txt", new string[] {
                "1\t1:3\t4:6",
                "  -1 cost:5\t2:4 \t4:7\t6:-1   ",
                "",
                "1\t5:-2 # A comment! 2:3",
                "1 cost:0.5\t2:3.14159",
                "-1 2:5 11:0.34"
            });

            var schemaDef = SchemaDefinition.Create(typeof(SvmLightOutput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 6);

            var expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 2, new[] { 3f, 6f }, new[] { 0, 3 }) },
                new SvmLightOutput() { Label = -1, Weight = 5, Features = new VBuffer<float>(6, 3, new[] { 4f, 7f, -1f }, new[] { 1, 3, 5 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 1, new[] { -2f }, new[] { 4 }), Comment = " A comment! 2:3".AsMemory() },
                new SvmLightOutput() { Label = 1, Weight = 0.5f, Features = new VBuffer<float>(6, 1, new[] { 3.14159f }, new[] { 1 }) },
                new SvmLightOutput() { Label = -1, Weight = 1, Features = new VBuffer<float>(6, 1, new[] { 5f }, new[] { 1 }) }
            }, schemaDef);
            var savingPath = DeleteOutputPath(TestName + "-saved-data.txt");
            TestSvmLight(path, savingPath, 0, 6, false, expectedData, numberOfRows: 4);
        }

        [Fact]
        public void TestSvmLightLoaderLongIndex()
        {
            var path = CreateDataset("-data.txt", new string[] {
                "1\t1:3\t4:6",
                "  -1 cost:5\t2:4 \t4:7\t6:-1   ",
                "",
                "1\t5:-2 # A comment! 2:3",
                "1 cost:0.5\t2:3.14159",
                $"-1 2:5 {(long)int.MaxValue + 2}:0.34"
            });

            var schemaDef = SchemaDefinition.Create(typeof(SvmLightOutput));
            schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 6);

            var expectedData = ML.Data.LoadFromEnumerable(new SvmLightOutput[]
            {
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 2, new[] { 3f, 6f }, new[] { 0, 3 }) },
                new SvmLightOutput() { Label = -1, Weight = 5, Features = new VBuffer<float>(6, 3, new[] { 4f, 7f, -1f }, new[] { 1, 3, 5 }) },
                new SvmLightOutput() { Label = 1, Weight = 1, Features = new VBuffer<float>(6, 1, new[] { -2f }, new[] { 4 }), Comment = " A comment! 2:3".AsMemory() },
                new SvmLightOutput() { Label = 1, Weight = 0.5f, Features = new VBuffer<float>(6, 1, new[] { 3.14159f }, new[] { 1 }) },
                new SvmLightOutput() { Label = -1, Weight = 1, Features = new VBuffer<float>(6, 1, new[] { 5f }, new[] { 1 }) }
            }, schemaDef);
            var savingPath = DeleteOutputPath(TestName + "-saved-data.txt");
            TestSvmLight(path, savingPath, 0, 6, false, expectedData);
        }

        [Fact]
        public void TestSvmLightSaverBadInputSchema()
        {
            var loader = ML.Data.CreateTextLoader(new[] { new TextLoader.Column("Column", DataKind.Single, 0) });
            var ex = Assert.Throws<InvalidOperationException>(() =>
            {
                var path = DeleteOutputPath(TestName + "-no-label.txt");
                using (var stream = new FileStream(path, FileMode.Create))
                    ML.Data.SaveInSvmLightFormat(loader.Load(new MultiFileSource(null)), stream);
            });
            Assert.Contains("Column Label not found in data", ex.Message);

            ex = Assert.Throws<InvalidOperationException>(() =>
            {
                var path = DeleteOutputPath(TestName + "-no-features.txt");
                using (var stream = new FileStream(path, FileMode.Create))
                    ML.Data.SaveInSvmLightFormat(loader.Load(new MultiFileSource(null)), stream, labelColumnName: "Column");
            });
            Assert.Contains("Column Features not found in data", ex.Message);

            ex = Assert.Throws<InvalidOperationException>(() =>
            {
                var path = DeleteOutputPath(TestName + "-no-group.txt");
                using (var stream = new FileStream(path, FileMode.Create))
                    ML.Data.SaveInSvmLightFormat(loader.Load(new MultiFileSource(null)), stream, labelColumnName: "Column", featureColumnName: "Column", rowGroupColumnName: "Group");
            });
            Assert.Contains("Column Group not found in data", ex.Message);

            ex = Assert.Throws<InvalidOperationException>(() =>
            {
                var path = DeleteOutputPath(TestName + "-no-weight.txt");
                using (var stream = new FileStream(path, FileMode.Create))
                    ML.Data.SaveInSvmLightFormat(loader.Load(new MultiFileSource(null)), stream, labelColumnName: "Column", featureColumnName: "Column", exampleWeightColumnName: "Weight");
            });
            Assert.Contains("Column Weight not found in data", ex.Message);
        }
    }
}
