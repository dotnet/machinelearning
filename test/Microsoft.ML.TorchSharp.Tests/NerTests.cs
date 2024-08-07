// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.TorchSharp.Tests
{
    public class NerTests : TestDataPipeBase
    {
        public NerTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestSingleSentenceData
        {
            public string Sentence;
            public string[] Label;
        }

        private class Label
        {
            public string Key { get; set; }
        }

        [Fact]
        public void TestSimpleNer()
        {
            var labels = ML.Data.LoadFromEnumerable(
                new[] {
                new Label { Key = "PERSON" },
                new Label { Key = "CITY" },
                new Label { Key = "COUNTRY"  },
                new Label { Key = "B_WORK_OF_ART"  },
                new Label { Key = "WORK_OF_ART"  },
                new Label { Key = "B_NORP"  },
                });

            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new()
                    {
                        Sentence = "Alice and Bob live in the liechtenstein",
                        Label = new string[]{"PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY" }
                    },
                     new()
                     {
                        Sentence = "Alice and Bob live in the USA",
                        Label = new string[]{"PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY"}
                     },
                     new()
                     {
                         Sentence = "WW II Landmarks on the Great Earth of China : Eternal Memories of Taihang Mountain",
                         Label = new string[]{"B_WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART" }
                     },
                     new()
                     {
                         Sentence = "This campaign broke through the Japanese army 's blockade to reach base areas behind enemy lines , stirring up anti-Japanese spirit throughout the nation and influencing the situation of the anti-fascist war of the people worldwide .",
                         Label = new string[]{"0", "0", "0", "0", "0", "B_NORP", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "B_NORP", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0" }
                     }
                }));
            var chain = new EstimatorChain<ITransformer>();
            var estimator = chain.Append(ML.Transforms.Conversion.MapValueToKey("Label", keyData: labels))
               .Append(ML.MulticlassClassification.Trainers.NamedEntityRecognition(outputColumnName: "outputColumn"))
               .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            Assert.Equal(3, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[2].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[2].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(5, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[4].Name);

            var output = transformer.Transform(dataView);
            using (var cursor = output.GetRowCursorForAllColumns())
            {

                var labelGetter = cursor.GetGetter<VBuffer<uint>>(output.Schema[2]);
                var predictedLabelGetter = cursor.GetGetter<VBuffer<uint>>(output.Schema[3]);

                VBuffer<uint> labelData = default;
                VBuffer<uint> predictedLabelData = default;

                while (cursor.MoveNext())
                {
                    labelGetter(ref labelData);
                    predictedLabelGetter(ref predictedLabelData);

                    // Make sure that the expected label and the predicted label have same length
                    Assert.Equal(labelData.Length, predictedLabelData.Length);
                }
            }

            TestEstimatorCore(estimator, dataView, shouldDispose: true);
            transformer.Dispose();
        }

        [Fact]
        public void TestSimpleNerOptions()
        {
            var labels = ML.Data.LoadFromEnumerable(
                new[] {
                new Label { Key = "PERSON" },
                new Label { Key = "CITY" },
                new Label { Key = "COUNTRY"  },
                new Label { Key = "B_WORK_OF_ART"  },
                new Label { Key = "WORK_OF_ART"  },
                new Label { Key = "B_NORP"  },
                });

            var options = new NerTrainer.NerOptions();
            options.PredictionColumnName = "outputColumn";

            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new()
                    {
                        Sentence = "Alice and Bob live in the liechtenstein",
                        Label = new string[]{"PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY" }
                    },
                     new()
                     {
                        Sentence = "Alice and Bob live in the USA",
                        Label = new string[]{"PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY"}
                     },
                     new()
                     {
                         Sentence = "WW II Landmarks on the Great Earth of China : Eternal Memories of Taihang Mountain",
                         Label = new string[]{"B_WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART", "WORK_OF_ART" }
                     },
                     new()
                     {
                         Sentence = "This campaign broke through the Japanese army 's blockade to reach base areas behind enemy lines , stirring up anti-Japanese spirit throughout the nation and influencing the situation of the anti-fascist war of the people worldwide .",
                         Label = new string[]{"0", "0", "0", "0", "0", "B_NORP", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "B_NORP", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0" }
                     }
                }));
            var chain = new EstimatorChain<ITransformer>();
            var estimator = chain.Append(ML.Transforms.Conversion.MapValueToKey("Label", keyData: labels))
               .Append(ML.MulticlassClassification.Trainers.NamedEntityRecognition(options))
               .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            Assert.Equal(3, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[2].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[2].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(5, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[4].Name);

            var output = transformer.Transform(dataView);
            using (var cursor = output.GetRowCursorForAllColumns())
            {

                var labelGetter = cursor.GetGetter<VBuffer<uint>>(output.Schema[2]);
                var predictedLabelGetter = cursor.GetGetter<VBuffer<uint>>(output.Schema[3]);

                VBuffer<uint> labelData = default;
                VBuffer<uint> predictedLabelData = default;

                while (cursor.MoveNext())
                {
                    labelGetter(ref labelData);
                    predictedLabelGetter(ref predictedLabelData);

                    // Make sure that the expected label and the predicted label have same length
                    Assert.Equal(labelData.Length, predictedLabelData.Length);
                }
            }

            TestEstimatorCore(estimator, dataView, shouldDispose: true);
            transformer.Dispose();
        }

        [Fact]
        public void TestNERLargeFileGpu()
        {
            ML.FallbackToCpu = false;
            ML.GpuDeviceId = 0;

            var labelFilePath = GetDataPath("ner-key-info.txt");
            var labels = ML.Data.LoadFromTextFile(labelFilePath, new TextLoader.Column[]
                {
                    new TextLoader.Column("Key", DataKind.String, 0)
                }
            );

            var dataFilePath = GetDataPath("ner-conll2012_english_v4_train.txt");
            var dataView = TextLoader.Create(ML, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("Sentence", DataKind.String, 0),
                    new TextLoader.Column("Label", DataKind.String, new TextLoader.Range[]
                    {
                        new TextLoader.Range(1, null) { VariableEnd = true, AutoEnd = false }
                    })
                },
                HasHeader = false,
                Separators = new char[] { '\t' },
                MaxRows = 75187 // Dataset has 75187 rows. Only load 1k for quicker training,
            }, new MultiFileSource(dataFilePath));

            var trainTest = ML.Data.TrainTestSplit(dataView);

            var options = new NerTrainer.NerOptions();
            options.PredictionColumnName = "outputColumn";

            var chain = new EstimatorChain<ITransformer>();
            var estimator = chain.Append(ML.Transforms.Conversion.MapValueToKey("Label", keyData: labels))
               .Append(ML.MulticlassClassification.Trainers.NamedEntityRecognition(options))
               .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            Assert.Equal(3, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[2].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[2].ItemType);

            var transformer = estimator.Fit(trainTest.TrainSet);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            var output = transformer.Transform(trainTest.TestSet);
            using var cursor = output.GetRowCursorForAllColumns();

            var labelGetter = cursor.GetGetter<VBuffer<uint>>(output.Schema[2]);
            var predictedLabelGetter = cursor.GetGetter<VBuffer<uint>>(output.Schema[3]);

            VBuffer<uint> labelData = default;
            VBuffer<uint> predictedLabelData = default;

            double correct = 0;
            double total = 0;

            while (cursor.MoveNext())
            {
                labelGetter(ref labelData);
                predictedLabelGetter(ref predictedLabelData);

                Assert.Equal(labelData.Length, predictedLabelData.Length);

                for (var i = 0; i < labelData.Length; i++)
                {
                    if (labelData.GetItemOrDefault(i) == predictedLabelData.GetItemOrDefault(i) || (labelData.GetItemOrDefault(i) == default && predictedLabelData.GetItemOrDefault(i) == 0))
                        correct++;
                    total++;
                }
            }
            Assert.True(correct / total > .80);
            Assert.Equal(5, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[4].Name);

            transformer.Dispose();
        }
    }
}
