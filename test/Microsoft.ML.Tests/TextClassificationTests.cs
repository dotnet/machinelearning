// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Apache.Arrow;
using ICSharpCode.SharpZipLib.Tar;
using Microsoft.Data.Analysis;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    [Collection("NoParallelization")]
    public class TextClassificationTests : TestDataPipeBase
    {
        public TextClassificationTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestSingleSentenceData
        {
            public string Sentence1;
            public string Sentiment;
        }

        private class TestSingleSentenceDataNoLabel
        {
            public string Sentence1;
        }

        private class TestDoubleSentenceData
        {
            public string Sentence;
            public string Sentence2;
            public string Label;
        }

        private class TestSentenceSimilarityData
        {
            public string Sentence;
            public string Sentence2;
            public float Label;
        }

        [Fact]
        public void TestSingleSentence2Classes()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new TestSingleSentenceData()
                    {   // Testing longer than 512 words.
                        Sentence1 = "ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .",
                        Sentiment = "Negative"
                    },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "with a sharp script and strong performances",
                         Sentiment = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "that director m. night shyamalan can weave an eerie spell and",
                         Sentiment = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "comfortable",
                         Sentiment = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "does have its charms .",
                         Sentiment = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "banal as the telling",
                         Sentiment = "Negative"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Sentiment = "Negative"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "leguizamo 's best movie work so far",
                         Sentiment = "Negative"
                     }
                }));
            var chain = new EstimatorChain<ITransformer>();
            var estimator = chain.Append(ML.Transforms.Conversion.MapValueToKey("Label", "Sentiment"), TransformerScope.TrainTest)
                .Append(ML.MulticlassClassification.Trainers.TextClassification(outputColumnName: "outputColumn"))
                .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            Assert.Equal(5, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[3].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[3].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            var filteredModel = transformer.GetModelFor(TransformerScope.Scoring);

            Assert.Equal(5, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[3].Name);
            Assert.Equal(TextDataViewType.Instance, transformerSchema[3].Type);

            var dataNoLabel = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceDataNoLabel>(new TestSingleSentenceDataNoLabel[] {
                    new ()
                    {   // Testing longer than 512 words.
                        Sentence1 = "ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .",
                    },
                     new ()
                     {
                         Sentence1 = "with a sharp script and strong performances",
                     },
                     new ()
                     {
                         Sentence1 = "that director m. night shyamalan can weave an eerie spell and",
                     },
                     new ()
                     {
                         Sentence1 = "comfortable",
                     },
                     new ()
                     {
                         Sentence1 = "does have its charms .",
                     },
                     new ()
                     {
                         Sentence1 = "banal as the telling",
                     },
                     new ()
                     {
                         Sentence1 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                     },
                     new ()
                     {
                         Sentence1 = "leguizamo 's best movie work so far",
                     }
                }));

            var predictedLabel = filteredModel.Transform(dataNoLabel).GetColumn<ReadOnlyMemory<char>>(transformerSchema[3].Name);

            // Make sure that we can use the multiclass evaluate method
            var metrics = ML.MulticlassClassification.Evaluate(transformer.Transform(dataView, TransformerScope.Everything), predictedLabelColumnName: "outputColumn");
            Assert.NotNull(metrics);

            // Not enough training is done to get good results so just make sure the count is right.
            var a = predictedLabel.ToList();
            Assert.Equal(8, a.Count());
        }

        // To run the TestTextClassificationWithBigDataOnGpu, set the EnableRunningGpuTest property to true and in the csproj enable the package TorchSharp-cuda-windows and disable libtorch-cpu-win-x64.
        private static bool EnableRunningGpuTest => false;

        [ConditionalFact(nameof(EnableRunningGpuTest))]
        public void TestTextClassificationWithBigDataOnGpu()
        {
            var mlContext = new MLContext();
            mlContext.GpuDeviceId = 0;
            mlContext.FallbackToCpu = false;
            var df = DataFrame.LoadCsv(@"Data\github-issues-train.tsv", separator: '\t', header: true, columnNames: new[] { "ID", "Label", "Title", "Description" });
            var trainTestSplit = mlContext.Data.TrainTestSplit(df, testFraction: 0.2);
            var pipeline =
                    mlContext.Transforms.Conversion.MapValueToKey("Label")
                        .Append(mlContext.MulticlassClassification.Trainers.TextClassification(sentence1ColumnName: "Title", sentence2ColumnName: "Description", maxEpochs: 10, batchSize: 8))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var model = pipeline.Fit(trainTestSplit.TrainSet);
            var predictionIdv = model.Transform(trainTestSplit.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictionIdv);
            Assert.True(metrics.MacroAccuracy > .69);
            Assert.True(metrics.MicroAccuracy > .70);
        }

        [Fact]
        public void TestSingleSentence3Classes()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new TestSingleSentenceData()
                    {
                        Sentence1 = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Sentiment = "Class One"
                    },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "with a sharp script and strong performances",
                         Sentiment = "Class Two"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "that director m. night shyamalan can weave an eerie spell and",
                         Sentiment = "Class Three"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "comfortable",
                         Sentiment = "Class One"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "does have its charms .",
                         Sentiment = "Class Two"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "banal as the telling",
                         Sentiment = "Class Three"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Sentiment = "Class One"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "leguizamo 's best movie work so far",
                         Sentiment = "Class Two"
                     }
                }));

            var estimator = ML.Transforms.Conversion.MapValueToKey("Label", "Sentiment")
                .Append(ML.MulticlassClassification.Trainers.TextClassification(outputColumnName: "outputColumn"))
                .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            TestEstimatorCore(estimator, dataView);
            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            Assert.Equal(5, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[3].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[3].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(6, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[4].Name);
            Assert.Equal(TextDataViewType.Instance, transformerSchema[4].Type);

            var transformedData = transformer.Transform(dataView).Preview();

            Assert.NotNull(transformedData);
            // Not enough training is done to get good results so just make sure the count is right.
            Assert.Equal(8, transformedData.RowView.Count());
        }

        [Fact]
        public void TestDoubleSentence2Classes()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestDoubleSentenceData>(new TestDoubleSentenceData[] {
                    new ()
                    {   //Testing longer than 512 words.
                        Sentence = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Sentence2 = "ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .",
                        Label = "Class One"
                    },
                     new ()
                     {   //Testing longer than 512 words.
                         Sentence = "ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .",
                         Sentence2 = "with a sharp script and strong performances",
                         Label = "Class Two"
                     },
                     new ()
                     {
                         Sentence = "that director m. night shyamalan can weave an eerie spell and",
                         Sentence2 = "that director m. night shyamalan can weave an eerie spell and",
                         Label = "Class Two"
                     },
                     new ()
                     {
                         Sentence = "comfortable",
                         Sentence2 = "comfortable",
                         Label = "Class Two"
                     },
                     new ()
                     {
                         Sentence = "does have its charms .",
                         Sentence2 = "does have its charms .",
                         Label = "Class Two"
                     },
                     new ()
                     {
                         Sentence = "banal as the telling",
                         Sentence2 = "banal as the telling",
                         Label = "Class Two"
                     },
                     new ()
                     {
                         Sentence = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Sentence2 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Label = "Class One"
                     },
                     new ()
                     {
                         Sentence = "leguizamo 's best movie work so far",
                         Sentence2 = "leguizamo 's best movie work so far",
                         Label = "Class Two"
                     }
                }));

            var dataPrep = ML.Transforms.Conversion.MapValueToKey("Label");
            var dataPrepTransformer = dataPrep.Fit(dataView);
            var preppedData = dataPrepTransformer.Transform(dataView);

            var estimator = ML.MulticlassClassification.Trainers.TextClassification(outputColumnName: "outputColumn", sentence1ColumnName: "Sentence", sentence2ColumnName: "Sentence2", validationSet: preppedData)
                .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            TestEstimatorCore(estimator, preppedData);
            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(preppedData.Schema));

            Assert.Equal(5, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[3].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[3].ItemType);

            var transformer = estimator.Fit(preppedData);
            var transformerSchema = transformer.GetOutputSchema(preppedData.Schema);

            Assert.Equal(7, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[5].Name);
            Assert.Equal(TextDataViewType.Instance, transformerSchema[5].Type);

            var predictedLabel = transformer.Transform(preppedData).GetColumn<ReadOnlyMemory<char>>(transformerSchema[5].Name);
            // Not enough training is done to get good results so just make sure there is the correct number.
            Assert.NotNull(predictedLabel);
            Assert.Equal(8, predictedLabel.Count());
        }

        [Fact]
        public void TestSentenceSimilarity()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSentenceSimilarityData>(new TestSentenceSimilarityData[] {
                     new ()
                     {
                         Sentence = "Two females jump off of swings.",
                         Sentence2 = "Two females jump off of swings.",
                         Label = 1
                     },
                     new ()
                     {
                         Sentence = "Avengers sets box office record",
                         Sentence2 = "The Hunger Games breaks US box office record",
                         Label = .24f
                     },
                     new ()
                     {
                        Sentence = "A plane is taking off.",
                        Sentence2 = "An air plane is taking off.",
                        Label = 1
                     },
                     new ()
                     {
                        Sentence = "A man is playing a large flute.",
                        Sentence2 = "A man is playing a flute.",
                        Label = .75f
                     },
                     new ()
                     {
                        Sentence = "A man is smoking.",
                        Sentence2 = "A man is skating.",
                        Label = .1f
                     },
                     new ()
                     {
                        Sentence = "The man drove his little red car around the traffic.",
                        Sentence2 = "The dog ran in the water at the beach.",
                        Label = 0
                     }
                }));

            var estimator = ML.Regression.Trainers.SentenceSimilarity(sentence1ColumnName: "Sentence", sentence2ColumnName: "Sentence2");

            TestEstimatorCore(estimator, dataView);
            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            Assert.Equal(4, estimatorSchema.Count);
            Assert.Equal("Score", estimatorSchema[3].Name);
            Assert.Equal(NumberDataViewType.Single, estimatorSchema[3].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(4, transformerSchema.Count);
            Assert.Equal("Score", estimatorSchema[3].Name);
            Assert.Equal(NumberDataViewType.Single, estimatorSchema[3].ItemType);

            var score = transformer.Transform(dataView).GetColumn<float>(transformerSchema[3].Name);
            // Not enough training is done to get good results so just make sure there is the correct number.
            Assert.NotNull(score);
        }
    }

}
