// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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

        private class TestDoubleSentenceData
        {
            public string Sentence;
            public string Sentence2;
            public string Label;
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

            var predictedLabel = transformer.Transform(dataView).GetColumn<ReadOnlyMemory<char>>(transformerSchema[4].Name);

            // Make sure that we can use the multiclass evaluate method
            var metrics = ML.MulticlassClassification.Evaluate(transformer.Transform(dataView), predictedLabelColumnName: "outputColumn");
            Assert.NotNull(metrics);

            // Not enough training is done to get good results so just make sure the count is right and are negative.
            var a = predictedLabel.ToList();
            Assert.Equal(8, a.Count());
            Assert.True(predictedLabel.All(value => value.ToString() == "Negative"));
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
#if NET461
            Assert.Equal("Class One", transformedData.ColumnView[4].Values[0].ToString());
            Assert.Equal("Class Two", transformedData.ColumnView[4].Values[1].ToString());
            Assert.Equal("Class Three", transformedData.ColumnView[4].Values[2].ToString());
            Assert.Equal("Class Three", transformedData.ColumnView[4].Values[4].ToString());
            Assert.Equal("Class One", transformedData.ColumnView[4].Values[6].ToString());
#else
            Assert.Equal("Class One", transformedData.ColumnView[4].Values[0].ToString());
            Assert.Equal("Class Two", transformedData.ColumnView[4].Values[1].ToString());
            Assert.Equal("Class Three", transformedData.ColumnView[4].Values[2].ToString());
            Assert.Equal("Class One", transformedData.ColumnView[4].Values[4].ToString());
            Assert.Equal("Class One", transformedData.ColumnView[4].Values[6].ToString());

#endif
            Assert.Equal("Class One", transformedData.ColumnView[4].Values[3].ToString());
            Assert.Equal("Class Two", transformedData.ColumnView[4].Values[7].ToString());
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
        }
    }

}
