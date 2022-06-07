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
            public string Label;
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
                    {
                        Sentence1 = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Label = "Negative"
                    },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "with a sharp script and strong performances",
                         Label = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "that director m. night shyamalan can weave an eerie spell and",
                         Label = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "comfortable",
                         Label = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "does have its charms .",
                         Label = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "banal as the telling",
                         Label = "Negative"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Label = "Negative"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "leguizamo 's best movie work so far",
                         Label = "Negative"
                     }
                }));
            var estimator = ML.Transforms.Conversion.MapValueToKey("Label")
                .Append(ML.MulticlassClassification.Trainers.SentenceClassification(outputColumnName: "outputColumn"))
                .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            TestEstimatorCore(estimator, dataView);
            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            Assert.Equal(3, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[2].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[2].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(5, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[4].Name);
            Assert.Equal(TextDataViewType.Instance, transformerSchema[4].Type);

            var transformedData = transformer.Transform(dataView).Preview();

            // Not enough training is done to get good results so all are 0
            Assert.True(transformedData.ColumnView[4].Values.All(value => value.ToString() == "Negative"));
        }

        [Fact]
        public void TestSingleSentence3Classes()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new TestSingleSentenceData()
                    {
                        Sentence1 = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Label = "Class One"
                    },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "with a sharp script and strong performances",
                         Label = "Class Two"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "that director m. night shyamalan can weave an eerie spell and",
                         Label = "Class Three"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "comfortable",
                         Label = "Class One"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "does have its charms .",
                         Label = "Class Two"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "banal as the telling",
                         Label = "Class Three"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Label = "Class One"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "leguizamo 's best movie work so far",
                         Label = "Class Two"
                     }
                }));

            var estimator = ML.Transforms.Conversion.MapValueToKey("Label")
                .Append(ML.MulticlassClassification.Trainers.SentenceClassification(outputColumnName: "outputColumn", numberOfClasses: 3))
                .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            TestEstimatorCore(estimator, dataView);
            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            Assert.Equal(3, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[2].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[2].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(5, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[4].Name);
            Assert.Equal(TextDataViewType.Instance, transformerSchema[4].Type);

            var transformedData = transformer.Transform(dataView).Preview();

            Assert.NotNull(transformedData);
#if NET461
            Assert.Equal("Class Three", transformedData.ColumnView[4].Values[0].ToString());
            Assert.Equal("Class Two", transformedData.ColumnView[4].Values[1].ToString());
            Assert.Equal("Class Three", transformedData.ColumnView[4].Values[2].ToString());
            Assert.Equal("Class Three", transformedData.ColumnView[4].Values[6].ToString());
#else
            Assert.Equal("Class One", transformedData.ColumnView[4].Values[0].ToString());
            Assert.Equal("Class Two", transformedData.ColumnView[4].Values[1].ToString());
            Assert.Equal("Class One", transformedData.ColumnView[4].Values[2].ToString());
            Assert.Equal("Class Two", transformedData.ColumnView[4].Values[6].ToString());

#endif
            Assert.Equal("Class Three", transformedData.ColumnView[4].Values[3].ToString());
            Assert.Equal("Class Three", transformedData.ColumnView[4].Values[4].ToString());
            Assert.Equal("Class Two", transformedData.ColumnView[4].Values[5].ToString());
            Assert.Equal("Class Three", transformedData.ColumnView[4].Values[7].ToString());
        }

        [Fact]
        public void TestDoubleSentence2Classes()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestDoubleSentenceData>(new TestDoubleSentenceData[] {
                    new ()
                    {
                        Sentence = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Sentence2 = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Label = "Class One"
                    },
                     new ()
                     {
                         Sentence = "with a sharp script and strong performances",
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

            var estimator = ML.MulticlassClassification.Trainers.SentenceClassification(outputColumnName: "outputColumn", sentence1ColumnName: "Sentence", sentence2ColumnName: "Sentence2", validationSet: preppedData)
                .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            TestEstimatorCore(estimator, preppedData);
            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(preppedData.Schema));

            Assert.Equal(4, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[3].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[3].ItemType);

            var transformer = estimator.Fit(preppedData);
            var transformerSchema = transformer.GetOutputSchema(preppedData.Schema);

            Assert.Equal(6, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[5].Name);
            Assert.Equal(TextDataViewType.Instance, transformerSchema[5].Type);

            var transformedData = transformer.Transform(preppedData).Preview();

            // Not enough training is done to get good results so all are 0
            Assert.NotNull(transformedData);
            Assert.True(transformedData.ColumnView[5].Values.All(value => value.ToString() == "Class One"));
        }
    }

}
