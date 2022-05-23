// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    [Collection("NoParallelization")]
    public class SentenceClassificationTests : TestDataPipeBase
    {
        public SentenceClassificationTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestSingleSentenceData
        {
            public string Sentence1;
            public long Label;
        }

        private class TestDoubleSentenceData
        {
            public string Sentence;
            public string Sentence2;
            public long Label;
        }

        [TorchSharpFact]
        public void TestSingleSentence2Classes()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new TestSingleSentenceData()
                    {
                        Sentence1 = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Label = 0
                    },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "with a sharp script and strong performances",
                         Label = 1
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "that director m. night shyamalan can weave an eerie spell and",
                         Label = 1
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "comfortable",
                         Label = 1
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "does have its charms .",
                         Label = 1
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "banal as the telling",
                         Label = 0
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Label = 0
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "leguizamo 's best movie work so far",
                         Label = 0
                     }
                }));

            var estimator = ML.MulticlassClassification.Trainers.SentenceClassification(outputColumnName: "outputColumn");

            TestEstimatorCore(estimator, dataView);
            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            Assert.Equal(3, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[2].Name);
            Assert.Equal(NumberDataViewType.Double, estimatorSchema[2].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(3, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[2].Name);
            Assert.Equal(NumberDataViewType.Double, transformerSchema[2].Type);

            var transformedData = transformer.Transform(dataView).Preview();

            // Not enough training is done to get good results so all are 0
            Assert.True(transformedData.ColumnView[2].Values.All(value => (double)value == 0D));
        }

        [TorchSharpFact]
        public void TestSingleSentence3Classes()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new TestSingleSentenceData()
                    {
                        Sentence1 = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Label = 0
                    },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "with a sharp script and strong performances",
                         Label = 1
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "that director m. night shyamalan can weave an eerie spell and",
                         Label = 2
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "comfortable",
                         Label = 0
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "does have its charms .",
                         Label = 1
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "banal as the telling",
                         Label = 2
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Label = 0
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "leguizamo 's best movie work so far",
                         Label = 1
                     }
                }));

            var estimator = ML.MulticlassClassification.Trainers.SentenceClassification(outputColumnName: "outputColumn", numberOfClasses: 3);

            TestEstimatorCore(estimator, dataView);
            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            Assert.Equal(3, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[2].Name);
            Assert.Equal(NumberDataViewType.Double, estimatorSchema[2].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(3, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[2].Name);
            Assert.Equal(NumberDataViewType.Double, transformerSchema[2].Type);

            var transformedData = transformer.Transform(dataView).Preview();

            Assert.NotNull(transformedData);
            Assert.Equal(0D, (double)transformedData.ColumnView[2].Values[0]);
            Assert.Equal(1D, (double)transformedData.ColumnView[2].Values[1]);
            Assert.Equal(0D, (double)transformedData.ColumnView[2].Values[2]);
            Assert.Equal(2D, (double)transformedData.ColumnView[2].Values[3]);
            Assert.Equal(2D, (double)transformedData.ColumnView[2].Values[4]);
            Assert.Equal(1D, (double)transformedData.ColumnView[2].Values[5]);
            Assert.Equal(1D, (double)transformedData.ColumnView[2].Values[6]);
            Assert.Equal(2D, (double)transformedData.ColumnView[2].Values[7]);
        }

        [TorchSharpFact]
        public void TestDoubleSentence2Classes()
        {
            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestDoubleSentenceData>(new TestDoubleSentenceData[] {
                    new ()
                    {
                        Sentence = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Sentence2 = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Label = 0
                    },
                     new ()
                     {
                         Sentence = "with a sharp script and strong performances",
                         Sentence2 = "with a sharp script and strong performances",
                         Label = 1
                     },
                     new ()
                     {
                         Sentence = "that director m. night shyamalan can weave an eerie spell and",
                         Sentence2 = "that director m. night shyamalan can weave an eerie spell and",
                         Label = 1
                     },
                     new ()
                     {
                         Sentence = "comfortable",
                         Sentence2 = "comfortable",
                         Label = 1
                     },
                     new ()
                     {
                         Sentence = "does have its charms .",
                         Sentence2 = "does have its charms .",
                         Label = 1
                     },
                     new ()
                     {
                         Sentence = "banal as the telling",
                         Sentence2 = "banal as the telling",
                         Label = 1
                     },
                     new ()
                     {
                         Sentence = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Sentence2 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Label = 0
                     },
                     new ()
                     {
                         Sentence = "leguizamo 's best movie work so far",
                         Sentence2 = "leguizamo 's best movie work so far",
                         Label = 1
                     }
                }));

            var estimator = ML.MulticlassClassification.Trainers.SentenceClassification(outputColumnName: "outputColumn", sentence1ColumnName: "Sentence", sentence2ColumnName: "Sentence2");

            TestEstimatorCore(estimator, dataView);
            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));

            Assert.Equal(4, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[3].Name);
            Assert.Equal(NumberDataViewType.Double, estimatorSchema[3].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(4, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[3].Name);
            Assert.Equal(NumberDataViewType.Double, transformerSchema[3].Type);

            var transformedData = transformer.Transform(dataView).Preview();

            // Not enough training is done to get good results so all are 0
            Assert.NotNull(transformedData);
            Assert.True(transformedData.ColumnView[3].Values.All(value => (double)value == 0D));
        }
    }

}
