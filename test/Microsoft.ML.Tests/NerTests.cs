// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    [Collection("NoParallelization")]
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
                new Label { Key = "COUNTRY"  }
                });

            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new TestSingleSentenceData()
                    {   // Testing longer than 512 words.
                        Sentence = "Alice and Bob live in the USA",
                        Label = new string[]{"PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY"}
                    },
                     new TestSingleSentenceData()
                     {
                        Sentence = "Alice and Bob live in the USA",
                        Label = new string[]{"PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY"}
                     },
                }));
            var chain = new EstimatorChain<ITransformer>();
            var estimator = chain.Append(ML.Transforms.Conversion.MapValueToKey("Label", keyData: labels))
               .Append(ML.MulticlassClassification.Trainers.NameEntityRecognition(outputColumnName: "outputColumn"))
               .Append(ML.Transforms.Conversion.MapKeyToValue("outputColumn"));

            var estimatorSchema = estimator.GetOutputSchema(SchemaShape.Create(dataView.Schema));
            var prev = estimator.Fit(dataView).Transform(dataView).Preview();
            Assert.Equal(3, estimatorSchema.Count);
            Assert.Equal("outputColumn", estimatorSchema[2].Name);
            Assert.Equal(TextDataViewType.Instance, estimatorSchema[2].ItemType);

            var transformer = estimator.Fit(dataView);
            var transformerSchema = transformer.GetOutputSchema(dataView.Schema);

            Assert.Equal(5, transformerSchema.Count);
            Assert.Equal("outputColumn", transformerSchema[4].Name);

            TestEstimatorCore(estimator, dataView, shouldDispose: true);
            transformer.Dispose();
        }
    }
}
