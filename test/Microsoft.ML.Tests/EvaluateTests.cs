// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class EvaluateTests : BaseTestClass
    {
        public EvaluateTests(ITestOutputHelper output)
            : base(output)
        {
        }

        public class MulticlassEvaluatorInput
        {
            [KeyType(4)]
            public uint Label { get; set; }

            [VectorType(4)]
            public float[] Score { get; set; }

            [KeyType(4)]
            public uint PredictedLabel { get; set; }
        }

        [Fact]
        public void MulticlassEvaluatorTopKArray()
        {
            var mlContext = new MLContext(seed: 1);

            // Notice that the probability assigned to the correct label (i.e. Score[0])
            // decreases on each row so as to get the expected TopK accuracy array hardcoded below.
            var inputArray = new[]
            {
                new MulticlassEvaluatorInput{Label = 0, Score = new[] {0.4f, 0.3f, 0.2f, 0.1f }, PredictedLabel=0},
                new MulticlassEvaluatorInput{Label = 0, Score = new[] {0.3f, 0.4f, 0.2f, 0.1f }, PredictedLabel=1},
                new MulticlassEvaluatorInput{Label = 0, Score = new[] {0.2f, 0.3f, 0.4f, 0.1f }, PredictedLabel=2},
                new MulticlassEvaluatorInput{Label = 0, Score = new[] {0.1f, 0.3f, 0.2f, 0.4f }, PredictedLabel=3}
            };

            var expectedTopKArray = new[] { 0.25d, 0.5d, 0.75d, 1.0d };

            var inputDV = mlContext.Data.LoadFromEnumerable(inputArray);
            var metrics = mlContext.MulticlassClassification.Evaluate(inputDV, topKPredictionCount: 4);

            Assert.Equal(expectedTopKArray, metrics.TopKAccuracyForAllK);
        }
    }
}
