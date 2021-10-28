// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
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
            public float Label { get; set; }

            [VectorType(4)]
            public float[] Score { get; set; }

            public float PredictedLabel { get; set; }
        }

        [Fact]
        public void MulticlassEvaluatorTopKArray()
        {
            var mlContext = new MLContext(seed: 1);

            // Notice that the probability assigned to the correct label (i.e. Score[0])
            // decreases on each row so as to get the expected TopK accuracy array hardcoded below.
            var inputArray = new[]
            {
                new MulticlassEvaluatorInput{Label = 0, Score = new[] {0.4f, 0.3f, 0.2f, 0.1f}, PredictedLabel = 0},
                new MulticlassEvaluatorInput{Label = 0, Score = new[] {0.3f, 0.4f, 0.2f, 0.1f}, PredictedLabel = 1},
                new MulticlassEvaluatorInput{Label = 0, Score = new[] {0.2f, 0.3f, 0.4f, 0.1f}, PredictedLabel = 2},
                new MulticlassEvaluatorInput{Label = 0, Score = new[] {0.1f, 0.3f, 0.2f, 0.4f}, PredictedLabel = 3},
            };

            var expectedTopKArray = new[] { 0.25d, 0.5d, 0.75d, 1.0d };

            var inputDV = mlContext.Data.LoadFromEnumerable(inputArray);
            var metrics = mlContext.MulticlassClassification.Evaluate(inputDV, topKPredictionCount: 4);
            Assert.Equal(expectedTopKArray, metrics.TopKAccuracyForAllK);


            // After introducing a sample whose label was unseen (i.e. the Score array doesn't assign it a probability)
            // then the Top K array changes, as its values are divided by the total number of instances
            // that were evaluated.
            var inputArray2 = inputArray.AppendElement(new MulticlassEvaluatorInput
            {
                Label = 5,
                Score = new[] { 0.1f, 0.3f, 0.2f, 0.4f },
                PredictedLabel = 3
            });

            var expectedTopKArray2 = new[] { 0.2d, 0.4d, 0.6d, 0.8d };

            var inputDV2 = mlContext.Data.LoadFromEnumerable(inputArray2);
            var metrics2 = mlContext.MulticlassClassification.Evaluate(inputDV2, topKPredictionCount: 4);
            var output2 = metrics2.TopKAccuracyForAllK.ToArray();
            for (int i = 0; i < expectedTopKArray2.Length; i++)
                Assert.Equal(expectedTopKArray2[i], output2[i], precision: 7);
        }
    }
}
