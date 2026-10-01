// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    /// <summary>
    /// Tests for <see cref="PredictionEngineBase{TSrc,TDst}.PredictBatch"/> — the zero-allocation
    /// batch prediction path tracked in issue #6422.
    /// </summary>
    public class PredictionEngineBatchTests : BaseTestClass
    {
        public PredictionEngineBatchTests(ITestOutputHelper output) : base(output)
        {
        }

        private class InputData
        {
            public float Value { get; set; }
        }

        private class OutputData
        {
            public float NormalizedValue { get; set; }
        }

        private static PredictionEngine<InputData, OutputData> BuildEngine(MLContext mlContext)
        {
            var trainingData = mlContext.Data.LoadFromEnumerable(new[]
            {
                new InputData { Value = 1f },
                new InputData { Value = 2f },
                new InputData { Value = 3f },
                new InputData { Value = 4f },
            });

            var pipeline = mlContext.Transforms.CopyColumns("NormalizedValue", "Value");
            var model = pipeline.Fit(trainingData);
            return mlContext.Model.CreatePredictionEngine<InputData, OutputData>(model);
        }

        [Fact]
        public void TestPredictBatch_ReusesOutputObjects()
        {
            var mlContext = new MLContext(seed: 42);
            using var engine = BuildEngine(mlContext);

            var examples = new[] { new InputData { Value = 10f }, new InputData { Value = 20f } };
            var predictions = new[] { new OutputData(), new OutputData() };

            // Capture original references — PredictBatch must NOT replace them
            OutputData ref0 = predictions[0];
            OutputData ref1 = predictions[1];

            engine.PredictBatch(examples, predictions);

            Assert.Same(ref0, predictions[0]);
            Assert.Same(ref1, predictions[1]);
            Assert.Equal(10f, predictions[0].NormalizedValue);
            Assert.Equal(20f, predictions[1].NormalizedValue);
        }

        [Fact]
        public void TestPredictBatch_MatchesSinglePredict()
        {
            var mlContext = new MLContext(seed: 42);
            using var engine = BuildEngine(mlContext);

            var inputs = Enumerable.Range(1, 5).Select(i => new InputData { Value = i * 1.5f }).ToArray();
            var batchOutputs = inputs.Select(_ => new OutputData()).ToArray();

            engine.PredictBatch(inputs, batchOutputs);

            for (int i = 0; i < inputs.Length; i++)
            {
                var singleOutput = engine.Predict(inputs[i]);
                Assert.Equal(singleOutput.NormalizedValue, batchOutputs[i].NormalizedValue);
            }
        }

        [Fact]
        public void TestPredictBatch_WithCount_ProcessesOnlyN()
        {
            var mlContext = new MLContext(seed: 42);
            using var engine = BuildEngine(mlContext);

            var examples = new[]
            {
                new InputData { Value = 1f },
                new InputData { Value = 2f },
                new InputData { Value = 3f },
            };
            var predictions = new[] { new OutputData(), new OutputData(), new OutputData() };
            float sentinel = -999f;
            predictions[2].NormalizedValue = sentinel;

            engine.PredictBatch(examples, predictions, count: 2);

            Assert.Equal(1f, predictions[0].NormalizedValue);
            Assert.Equal(2f, predictions[1].NormalizedValue);
            Assert.Equal(sentinel, predictions[2].NormalizedValue); // untouched
        }

        [Fact]
        public void TestPredictBatch_CountExceedsExamples_Throws()
        {
            var mlContext = new MLContext(seed: 42);
            using var engine = BuildEngine(mlContext);

            var ex = new[] { new InputData { Value = 1f } };
            var pr = new[] { new OutputData(), new OutputData() };

            Assert.Throws<InvalidOperationException>(() => engine.PredictBatch(ex, pr, count: 2));
        }
    }
}
