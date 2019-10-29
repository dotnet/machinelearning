// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.RunTests;
using Microsoft.ML.Trainers;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public sealed class LearningRateSchedulerTest : TestDataPipeBase
    {
        public LearningRateSchedulerTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestPolynomialDecayNoCycle()
        {
            //Values obtained by running tf.compat.v1.train.polynomial_decay on TF 1.14
            float[] expectedValues = new float[] { 0.1f, 0.091f, 0.082f, 0.073f, 0.064f, 0.055f, 0.045999996f, 0.037f,
                0.027999999f, 0.019000001f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f };

            TestPolynomialDecay(expectedValues, false);
        }

        [Fact]
        public void TestPolynomialDecayCycle()
        {
            //Values obtained by running tf.compat.v1.train.polynomial_decay on TF 1.14
            float[] expectedValues = new float[] { 0.1f, 0.091f, 0.082f, 0.073f, 0.064f, 0.055f, 0.045999996f, 0.037f,
                0.027999999f, 0.019000001f, 0.01f, 0.050499998f, 0.045999996f, 0.041500002f, 0.037f };

            TestPolynomialDecay(expectedValues, true);
        }

        internal void TestPolynomialDecay(float[] expectedValues, bool cycle)
        {
            LearningRateScheduler learningRateScheduler = new PolynomialLRDecay(0.1f, 1.0f, 0.01f, 1.0f, cycle);
            DnnTrainState trainState = new DnnTrainState();
            trainState.CurrentBatchIndex = 0;
            trainState.CurrentEpoch = 0;
            trainState.BatchSize = 10;
            trainState.BatchesPerEpoch = 10;
            for (int i = 0; i < expectedValues.Length; ++i)
            {
                trainState.CurrentBatchIndex = i % trainState.BatchesPerEpoch;
                trainState.CurrentEpoch = i / trainState.BatchesPerEpoch;
                float decayedLR = learningRateScheduler.GetLearningRate(trainState);
                Assert.Equal(expectedValues[i], decayedLR, 4);
            }
        }
    }
}
