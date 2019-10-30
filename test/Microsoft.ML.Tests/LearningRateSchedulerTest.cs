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

        [Fact]
        public void TestExponentialDecayWithStaircase()
        {
            // The following expected values were evaluated for a certain epoch and batch size to ensure the nature of expoenetial decay is captured and tested
            // results compared with https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/exponential_decay
            float[] expectedValues = new float[] { 0.01f, 0.00733904f, 0.005386151f, 0.0029010624f, 0.0015625558f, 0.0011467661f, 0.00084161625f, 0.00061766553f, 0.00048224168f};
            int[] epochs = new int[] { 0, 10, 20, 40, 60, 70, 80, 90, 99 };
            int[] batchIndex = new int[] { 0, 4, 6, 9, 1, 2, 3, 5, 7 };
            TestExponentialDecay(expectedValues, epochs, batchIndex, true);
        }

        [Fact]
        public void TestExponentialDecayWithoutStaircase()
        {
            // The following expected values were evaluated for a certain epoch and batch size to ensure the nature of expoenetial decay is captured and tested
            // results compared with https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/exponential_decay
            float[] expectedValues = new float[] { 0.01f, 0.007248779f, 0.005287092f, 0.0028213991f, 0.0015577292f, 0.0011396924f, 0.0008338409f, 0.0006081845f, 0.0004575341f };
            int[] epochs = new int[] { 0, 10, 20, 40, 60, 70, 80, 90, 99 };
            int[] batchIndex = new int[] { 0, 4, 6, 9, 1, 2, 3, 5, 7 };
            TestExponentialDecay(expectedValues, epochs, batchIndex, false);
        }

        internal void TestExponentialDecay(float[] expectedValues, int[] epochs, int[] batchIndex, bool staircase)
        {
            LearningRateScheduler expDecayLR = new ExponentialLRDecay(0.01f, 2.0f, 0.94f, staircase);
            TrainState trainState = new TrainState();
            trainState.CurrentBatchIndex = 0;
            trainState.CurrentEpoch = 0;
            trainState.BatchSize = 10;
            trainState.BatchesPerEpoch = 10;
            
            for(int i = 0; i < expectedValues.Length; ++i)
            {
                trainState.CurrentBatchIndex = batchIndex[i];
                trainState.CurrentEpoch = epochs[i];
                float expDecayedLR = expDecayLR.GetLearningRate(trainState);
                Assert.Equal(expectedValues[i], expDecayedLR, 4);
            }

        }

        [Fact]
        public void TestCyclicLRTriangularMode()
        {
            float[] expectedValues = new float[] { 0.01f, 0.08199999999999999f, 0.03700000000000007f, 0.05049999999999994f, 0.014500000000000065f, 0.09099999999999987f, 0.023499999999999875f, 0.0775f, 0.023499999999999875f };
            int[] epochs = new int[] { 0, 10, 20, 40, 60, 70, 80, 90, 99 };
            int[] batchIndex = new int[] { 0, 4, 6, 9, 1, 2, 3, 5, 7 };
            TestCyclicLR(expectedValues, epochs, batchIndex, "triangular");
        }

        [Fact]
        public void TestCyclicLRTriangular2Mode()
        {
            float[] expectedValues = new float[] { 0.01f, 0.027999999999999997f, 0.010843750000000003f, 0.01003955078125f, 0.010000137329101563f, 0.01000061798095703f, 0.01000001287460327f, 0.010000016093254089f, 0.010000000804662705f };
            int[] epochs = new int[] { 0, 10, 20, 40, 60, 70, 80, 90, 99 };
            int[] batchIndex = new int[] { 0, 4, 6, 9, 1, 2, 3, 5, 7 };
            TestCyclicLR(expectedValues, epochs, batchIndex, "triangular2");
        }

        [Fact]
        public void TestCyclicLRExpMode()
        {
            float[] expectedValues = new float[] { 0.01f, 0.08155210544740442f, 0.03666832402973435f, 0.049518196546084844f, 0.01434061617973948f, 0.08765903443735394f, 0.022864971623094716f, 0.07393238041615638f, 0.02271608707595756f };
            int[] epochs = new int[] { 0, 10, 20, 40, 60, 70, 80, 90, 99 };
            int[] batchIndex = new int[] { 0, 4, 6, 9, 1, 2, 3, 5, 7 };
            TestCyclicLR(expectedValues, epochs, batchIndex, "exp_range");
        }

        internal void TestCyclicLR(float[] expectedValues, int[] epochs, int[] batchIndex, string mode)
        {
            LearningRateScheduler clr = new CyclicLR(20, 0.01f, 0.1f, mode, 0.99994f) ;
            TrainState trainState = new TrainState();
            trainState.CurrentBatchIndex = 0;
            trainState.CurrentEpoch = 0;
            trainState.BatchSize = 10;
            trainState.BatchesPerEpoch = 10;

            for (int i = 0; i < expectedValues.Length; ++i)
            {
                trainState.CurrentBatchIndex = batchIndex[i];
                trainState.CurrentEpoch = epochs[i];
                float clrDecayedLR = clr.GetLearningRate(trainState);
                Assert.Equal(expectedValues[i], clrDecayedLR, 4);
            }
        }

        [Fact]
        public void TestLsrDecay()
        {
            int[] epochs = { 200, 150, 100, 10 };
            float[] expectedValues = new float[] {7.8125e-06f, 7.8125e-05f, 0.00078125f, 0.0078125f};
            LearningRateScheduler lsr = new LsrDecay();

            TrainState trainState = new TrainState();
            trainState.CurrentBatchIndex = 0;
            trainState.CurrentEpoch = 0;
            trainState.BatchSize = 10;
            trainState.BatchesPerEpoch = 10;

            for (int i = 0; i < epochs.Length; i++)
            {
                trainState.CurrentEpoch = epochs[i];
                float lsrDecayedLR = lsr.GetLearningRate(trainState);
                Assert.Equal(expectedValues[i], lsrDecayedLR, 4);
            }
        }

    }
}
