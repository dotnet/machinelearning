// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.RunTests
{
    /// <summary>
    /// These are tests of the loss functions in the Learners assembly.
    /// </summary>
    public class TestLoss
    {
        private const string _category = "Loss";

        private const float _epsilon = 1e-4f;

        /// <summary>
        /// A small helper for comparing a loss's computations to expected values.
        /// </summary>
        /// <param name="lossFunc">The training loss.</param>
        /// <param name="label">The ideal labeled output.</param>
        /// <param name="output">The actual output.</param>
        /// <param name="expectedLoss">The expected value of this loss, given
        /// <c>label</c> and <c>output</c></param>
        /// <param name="expectedUpdate">The expected value of the update
        /// step, given <c>label</c> and <c>output</c></param>
        /// <param name="differentiable">Whether the loss function is differentiable
        /// w.r.t. the output in the vicinity of the output value</param>
        private void TestHelper(IScalarLoss lossFunc, double label, double output, double expectedLoss, double expectedUpdate, bool differentiable = true)
        {
            Double loss = lossFunc.Loss((float)output, (float)label);
            float derivative = lossFunc.Derivative((float)output, (float)label);
            Assert.Equal(expectedLoss, loss, 5);
            Assert.Equal(expectedUpdate, -derivative, 5);

            if (differentiable)
            {
                // In principle, the update should be the negative of the first derivative of the loss.
                // Use a simple finite difference method to see if it's in the right ballpark.
                float almostOutput = Math.Max((float)output * (1 + _epsilon), (float)output + _epsilon);
                Double almostLoss = lossFunc.Loss(almostOutput, (float)label);
                Assert.Equal((almostLoss - loss) / (almostOutput - output), derivative, 1);
            }
        }

        [Fact]
        public void LossHinge()
        {
            var loss = new HingeLoss();
            // Positive examples.
            TestHelper(loss, 1, 2, 0, 0);
            TestHelper(loss, 1, 1, 0, 0, false);
            TestHelper(loss, 1, 0.99, 0.01, 1, false);
            TestHelper(loss, 1, 0.5, 0.5, 1);
            // Negative examples.
            TestHelper(loss, 0, 0.5, 1.5, -1);
            TestHelper(loss, 0, -0.5, 0.5, -1);
            TestHelper(loss, 0, -1, 0, 0, false);
            TestHelper(loss, 0, -2, 0, 0);
        }

        [Fact]
        public void LossExponential()
        {
            ExpLoss.Options options = new ExpLoss.Options();
            ExpLoss loss = new ExpLoss(options);
            TestHelper(loss, 1, 3, Math.Exp(-3), Math.Exp(-3));
            TestHelper(loss, 0, 3, Math.Exp(3), -Math.Exp(3));
            TestHelper(loss, 0, -3, Math.Exp(-3), -Math.Exp(-3));
        }

        [Fact]
        public void LossSquared()
        {
            SquaredLoss loss = new SquaredLoss();
            TestHelper(loss, 3, 2, 1, 2);
            TestHelper(loss, 3, 4, 1, -2);
            TestHelper(loss, 3, 5, 4, -4);
            TestHelper(loss, 0, -3, 9, 6);
            TestHelper(loss, -3, -5, 4, 4);
        }
    }
}
