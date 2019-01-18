// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Trainers.FastTree;
using Xunit;

namespace Microsoft.ML.RunTests
{
    public class TestGamPublicInterfaces
    {
        public TestGamPublicInterfaces()
        {
        }

        [Fact]
        [TestCategory("FastTree")]
        public void TestGamDirectInstatiation()
        {
            var mlContext = new MLContext(seed: 1);

            double intercept = 1;

            var binUpperBounds = new double[2][];
            binUpperBounds[0] = new double[] { 1, 2, 3 };
            binUpperBounds[1] = new double[] { 4, 5, 6 };

            var binEffects = new double[2][];
            binEffects[0] = new double[] { 0, 1, 2 };
            binEffects[1] = new double[] { 2, 1, 0 };

            var gam = new RegressionGamModelParameters(mlContext, binUpperBounds, binEffects, intercept);

            // Check that the model has the right number of shape functions
            Assert.Equal(binUpperBounds.Length, gam.NumShapeFunctions);

            // Check the intercept
            Assert.Equal(intercept, gam.Intercept, 6);

            // Check that the binUpperBounds were made correctly
            CheckArrayOfArrayEquality(binUpperBounds, gam.GetBinUpperBounds());
            for (int i = 0; i < gam.NumShapeFunctions; i++)
                CheckArrayEquality(binUpperBounds[i], gam.GetBinUpperBounds(i));

            // Check that the bin effects were made correctly
            CheckArrayOfArrayEquality(binEffects, gam.GetBinEffects());
            for (int i = 0; i < gam.NumShapeFunctions; i++)
                CheckArrayEquality(binEffects[i], gam.GetBinEffects(i));

            // Check that the constructor handles bad input properly
            Assert.Throws<System.ArgumentNullException>(() => new RegressionGamModelParameters(mlContext, binUpperBounds, null, intercept));
            Assert.Throws<System.ArgumentNullException>(() => new RegressionGamModelParameters(mlContext, null, binEffects, intercept));
            Assert.Throws<System.ArgumentNullException>(() => new RegressionGamModelParameters(mlContext, null, null, intercept));
            var misMatchArray = new double[1][];
            misMatchArray[0] = new double[] { 0 };
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new RegressionGamModelParameters(mlContext, binUpperBounds, misMatchArray, intercept));
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new RegressionGamModelParameters(mlContext, misMatchArray, binEffects, intercept));
        }

        private void CheckArrayOfArrayEquality(double[][] array1, double[][] array2)
        {
            Assert.Equal(array1.Length, array2.Length);
            for (int i = 0; i < array1.Length; i++)
                CheckArrayEquality(array1[i], array2[i]);
        }

        private void CheckArrayEquality(double[] array1, double[] array2)
        {
            Assert.Equal(array1.Length, array2.Length);
            for (int i = 0; i < array1.Length; i++)
                Assert.Equal(array1[i], array2[i], 6);
        }
    }
}
