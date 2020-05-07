// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public class TestGamPublicInterfaces : BaseTestClass
    {
        public TestGamPublicInterfaces(ITestOutputHelper output) : base(output)
        {
        }

        [Theory]
        [IterationData(iterations: 20)]
        [Trait("Category", "RunSpecificTest")]
        public void CompletesTestGamDirectInstatiation(int iterations)
        {
            Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(TestGamDirectInstatiation);
            var timeoutTask = Task.Delay(timeout + iterations);
            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("TestGamDirectInstatiation test Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
        }

        [Fact]
        [TestCategory("FastTree")]
        public void TestGamDirectInstatiation()
        {
            var mlContext = new MLContext(seed: 1);

            double intercept = 1;

            var binUpperBounds = new double[2][]
            {
                new double[] { 1, 2, 3 },
                new double[] { 4, 5, 6 }
            };

            var binEffects = new double[2][]
            {
                new double[] { 0, 1, 2 },
                new double[] { 2, 1, 0 }
            };

            var gam = new GamRegressionModelParameters(mlContext, binUpperBounds, binEffects, intercept);

            // Check that the model has the right number of shape functions
            Assert.Equal(binUpperBounds.Length, gam.NumberOfShapeFunctions);

            // Check the intercept
            Assert.Equal(intercept, gam.Bias, 6);

            // Check that the binUpperBounds were made correctly
            CheckArrayOfArrayEquality(binUpperBounds, gam.GetBinUpperBounds());
            for (int i = 0; i < gam.NumberOfShapeFunctions; i++)
                Utils.AreEqual(binUpperBounds[i], gam.GetBinUpperBounds(i).ToArray());

            // Check that the bin effects were made correctly
            CheckArrayOfArrayEquality(binEffects, gam.GetBinEffects());
            for (int i = 0; i < gam.NumberOfShapeFunctions; i++)
                Utils.AreEqual(binEffects[i], gam.GetBinEffects(i).ToArray());

            // Check that the constructor handles null inputs properly
            Assert.Throws<System.ArgumentNullException>(() => new GamRegressionModelParameters(mlContext, binUpperBounds, null, intercept));
            Assert.Throws<System.ArgumentNullException>(() => new GamRegressionModelParameters(mlContext, null, binEffects, intercept));
            Assert.Throws<System.ArgumentNullException>(() => new GamRegressionModelParameters(mlContext, null, null, intercept));

            // Check that the constructor handles mismatches in length between bin upper bounds and bin effects
            var misMatchArray = new double[1][];
            misMatchArray[0] = new double[] { 0 };
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new GamRegressionModelParameters(mlContext, binUpperBounds, misMatchArray, intercept));
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new GamRegressionModelParameters(mlContext, misMatchArray, binEffects, intercept));

            // Check that the constructor handles a mismatch in bin upper bounds and bin effects for a feature
            var fewerBinEffects = new double[2][]
            {
                new double[] { 0, 1 },
                new double[] { 2, 1, 0 }
            };
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new GamRegressionModelParameters(mlContext, binUpperBounds, fewerBinEffects, intercept));
            var moreBinEffects = new double[2][]
            {
                new double[] { 0, 1, 2, 3 },
                new double[] { 2, 1, 0 }
            };
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new GamRegressionModelParameters(mlContext, binUpperBounds, moreBinEffects, intercept));

            // Check that the constructor handles bin upper bounds that are not sorted
            var unsortedUpperBounds = new double[2][]
            {
                new double[] { 1, 3, 2 },
                new double[] { 4, 5, 6 }
            };
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new GamRegressionModelParameters(mlContext, unsortedUpperBounds, binEffects, intercept));
        }

        private void CheckArrayOfArrayEquality(double[][] array1, double[][] array2)
        {
            Assert.Equal(array1.Length, array2.Length);
            for (int i = 0; i < array1.Length; i++)
                Utils.AreEqual(array1[i], array2[i]);
        }
    }
}
