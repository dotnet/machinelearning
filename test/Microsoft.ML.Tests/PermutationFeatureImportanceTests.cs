//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Tests
{
    public class PermutationFeatureImportanceTests : BaseTestPredictors
    {
        public PermutationFeatureImportanceTests(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random and Label y is to dependant on xRand.
        /// Test verifies that xRand has the least importance: L1, L2, RMS and Loss-Fn do not change a lot when xRand is permuted.
        /// Also test checks that x2 has the biggest importance.
        /// </summary>
        [Fact]
        public void TestDenseSGD()
        {
            // Setup synthetic dataset.
            const int numberOfInstances = 1000;
            var rand = new Random(10);
            Float[] yArray = new Float[numberOfInstances],
                x1Array = new Float[numberOfInstances],
                x2Array = new Float[numberOfInstances],
                x3Array = new Float[numberOfInstances],
                x4RandArray = new Float[numberOfInstances];

            for (var i = 0; i < numberOfInstances; i++)
            {
                var x1 = rand.Next(1000);
                x1Array[i] = x1;
                var x2Important = rand.Next(10000);
                x2Array[i] = x2Important;
                var x3 = rand.Next(5000);
                x3Array[i] = x3;
                var x4Rand = rand.Next(1000);
                x4RandArray[i] = x4Rand;

                var noise = rand.Next(50);
                yArray[i] = (Float)(10 * x1 + 20 * x2Important + 5.5 * x3 + noise);
            }

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberType.Float, x1Array);
            bldr.AddColumn("X2Important", NumberType.Float, x2Array);
            bldr.AddColumn("X3", NumberType.Float, x3Array);
            bldr.AddColumn("X4Rand", NumberType.Float, x4RandArray);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2Important", "X3", "X4Rand")
                .Append(ML.Transforms.Normalize("Features"));
            var data = pipeline.Fit(srcDV).Transform(srcDV);
            var model = ML.Regression.Trainers.OnlineGradientDescent().Fit(data);

            var pfi = new PermutationFeatureImportanceRegression(Env);
            var results = pfi.GetImportanceMetricsMatrix(model, data);

            // For the following metrics lower is better, so maximum delta means more important feature, and vice versa
            Assert.True(MinDeltaFeature(results, m => m.L1) == "X4Rand");
            Assert.True(MaxDeltaFeature(results, m => m.L1) == "X2Important");

            Assert.True(MinDeltaFeature(results, m => m.L2) == "X4Rand");
            Assert.True(MaxDeltaFeature(results, m => m.L2) == "X2Important");

            Assert.True(MinDeltaFeature(results, m => m.Rms) == "X4Rand");
            Assert.True(MaxDeltaFeature(results, m => m.Rms) == "X2Important");

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.True(MaxDeltaFeature(results, m => m.RSquared) == "X4Rand");
            Assert.True(MinDeltaFeature(results, m => m.RSquared) == "X2Important");

            Done();
        }

        /// <summary>
        /// Features: x1, x2vBuff(sparce vector), x3. 
        /// y = 10x1 + 10x2vBuff + 30x3 + e.
        /// Within xBuff feature  2nd slot will be sparse most of the time.
        /// Test verifies that 2nd slot of xBuff has the least importance: L1, L2, RMS and Loss-Fn do not change a lot when this slot is permuted.
        /// Also test checks that x2 has the biggest importance.
        /// </summary>
        [Fact]
        public void TestSparseSGD()
        {
            // Setup synthetic dataset.
            const int numberOfInstances = 10000;
            var rand = new Random(10);
            Float[] yArray = new Float[numberOfInstances],
                x1Array = new Float[numberOfInstances],
                x3Array = new Float[numberOfInstances];

            VBuffer<Float>[] vbArray = new VBuffer<Float>[numberOfInstances];

            for (var i = 0; i < numberOfInstances; i++)
            {
                var x1 = rand.Next(1000);
                x1Array[i] = x1;
                var x3Important = rand.Next(10000);
                x3Array[i] = x3Important;

                VBuffer<Float> vb;

                if (i % 10 != 0)
                {
                    vb = new VBuffer<Float>(4, 3, new Float[] { rand.Next(1000), rand.Next(1000), rand.Next(1000) }, new int[] { 0, 2, 3 });
                }
                else
                {
                    vb = new VBuffer<Float>(4, 4, new Float[] { rand.Next(1000), rand.Next(1000), rand.Next(1000), rand.Next(1000) }, new int[] { 0, 1, 2, 3 });
                }

                vbArray[i] = vb;

                Float vbSum = 0;
                foreach (var vbValue in vb.DenseValues())
                {
                    vbSum += vbValue * 10;
                }

                var noise = rand.Next(50);
                yArray[i] = 10 * x1 + vbSum + 20 * x3Important + noise;
            }

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberType.Float, x1Array);
            bldr.AddColumn("X2VBuffer", NumberType.Float, vbArray);
            bldr.AddColumn("X3Important", NumberType.Float, x3Array);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2VBuffer", "X3Important")
                .Append(ML.Transforms.Normalize("Features"));
            var data = pipeline.Fit(srcDV).Transform(srcDV);
            var model = ML.Regression.Trainers.OnlineGradientDescent().Fit(data);

            var pfi = new PermutationFeatureImportanceRegression(Env);
            var results = pfi.GetImportanceMetricsMatrix(model, data);

            // Permuted 2nd slot (f2) should have min impact on SGD metrics, X3 -- max impact.
            // For the following metrics lower is better, so maximum delta means more important feature, and vice versa
            Assert.True(MinDeltaFeature(results, m => m.L1) == "f2");
            Assert.True(MaxDeltaFeature(results, m => m.L1) == "X3Important");

            Assert.True(MinDeltaFeature(results, m => m.L2) == "f2");
            Assert.True(MaxDeltaFeature(results, m => m.L2) == "X3Important");

            Assert.True(MinDeltaFeature(results, m => m.Rms) == "f2");
            Assert.True(MaxDeltaFeature(results, m => m.Rms) == "X3Important");

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.True(MaxDeltaFeature(results, m => m.RSquared) == "f2");
            Assert.True(MinDeltaFeature(results, m => m.RSquared) == "X3Important");
        }

        private string MinDeltaFeature(
            List<(string featureName, RegressionEvaluator.Result metricsDelta)> results,
            Func<RegressionEvaluator.Result, double> metricSelector)
        {
            return results.OrderBy(r => metricSelector(r.metricsDelta))
                    .First().featureName;
        }

        private string MaxDeltaFeature(
            List<(string featureName, RegressionEvaluator.Result metricsDelta)> results,
            Func<RegressionEvaluator.Result, double> metricSelector)
        {
            return results.OrderByDescending(r => metricSelector(r.metricsDelta))
                    .First().featureName;
        }
    }
}
