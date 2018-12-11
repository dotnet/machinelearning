// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public sealed class FeatureContributionTests : TestDataPipeBase
    {
        private sealed class ScoreAndContribution
        {
            public float X1 { get; set; }
            public float X2Important { get; set; }
            public float X3 { get; set; }
            public float X4Rand { get; set; }
            public float Score { get; set; }
            [VectorType(4)]
            public float[] FeatureContributions { get; set; }
        }

        public FeatureContributionTests(ITestOutputHelper output) : base(output)
        {
        }

        public IDataView GenerateRegressionData()
        {
            // Setup synthetic dataset.
            const int numberOfInstances = 1000;
            var rand = new Random(10);
            float[] yArray = new float[numberOfInstances],
                x1Array = new float[numberOfInstances],
                x2Array = new float[numberOfInstances],
                x3Array = new float[numberOfInstances],
                x4RandArray = new float[numberOfInstances];

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
                yArray[i] = (float)(10 * x1 + 20 * x2Important + 5.5 * x3 + noise);
            }

            // Create data view.
            var bldr = new ArrayDataViewBuilder(ML);
            bldr.AddColumn("X1", NumberType.Float, x1Array);
            bldr.AddColumn("X2Important", NumberType.Float, x2Array);
            bldr.AddColumn("X3", NumberType.Float, x3Array);
            bldr.AddColumn("X4Rand", NumberType.Float, x4RandArray);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            return bldr.GetDataView();
        }

        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random, Label y is dependant on xRand.
        /// Test verifies that feature contribution scores are outputted along with a score for predicted data. 
        /// </summary>
        [Fact]
        public void TestFeatureContribution()
        {
            var srcDV = GenerateRegressionData();
            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2Important", "X3", "X4Rand")
                .AppendCacheCheckpoint(ML)
                .Append(ML.Transforms.Normalize("Features"));
            var data = pipeline.Fit(srcDV).Transform(srcDV);
            var model = ML.Regression.Trainers.OrdinaryLeastSquares().Fit(data);

            var outputTran = new FeatureContributionCalculatingTransformer(ML, model.FeatureColumn, model.Model).Transform(data);
            var outputEst = new FeatureContributionCalculatingEstimator(ML, model.FeatureColumn, model.Model).Fit(data).Transform(data);

            // Get prediction scores and contributions
            var enumeratorTran = outputTran.AsEnumerable<ScoreAndContribution>(ML, true).GetEnumerator();
            ScoreAndContribution row = null;
            var expectedValues = new List<float[]>();
            expectedValues.Add(new float[4] { 0.06319684F, 1, 0.1386623F, 4.46209469E-06F });
            expectedValues.Add(new float[4] { 0.03841561F, 1, 0.1633037F, 2.68303256E-06F });
            expectedValues.Add(new float[4] { 0.12006103F, 1, 0.254072F, 1.18671605E-05F });
            expectedValues.Add(new float[4] { 0.20861618F, 0.99999994F, 0.407312155F, 6.963478E-05F });
            expectedValues.Add(new float[4] { 0.024050576F, 0.99999994F, 0.31106182F, 8.456762E-06F });
            int index = 0;
            while (enumeratorTran.MoveNext() && index < expectedValues.Count)
            {
                row = enumeratorTran.Current;
                // We set predicion to 6 because the limit of floating-point numbers is 7.
                Assert.Equal(expectedValues[index][0], row.FeatureContributions[0], 6);
                Assert.Equal(expectedValues[index][1], row.FeatureContributions[1], 6);
                Assert.Equal(expectedValues[index][2], row.FeatureContributions[2], 6);
                Assert.Equal(expectedValues[index++][3], row.FeatureContributions[3], 6);
            }

            var enumeratorEst = outputEst.AsEnumerable<ScoreAndContribution>(ML, true).GetEnumerator();
            while (enumeratorEst.MoveNext() && index < expectedValues.Count)
            {
                row = enumeratorEst.Current;
                // We set predicion to 6 because the limit of floating-point numbers is 7.
                Assert.Equal(expectedValues[index][0], row.FeatureContributions[0], 6);
                Assert.Equal(expectedValues[index][1], row.FeatureContributions[1], 6);
                Assert.Equal(expectedValues[index][2], row.FeatureContributions[2], 6);
                Assert.Equal(expectedValues[index++][3], row.FeatureContributions[3], 6);
            }

            Done();
        }

        [Fact]
        public void FeatureContributionEstimatorWorkout()
        {
            var srcDV = GenerateRegressionData();
            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2Important", "X3", "X4Rand")
                .AppendCacheCheckpoint(ML)
                .Append(ML.Transforms.Normalize("Features"));
            var data = pipeline.Fit(srcDV).Transform(srcDV);
            var model = ML.Regression.Trainers.OrdinaryLeastSquares().Fit(data);

            var estPipe = new FeatureContributionCalculatingEstimator(ML, model.FeatureColumn, model.Model, stringify: true)
                .Append(new FeatureContributionCalculatingEstimator(ML, model.FeatureColumn, model.Model))
                .Append(new FeatureContributionCalculatingEstimator(ML, model.FeatureColumn, model.Model, top: 0))
                .Append(new FeatureContributionCalculatingEstimator(ML, model.FeatureColumn, model.Model, bottom: 0))
                .Append(new FeatureContributionCalculatingEstimator(ML, model.FeatureColumn, model.Model, top: 0, bottom: 0));
            TestEstimatorCore(estPipe, data);

            Done();
        }
    }
}
