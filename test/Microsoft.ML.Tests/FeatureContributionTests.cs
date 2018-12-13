// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public sealed class FeatureContributionTests : BaseTestPredictors
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

        [Fact]
        public void TestOrdinaryLeastSquares()
        {
            var expectedValues = new List<float[]> {
                new float[4] { 0.06319684F, 1, 0.1386623F, 4.46209469E-06F },
                new float[4] { 0.03841561F, 1, 0.1633037F, 2.68303256E-06F },
                new float[4] { 0.12006103F, 1, 0.254072F, 1.18671605E-05F },
                new float[4] { 0.20861618F, 0.99999994F, 0.407312155F, 6.963478E-05F },
                new float[4] { 0.024050576F, 0.99999994F, 0.31106182F, 8.456762E-06F }, };

            TestFeatureContribution(ML.Regression.Trainers.OrdinaryLeastSquares(), expectedValues);
        }

        [Fact]
        public void TestGam()
        {
            // Index 1: Most important feature
            // Index 3: Random feature
            var expectedValues = new List<float[]> {
                new float[4] { 0.08439296F, 1F, 0.1442171F, -0.001832674F },
                new float[4] { -0.07902145F, -1F, -0.01937493F, 0.02314214F },
                new float[4] { 0.04072217F, -1F, 0.01370963F, -0.00197823F },
                new float[4] { -0.02197981F, -1F, -0.1051985F, -0.004131221F },
                new float[4] { -0.1072952F, -1F, 0.1284171F, -0.002337188F }, };

            TestFeatureContribution(ML.Regression.Trainers.GeneralizedAdditiveModels(), expectedValues, 5);
        }

        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random, Label y is dependant on xRand.
        /// Test verifies that feature contribution scores are outputted along with a score for predicted data. 
        /// </summary>
        private void TestFeatureContribution(
            ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictor>, IPredictor> trainer,
            List<float[]> expectedValues,
            int precision = 6)
        {
            // Setup synthetic dataset.
            const int numInstances = 1000;
            const int numFeatures = 4;

            var rand = new Random(10);
            float[] yArray = new float[numInstances];
            float[][] xArray = new float[numFeatures][];
            int[] xRangeArray = new[] { 1000, 10000, 5000, 1000 };
            float[] xWeightArray = new[] {
                10,
                20, // Most important feature with high weight. Should have the highest contribution.
                5.5f,
                0, // Least important feature. Should have the least contribution.
            };

            for (var instanceIndex = 0; instanceIndex < numInstances; instanceIndex++)
            {
                for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++)
                {
                    if (xArray[featureIndex] == null)
                        xArray[featureIndex] = new float[numInstances];
                    xArray[featureIndex][instanceIndex] = rand.Next(xRangeArray[featureIndex]);
                    yArray[instanceIndex] += xArray[featureIndex][instanceIndex] * xWeightArray[featureIndex];
                }

                var noise = rand.Next(50);
                yArray[instanceIndex] += noise;
            }

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberType.Float, xArray[0]);
            bldr.AddColumn("X2Important", NumberType.Float, xArray[1]);
            bldr.AddColumn("X3", NumberType.Float, xArray[2]);
            bldr.AddColumn("X4Rand", NumberType.Float, xArray[3]);
            bldr.AddColumn("Label", NumberType.Float, yArray);
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2Important", "X3", "X4Rand")
                .AppendCacheCheckpoint(ML)
                .Append(ML.Transforms.Normalize("Features"));
            var data = pipeline.Fit(srcDV).Transform(srcDV);
            var model = trainer.Fit(data);
            var args = new FeatureContributionCalculationTransform.Arguments()
            {
                Bottom = 10,
                Top = 10
            };
            var output = FeatureContributionCalculationTransform.Create(Env, args, data, model.Model, model.FeatureColumn);

            var transformedOutput = output.AsEnumerable<ScoreAndContribution>(Env, true);
            int rowIndex = 0;
            foreach (var row in transformedOutput.Take(expectedValues.Count))
            {
                var expectedValue = expectedValues[rowIndex++];
                for (int i = 0; i < numFeatures; i++)
                    Assert.Equal(expectedValue[i], row.FeatureContributions[i], precision);
            }

            Done();
        }
    }
}
