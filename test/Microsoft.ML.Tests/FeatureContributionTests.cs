// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.RunTests;
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

        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random, Label y is dependant on xRand.
        /// Test verifies that feature contribution scores are outputted along with a score for predicted data. 
        /// </summary>
        [Fact]
        public void TestFeatureImportance()
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
            var args = new FeatureContributionCalculationTransform.Arguments()
            {
                Bottom = 10,
                Top = 10
            };
            var output = FeatureContributionCalculationTransform.Create(Env, args, data, model.Model, model.FeatureColumn);

            // Get prediction scores and contributions
            var enumerator = output.AsEnumerable<ScoreAndContribution>(Env, true).GetEnumerator();
            ScoreAndContribution row = null;
            var expectedValues = new List<float[]>();
            expectedValues.Add(new float[4] { 0.15640761F, 1, 0.155862764F, 0.07276783F });
            expectedValues.Add(new float[4] { 0.09507586F, 1, 0.1835608F, 0.0437548943F });
            expectedValues.Add(new float[4] { 0.297142357F, 1, 0.2855884F, 0.193529665F });
            expectedValues.Add(new float[4] { 0.45465675F, 0.8805887F, 0.4031663F, 1 });
            expectedValues.Add(new float[4] { 0.0595234372F, 0.99999994F, 0.349647522F, 0.137912869F });
            int index = 0;
            while (enumerator.MoveNext() && index < expectedValues.Count)
            {
                row = enumerator.Current;
                Assert.True(row.FeatureContributions[0] == expectedValues[index][0]);
                Assert.True(row.FeatureContributions[1] == expectedValues[index][1]);
                Assert.True(row.FeatureContributions[2] == expectedValues[index][2]);
                Assert.True(row.FeatureContributions[3] == expectedValues[index++][3]);
            }

            Done();
        }
    }
}
