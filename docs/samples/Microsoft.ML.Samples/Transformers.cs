// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Samples
{
    public static class Transformers
    {

        public class Input
        {
            public float Feature0 { get; set; }
            public float Feature1 { get; set; } 
            public float Feature2 { get; set; }
            public float Feature3 { get; set; }
            public float Target { get; set; }
        }

        public static IEnumerable<Input> GetInputData()
        {
            var data = new List<Input>();
            data.Add(new Input { Feature0 = -2.75f, Feature1 = 0.77f, Feature2 = -0.61f, Feature3 = 0.14f, Target = 140.66f });
            data.Add(new Input { Feature0 = -0.61f, Feature1 = -0.37f, Feature2 = -0.12f, Feature3 = 0.55f, Target = 148.12f });
            data.Add(new Input { Feature0 = -0.85f, Feature1 = -0.91f, Feature2 = 1.81f, Feature3 = 0.02f, Target = 402.20f });

            return data;
        }

        /// <summary>
        /// The example for the staticlly typed concat estimator.
        /// </summary>
        public static void ConcatEstimator()
        {
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            IEnumerable<Input> data = GetInputData();

            // A preview of InputData:
            // feature_0; feature_1; feature_2; feature_3;  target
            // -2.75;     0.77;     -0.61;      0.14;       140.66
            // -0.61;    -0.37;     -0.12;      0.55;       148.12
            // -0.85;    -0.91;      1.81;      0.02;       402.20

            // Convert to an DataView.
            var trainingData = env.CreateStreamingDataView(data);

            // Convert the IDataView to statically-typed data view, so its schema can be used on the 
            // pipelines that will get built in top of it. 
            var staticData = trainingData.AssertStatic(env, c => (
                   Feature0: c.R4.Scalar,
                   Feature1: c.R4.Scalar,
                   Feature2: c.R4.Scalar,
                   Feature3: c.R4.Scalar, 
                   Target: c.R4.Scalar));

            // Start creating our processing pipeline. 
            // Let just concatenate all the float columns together into one using ConcatWith.
            var staticLearningPipeline = staticData.MakeNewEstimator()
                .Append(r => (
                    r.Target,
                    Features: r.Feature0.ConcatWith(r.Feature1, r.Feature2, r.Feature3)));

            // Transform the data through the above pipeline.
            var transformedData = staticLearningPipeline.Fit(staticData).Transform(staticData);

            // The transformedData DataView is now of the type (Target:Scalar<float>, Features:Vector<float>).

            // Features                    target
            // -2.75  0.77 -0.61 0.14;     140.66
            // -0.61 -0.37 -0.12 0.55;     148.12
            // -0.85 -0.91  1.81 0.02;     402.20

            // Let's print out the new data.
            var features = transformedData.GetColumn(r => r.Features);

            Console.WriteLine("Features column obtained post-transformation.");
            foreach (var featureRow in features)
            {
                Console.WriteLine($"{featureRow[0]} {featureRow[1]} {featureRow[2]} {featureRow[3]}");
            }
        }
    }
}
