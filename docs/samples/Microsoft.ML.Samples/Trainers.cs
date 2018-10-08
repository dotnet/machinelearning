// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.StaticPipe;
using System;

namespace Microsoft.ML.Samples
{
    public static class Trainers
    { 
        public static void SdcaRegression()
        {
            var (trainDataPath, testDataPath) = DatasetCreator.CreateRegressionDataset();

            //creating the ML.Net IHostEnvironment object, needed for the pipeline
            var env = new LocalEnvironment(seed: 0);

            // creating the ML context, based on the task performed.
            var regressionContext = new RegressionContext(env);

            // Creating a data reader, based on the format of the data
            var reader = TextLoader.CreateReader(env, c => (
                     label: c.LoadFloat(2),
                     features: c.LoadFloat(0, 1)
                 ),
                separator: ',', hasHeader: true);

            // Read the data
            var trainData = reader.Read(new MultiFileSource(trainDataPath));

            // The predictor that gets produced out of training
            LinearRegressionPredictor pred = null;

            // Create the estimator
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (r.label, score: regressionContext.Trainers.Sdca(
                                            r.label,
                                            r.features,
                                            l1Threshold: 0f,
                                            maxIterations: 100,
                                        onFit: p => pred = p)
                                )
                        );

            // fit this pipeline to the training data
            var model = learningPipeline.Fit(trainData);

            // check the weights that the model learned
            VBuffer<float> weights = default;
            pred.GetFeatureWeights(ref weights);

            Console.WriteLine($"weight 0 - {weights.Values[0]}");
            Console.WriteLine($"weight 1 - {weights.Values[1]}");

            // test the model we just trained, using the test file. 
            var testData = reader.Read(new MultiFileSource(testDataPath));
            var data = model.Transform(testData);

            //Evaluate how the model is doing on the test data
            var metrics = regressionContext.Evaluate(data, r => r.label, r => r.score);

            Console.WriteLine($"L1 - {metrics.L1}");
            Console.WriteLine($"L2 - {metrics.L2}");
            Console.WriteLine($"LossFunction - {metrics.LossFn}");
            Console.WriteLine($"RMS - {metrics.Rms}");
            Console.WriteLine($"RSquared - {metrics.RSquared}");
        }
    }
}
