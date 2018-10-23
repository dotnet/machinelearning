// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

        // the alignment of the usings with the methods is intentional so they can display on the same level in the docs site. 
        using Microsoft.ML.Runtime.Data;
        using Microsoft.ML.Runtime.LightGBM;
        using Microsoft.ML.StaticPipe;
        using System;

// NOTE: WHEN ADDING TO THE FILE, ALWAYS APPEND TO THE END OF IT. 
// If you change the existinc content, check that the files referencing it in the XML documentation are still correct, as they reference
// line by line. 
namespace Microsoft.ML.Samples.Static
{
    public partial class TrainersSamples
    {
        public static void LightGbmRegression()
        {
            // Downloading a regression dataset from github.com/dotnet/machinelearning
            // this will create a housing.txt file in the filsystem this code will run
            // you can open the file to see the data. 
            string dataFile = SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Creating the ML.Net IHostEnvironment object, needed for the pipeline
            var env = new LocalEnvironment(seed: 0);

            // Creating the ML context, based on the task performed.
            var regressionContext = new RegressionContext(env);

            // Creating a data reader, based on the format of the data
            var reader = TextLoader.CreateReader(env, c => (
                        label: c.LoadFloat(0),
                        features: c.LoadFloat(1, 6)
                    ),
                separator: '\t', hasHeader: true);

            // Read the data, and leave 10% out, so we can use them for testing
            var data = reader.Read(new MultiFileSource(dataFile));
            var (trainData, testData) = regressionContext.TrainTestSplit(data, testFraction: 0.1);

            // The predictor that gets produced out of training
            LightGbmRegressionPredictor pred = null;

            // Create the estimator
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (r.label, score: regressionContext.Trainers.LightGbm(
                                            r.label,
                                            r.features,
                                            numLeaves: 4,
                                            minDataPerLeaf: 6,
                                            learningRate: 0.001,
                                        onFit: p => pred = p)
                                )
                        );

            // Fit this pipeline to the training data
            var model = learningPipeline.Fit(trainData);

            // Check the weights that the model learned
            VBuffer<float> weights = default;
            pred.GetFeatureWeights(ref weights);

            Console.WriteLine($"weight 0 - {weights.Values[0]}");
            Console.WriteLine($"weight 1 - {weights.Values[1]}");

            // Evaluate how the model is doing on the test data
            var dataWithPredictions = model.Transform(testData);
            var metrics = regressionContext.Evaluate(dataWithPredictions, r => r.label, r => r.score);

            Console.WriteLine($"L1 - {metrics.L1}");
            Console.WriteLine($"L2 - {metrics.L2}");
            Console.WriteLine($"LossFunction - {metrics.LossFn}");
            Console.WriteLine($"RMS - {metrics.Rms}");
            Console.WriteLine($"RSquared - {metrics.RSquared}");
        }
    }
}
