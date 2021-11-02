using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class CalculateFeatureContributionCalibrated
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset.
            var samples = GenerateData();

            // Convert training data to IDataView.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Create a pipeline to concatenate the features into a feature vector
            // and normalize it.
            var transformPipeline = mlContext.Transforms.Concatenate("Features",
                    new string[] { nameof(Data.Feature1), nameof(Data.Feature2) })
                .Append(mlContext.Transforms.NormalizeMeanVariance("Features"));

            // Fit the pipeline.
            var transformer = transformPipeline.Fit(data);

            // Transform the data.
            var transformedData = transformer.Transform(data);

            // Define a linear trainer.
            var linearTrainer = mlContext.BinaryClassification.Trainers
                .SdcaLogisticRegression();

            // Now we train the model and score it on the transformed data.
            var linearModel = linearTrainer.Fit(transformedData);
            // Print the model parameters.
            Console.WriteLine($"Linear Model Parameters");
            Console.WriteLine("Bias: {0} Feature1: {1} Feature2: {2}",
                linearModel.Model.SubModel.Bias,
                linearModel.Model.SubModel.Weights[0],
                linearModel.Model.SubModel.Weights[1]);

            // Define a feature contribution calculator for all the features, and
            // don't normalize the contributions. These are "trivial estimators" and
            // they don't need to fit to the data, so we can feed a subset.
            var simpleScoredDataset = linearModel.Transform(mlContext.Data
                .TakeRows(transformedData, 1));

            var linearFeatureContributionCalculator = mlContext.Transforms
                .CalculateFeatureContribution(linearModel, normalize: false)
                .Fit(simpleScoredDataset);

            // Create a transformer chain to describe the entire pipeline.
            var scoringPipeline = transformer.Append(linearModel)
                .Append(linearFeatureContributionCalculator);

            // Create the prediction engine to get the features extracted from the
            // text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<Data,
                ScoredData>(scoringPipeline);

            // Convert the text into numeric features.
            var prediction = predictionEngine.Predict(samples.First());

            // Write out the prediction, with contributions.
            // Note that for the linear model, the feature contributions for a
            // feature in an example is the feature-weight*feature-value. The total
            // prediction is thus the bias plus the feature contributions.
            Console.WriteLine("Label: " + prediction.Label + " Prediction-Score: " +
                prediction.Score + " Prediction-Probability: " + prediction
                .Probability);

            Console.WriteLine("Feature1: " + prediction.Features[0] + " Feature2: "
                + prediction.Features[1]);

            Console.WriteLine("Feature Contributions: " + prediction
                .FeatureContributions[0] + " " + prediction
                .FeatureContributions[1]);

            // Expected output:
            //  Linear Model Parameters
            //  Bias: 0.003757346 Feature1: 9.070082 Feature2: 17.7816
            //  Label: True Prediction-Score: 8.169167 Prediction-Probability: 0.9997168
            //  Feature1: -0.5053467 Feature2: 0.7169741
            //  Feature Contributions: -4.583536 12.74894
        }

        private class Data
        {
            public bool Label { get; set; }

            public float Feature1 { get; set; }

            public float Feature2 { get; set; }
        }

        private class ScoredData : Data
        {
            public float Score { get; set; }

            public float Probability { get; set; }

            public float[] Features { get; set; }

            public float[] FeatureContributions { get; set; }
        }

        /// <summary>
        /// Generate an enumerable of Data objects, creating the label as a simple
        /// linear combination of the features.
        /// </summary>
        /// <param name="nExamples">The number of examples.</param>
        /// <param name="bias">The bias, or offset, in the calculation of the label.</param>
        /// <param name="weight1">The weight to multiply the first feature with to compute the label.</param>
        /// <param name="weight2">The weight to multiply the second feature with to compute the label.</param>
        /// <param name="seed">The seed for generating feature values and label noise.</param>
        /// <returns>An enumerable of Data objects.</returns>
        private static IEnumerable<Data> GenerateData(int nExamples = 10000,
            double bias = 0, double weight1 = 1, double weight2 = 2, int seed = 1)
        {
            var rng = new Random(seed);
            for (int i = 0; i < nExamples; i++)
            {
                var data = new Data
                {
                    Feature1 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                    Feature2 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                };

                // Create a Boolean label with noise.
                var value = bias + weight1 * data.Feature1 + weight2 * data.Feature2
                    + rng.NextDouble() - 0.5;

                data.Label = Sigmoid(value) > 0.5;
                yield return data;
            }
        }
        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-1 * x));
    }
}
