using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class CalculateFeatureContribution
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 1);

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
            var linearTrainer = mlContext.Regression.Trainers.Ols();

            // Now we train the model and score it on the transformed data.
            var linearModel = linearTrainer.Fit(transformedData);
            // Print the model parameters.
            Console.WriteLine($"Linear Model Parameters");
            Console.WriteLine("Bias: " + linearModel.Model.Bias + " Feature1: " +
                linearModel.Model.Weights[0] + " Feature2: " + linearModel.Model
                .Weights[1]);

            // Define a feature contribution calculator for all the features, and
            // don't normalize the contributions.These are "trivial estimators" and
            // they don't need to fit to the data, so we can feed a subset.
            var simpleScoredDataset = linearModel.Transform(mlContext.Data
                .TakeRows(transformedData, 1));

            var linearFeatureContributionCalculator = mlContext.Transforms
                .CalculateFeatureContribution(linearModel, normalize: false).Fit(
                simpleScoredDataset);

            // Create a transformer chain to describe the entire pipeline.
            var scoringPipeline = transformer.Append(linearModel).Append(
                linearFeatureContributionCalculator);

            // Create the prediction engine to get the features extracted from the
            // text.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<Data,
                ScoredData>(scoringPipeline);

            // Convert the text into numeric features.
            var prediction = predictionEngine.Predict(samples.First());

            // Write out the prediction, with contributions.
            // Note that for the linear model, the feature contributions for a
            // feature in an example is the feature-weight*feature-value.
            // The total prediction is thus the bias plus the feature contributions.
            Console.WriteLine("Label: " + prediction.Label + " Prediction: " +
                prediction.Score);

            Console.WriteLine("Feature1: " + prediction.Features[0] +
                " Feature2: " + prediction.Features[1]);

            Console.WriteLine("Feature Contributions: " + prediction
                .FeatureContributions[0] + " " + prediction
                .FeatureContributions[1]);

            // Expected output:
            //  Linear Model Parameters
            //  Bias: -0.007505796 Feature1: 1.536963 Feature2: 3.031206
            //  Label: 1.55184 Prediction: 1.389091
            //  Feature1: -0.5053467 Feature2: 0.7169741
            //  Feature Contributions: -0.7766994 2.173296
        }

        private class Data
        {
            public float Label { get; set; }

            public float Feature1 { get; set; }

            public float Feature2 { get; set; }
        }

        private class ScoredData : Data
        {
            public float Score { get; set; }
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

                // Create a noisy label.
                data.Label = (float)(bias + weight1 * data.Feature1 + weight2 *
                    data.Feature2 + rng.NextDouble() - 0.5);

                yield return data;
            }
        }
    }
}
