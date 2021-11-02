using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace Samples.Dynamic.Trainers.BinaryClassification
{
    public static class GamWithOptionsAdvanced
    {
        // This example requires installation of additional NuGet package for
        // Microsoft.ML.FastTree found at
        // https://www.nuget.org/packages/Microsoft.ML.FastTree/
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness.
            var mlContext = new MLContext();

            // Create the dataset.
            var samples = GenerateData();

            // Convert the dataset to an IDataView.
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Create training and validation datasets.
            var dataSets = mlContext.Data.TrainTestSplit(data);
            var trainSet = dataSets.TrainSet;
            var validSet = dataSets.TestSet;

            // Create a GAM trainer.
            // Use a small number of bins for this example. The setting below means
            // for each feature, we divide its range into 16 discrete regions for
            // the training process. Note that these regions are not evenly spaced,
            // and that the final model may contain fewer bins, as neighboring bins
            // with identical values will be combined. In general, we recommend
            // using at least the default number of bins, as a small number of bins
            // limits the capacity of the model. Also, set the learning rate to half
            // the default to slow down the gradient descent, and double the number
            // of iterations to compensate.
            var trainer = mlContext.BinaryClassification.Trainers.Gam(
                new GamBinaryTrainer.Options
                {
                    NumberOfIterations = 19000,
                    MaximumBinCountPerFeature = 16,
                    LearningRate = 0.001
                });

            // Fit the model using both of training and validation sets. GAM can use
            // a technique called pruning to tune the model to the validation set
            // after training to improve generalization.
            var model = trainer.Fit(trainSet, validSet);

            // Extract the model parameters.
            var gam = model.Model.SubModel;

            // Now we can inspect the parameters of the Generalized Additive Model
            // to understand the fit and potentially learn about our dataset.
            // First, we will look at the bias; the bias represents the average
            // prediction for the training data.
            Console.WriteLine($"Average prediction: {gam.Bias:0.00}");

            // Now look at the shape functions that the model has learned. Similar
            // to a linear model, we have one response per feature, and they are
            // independent. Unlike a linear model, this response is a generic
            // function instead of a line. Because we have included a bias term,
            // each feature response represents the deviation from the average
            // prediction as a function of the feature value.
            for (int i = 0; i < gam.NumberOfShapeFunctions; i++)
            {
                // Break a line.
                Console.WriteLine();

                // Get the bin upper bounds for the feature.
                var binUpperBounds = gam.GetBinUpperBounds(i);

                // Get the bin effects; these are the function values for each bin.
                var binEffects = gam.GetBinEffects(i);

                // Now, write the function to the console. The function is a set of
                // bins, and the corresponding function values. You can think of
                // GAMs as building a bar-chart or lookup table for each feature.
                Console.WriteLine($"Feature{i}");
                for (int j = 0; j < binUpperBounds.Count; j++)
                    Console.WriteLine(
                        $"x < {binUpperBounds[j]:0.00} => {binEffects[j]:0.000}");
            }

            // Expected output:
            //  Average prediction: 0.82
            //
            //  Feature0
            //  x < -0.44 => 0.286
            //  x < -0.38 => 0.225
            //  x < -0.32 => 0.048
            //  x < -0.26 => -0.110
            //  x < -0.20 => -0.116
            //  x < 0.18 => -0.143
            //  x < 0.25 => -0.115
            //  x < 0.31 => -0.005
            //  x < 0.37 => 0.097
            //  x < 0.44 => 0.263
            //  x < ∞ => 0.284
            //
            //  Feature1
            //  x < 0.00 => -0.350
            //  x < 0.24 => 0.875
            //  x < 0.31 => -0.138
            //  x < ∞ => -0.188

            // Let's consider this output. To score a given example, we look up the
            // first bin where the inequality is satisfied for the feature value.
            // We can look at the whole function to get a sense for how the model
            // responds to the variable on a global level. The model can be seen to
            // reconstruct the parabolic and step-wise function, shifted with
            // respect to the average expected output over the training set. Very
            // few bins are used to model the second feature because the GAM model
            // discards unchanged bins to create smaller models. One last thing to
            // notice is that these feature functions can be noisy. While we know
            // that Feature1 should be symmetric, this is not captured in the model.
            // This is due to noise in the data. Common practice is to use
            // resampling methods to estimate a confidence interval at each bin.
            // This will help to determine if the effect is real or just sampling
            // noise. See for example: Tan, Caruana, Hooker, and Lou.
            // "Distill-and-Compare: Auditing Black-Box Models Using Transparent
            // Model Distillation."
            // <a href='https://arxiv.org/abs/1710.06169'>arXiv:1710.06169</a>."
        }

        private class Data
        {
            public bool Label { get; set; }

            [VectorType(2)]
            public float[] Features { get; set; }
        }

        /// <summary>
        /// Creates a dataset, an IEnumerable of Data objects, for a GAM sample.
        /// Feature1 is a parabola centered around 0, while Feature2 is a simple
        /// piecewise function.
        /// </summary>
        /// <param name="numExamples">The number of examples to generate.</param>
        /// <param name="seed">The seed for the random number generator used to
        /// produce data.</param>
        /// <returns></returns>
        private static IEnumerable<Data> GenerateData(int numExamples = 25000,
            int seed = 1)
        {
            var rng = new Random(seed);
            float centeredFloat() => (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < numExamples; i++)
            {
                // Generate random, uncoupled features.
                var data = new Data
                {
                    Features = new float[2] { centeredFloat(), centeredFloat() }
                };
                // Compute the label from the shape functions and add noise.
                data.Label = Sigmoid(Parabola(data.Features[0]) + SimplePiecewise(
                    data.Features[1]) + centeredFloat()) > 0.5;

                yield return data;
            }
        }

        private static float Parabola(float x) => x * x;

        private static float SimplePiecewise(float x)
        {
            if (x < 0)
                return 0;
            else if (x < 0.25)
                return 1;
            else
                return 0;
        }

        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-1 * x));
    }
}
