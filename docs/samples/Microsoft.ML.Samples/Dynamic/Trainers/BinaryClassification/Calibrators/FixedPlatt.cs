using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace Samples.Dynamic.Trainers.BinaryClassification.Calibrators
{
    public static class FixedPlatt
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Download and featurize the dataset.
            var data = Microsoft.ML.SamplesUtils.DatasetUtils
                .LoadFeaturizedAdultDataset(mlContext);

            // Leave out 10% of data for testing.
            var trainTestData = mlContext.Data
                .TrainTestSplit(data, testFraction: 0.3);

            // Create data training pipeline for non calibrated trainer and train
            // Naive calibrator on top of it.
            var pipeline = mlContext.BinaryClassification.Trainers
                .AveragedPerceptron();

            // Fit the pipeline, and get a transformer that knows how to score new
            // data.
            var transformer = pipeline.Fit(trainTestData.TrainSet);
            // Fit this pipeline to the training data.
            // Let's score the new data. The score will give us a numerical
            // estimation of the chance that the particular sample bears positive
            // sentiment. This estimate is relative to the numbers obtained.
            var scoredData = transformer.Transform(trainTestData.TestSet);
            var outScores = mlContext.Data
                .CreateEnumerable<ScoreValue>(scoredData, reuseRowObject: false);

            PrintScore(outScores, 5);
            // Preview of scoredDataPreview.RowView
            // Score  -0.09044361
            // Score  -9.105377
            // Score  -11.049
            // Score  -3.061928
            // Score  -6.375817

            // Let's train a calibrator estimator on this scored dataset. The
            // trained calibrator estimator produces a transformer that can
            // transform the scored data by adding a new column names "Probability".
            var calibratorEstimator = mlContext.BinaryClassification.Calibrators
                .Platt(slope: -1f, offset: -0.05f);

            var calibratorTransformer = calibratorEstimator.Fit(scoredData);

            // Transform the scored data with a calibrator transfomer by adding a
            // new column names "Probability". This column is a calibrated version
            // of the "Score" column, meaning its values are a valid probability
            // value in the [0, 1] interval representing the chance that the
            // respective sample bears positive sentiment.
            var finalData = calibratorTransformer.Transform(scoredData);
            var outScoresAndProbabilities = mlContext.Data
                .CreateEnumerable<ScoreAndProbabilityValue>(finalData,
                reuseRowObject: false);

            PrintScoreAndProbability(outScoresAndProbabilities, 5);
            // Score -0.09044361  Probability 0.4898905
            // Score -9.105377    Probability 0.0001167479
            // Score -11.049      Probability 1.671815E-05
            // Score -3.061928    Probability 0.04688989
            // Score -6.375817    Probability 0.001786307
        }

        private static void PrintScore(IEnumerable<ScoreValue> values, int numRows)
        {
            foreach (var value in values.Take(numRows))
                Console.WriteLine("{0, -10} {1, -10}", "Score", value.Score);
        }

        private static void PrintScoreAndProbability(
            IEnumerable<ScoreAndProbabilityValue> values, int numRows)

        {
            foreach (var value in values.Take(numRows))
                Console.WriteLine("{0, -10} {1, -10} {2, -10} {3, -10}", "Score",
                    value.Score, "Probability", value.Probability);
        }

        private class ScoreValue
        {
            public float Score { get; set; }
        }

        private class ScoreAndProbabilityValue
        {
            public float Score { get; set; }
            public float Probability { get; set; }
        }
    }
}
