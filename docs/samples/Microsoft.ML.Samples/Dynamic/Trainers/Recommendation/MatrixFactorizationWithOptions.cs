using System;
using System.Collections.Generic;
using Microsoft.ML.Trainers;
using static Microsoft.ML.SamplesUtils.DatasetUtils;

namespace Microsoft.ML.Samples.Dynamic.Trainers.Recommendation
{
    public static class MatrixFactorizationWithOptions
    {
        // This example requires installation of additional nuget package <a href="https://www.nuget.org/packages/Microsoft.ML.Recommender/">Microsoft.ML.Recommender</a>.
        // In this example we will create in-memory data and then use it to train a matrix factorization model.
        // Afterward, quality metrics are reported.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 0);

            // Get a small in-memory dataset.
            var data = GetRecommendationData();

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // Create a matrix factorization trainer which may consume "Value" as the training label, "MatrixColumnIndex" as the
            // matrix's column index, and "MatrixRowIndex" as the matrix's row index. Here nameof(...) is used to extract field
            // names' in MatrixElement class.
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(MatrixElement.MatrixColumnIndex),
                MatrixRowIndexColumnName = nameof(MatrixElement.MatrixRowIndex),
                LabelColumnName = nameof(MatrixElement.Value),
                NumberOfIterations = 10,
                NumberOfThreads = 1,
                ApproximationRank = 32,
                LearningRate = 0.3
            };

            var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(options);

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            // Apply the trained model to the training set.
            var prediction = model.Transform(dataView);

            // Calculate regression matrices for the prediction result.
            var metrics = mlContext.Recommendation().Evaluate(prediction,
                labelColumnName: nameof(MatrixElement.Value), scoreColumnName: nameof(MatrixElementForScore.Score));
            // Print out some metrics for checking the model's quality.
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);
            // L1: 0.16
            // L2: 0.04
            // LossFunction: 0.04
            // RMS: 0.21
            // RSquared: 0.98

            // Create two two entries for making prediction. Of course, the prediction value, Score, is unknown so it can be anything
            // (here we use Score=0 and it will be overwritten by the true prediction). If any of row and column indexes are out-of-range
            // (e.g., MatrixColumnIndex=99999), the prediction value will be NaN.
            var testMatrix = new List<MatrixElementForScore>() {
                new MatrixElementForScore() { MatrixColumnIndex = 1, MatrixRowIndex = 7, Score = 0 },
                new MatrixElementForScore() { MatrixColumnIndex = 3, MatrixRowIndex = 6, Score = 0 } };

            // Again, convert the test data to a format supported by ML.NET.
            var testDataView = mlContext.Data.LoadFromEnumerable(testMatrix);

            // Feed the test data into the model and then iterate through all predictions.
            foreach (var pred in mlContext.Data.CreateEnumerable<MatrixElementForScore>(model.Transform(testDataView), false))
                Console.WriteLine($"Predicted value at row {pred.MatrixRowIndex-1} and column {pred.MatrixColumnIndex-1} is {pred.Score}");
            // Predicted value at row 7 and column 1 is 2.828761
            // Predicted value at row 6 and column 3 is 3.642226
        }
    }
}
