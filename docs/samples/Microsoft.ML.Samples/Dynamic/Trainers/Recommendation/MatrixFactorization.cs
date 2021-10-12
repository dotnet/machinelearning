using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic.Trainers.Recommendation
{
    public static class MatrixFactorization
    {

        // This example requires installation of additional nuget package at
        // for Microsoft.ML.Recommender at
        // https://www.nuget.org/packages/Microsoft.ML.Recommender/
        // In this example we will create in-memory data and then use it to train
        // a matrix factorization model with default parameters. Afterward, quality
        // metrics are reported.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var dataPoints = GenerateMatrix();

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Define the trainer.
            var pipeline = mlContext.Recommendation().Trainers.
                MatrixFactorization(nameof(MatrixElement.Value),
                nameof(MatrixElement.MatrixColumnIndex),
                nameof(MatrixElement.MatrixRowIndex), 10, 0.2, 1);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Run the model on training data set.
            var transformedData = model.Transform(trainingData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data
                .CreateEnumerable<MatrixElement>(transformedData,
                reuseRowObject: false).Take(5).ToList();

            // Look at 5 predictions for the Label, side by side with the actual
            // Label for comparison.
            foreach (var p in predictions)
                Console.WriteLine($"Actual value: {p.Value:F3}," +
                    $"Predicted score: {p.Score:F3}");

            // Expected output:
            //   Actual value: 0.000, Predicted score: 1.234
            //   Actual value: 1.000, Predicted score: 0.792
            //   Actual value: 2.000, Predicted score: 1.831
            //   Actual value: 3.000, Predicted score: 2.670
            //   Actual value: 4.000, Predicted score: 2.362

            // Evaluate the overall metrics
            var metrics = mlContext.Regression.Evaluate(transformedData,
                labelColumnName: nameof(MatrixElement.Value),
                scoreColumnName: nameof(MatrixElement.Score));

            PrintMetrics(metrics);

            // Expected output:
            //   Mean Absolute Error: 0.67:
            //   Mean Squared Error: 0.79
            //   Root Mean Squared Error: 0.89
            //   RSquared: 0.61 (closer to 1 is better. The worst case is 0)
        }

        // The following variables are used to define the shape of the example
        // matrix. Its shape is MatrixRowCount-by-MatrixColumnCount. Because in 
        // ML.NET key type's minimal value is zero, the first row index is always
        // zero in C# data structure (e.g., MatrixColumnIndex=0 and MatrixRowIndex=0
        // in MatrixElement below specifies the value at the upper-left corner in
        // the training matrix). If user's row index starts with 1, their row index
        // 1 would be mapped to the 2nd row in matrix factorization module and their
        // first row may contain no values. This behavior is also true to column
        // index.
        private const uint MatrixColumnCount = 60;
        private const uint MatrixRowCount = 100;

        // Generate a random matrix by specifying all its elements.
        private static List<MatrixElement> GenerateMatrix()
        {
            var dataMatrix = new List<MatrixElement>();
            for (uint i = 0; i < MatrixColumnCount; ++i)
                for (uint j = 0; j < MatrixRowCount; ++j)
                    dataMatrix.Add(new MatrixElement()
                    {
                        MatrixColumnIndex = i,
                        MatrixRowIndex = j,
                        Value = (i + j) % 5
                    });

            return dataMatrix;
        }

        // A class used to define a matrix element and capture its prediction
        // result.
        private class MatrixElement
        {
            // Matrix column index. Its allowed range is from 0 to
            // MatrixColumnCount - 1.
            [KeyType(MatrixColumnCount)]
            public uint MatrixColumnIndex { get; set; }
            // Matrix row index. Its allowed range is from 0 to MatrixRowCount - 1.
            [KeyType(MatrixRowCount)]
            public uint MatrixRowIndex { get; set; }
            // The actual value at the MatrixColumnIndex-th column and the
            // MatrixRowIndex-th row.
            public float Value { get; set; }
            // The predicted value at the MatrixColumnIndex-th column and the
            // MatrixRowIndex-th row.
            public float Score { get; set; }
        }

        // Print some evaluation metrics to regression problems.
        private static void PrintMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine("Mean Absolute Error: " + metrics.MeanAbsoluteError);
            Console.WriteLine("Mean Squared Error: " + metrics.MeanSquaredError);
            Console.WriteLine("Root Mean Squared Error: " +
                metrics.RootMeanSquaredError);

            Console.WriteLine("RSquared: " + metrics.RSquared);
        }
    }
}

