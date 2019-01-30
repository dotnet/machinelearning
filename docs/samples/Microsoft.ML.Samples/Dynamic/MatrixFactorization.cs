using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic
{
    public class MatrixFactorizationExample
    {
        // The following variables defines the shape of a matrix. Its shape is _synthesizedMatrixRowCount-by-_synthesizedMatrixColumnCount.
        // Because in ML.NET key type's minimal value is zero, the first row index is always zero in C# data structure (e.g., MatrixColumnIndex=0
        // and MatrixRowIndex=0 in MatrixElement below specifies the value at the upper-left corner in the training matrix). If user's row index 
        // starts with 1, their row index 1 would be mapped to the 2nd row in matrix factorization module and their first row may contain no values.
        // This behavior is also true to column index.
        const int _synthesizedMatrixFirstColumnIndex = 1;
        const int _synthesizedMatrixFirstRowIndex = 1;
        const int _synthesizedMatrixColumnCount = 60;
        const int _synthesizedMatrixRowCount = 100;

        // A data structure used to encode a single value in matrix
        internal class MatrixElement
        {
            // Matrix column index is at most _synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex.
            [KeyType(Count = _synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            // Matrix row index is at most _synthesizedMatrixRowCount + _synthesizedMatrixFirstRowIndex.
            [KeyType(Count = _synthesizedMatrixRowCount + _synthesizedMatrixFirstRowIndex)]
            public uint MatrixRowIndex;
            // The value at the column MatrixColumnIndex and row MatrixRowIndex.
            public float Value;
        }

        // A data structure used to encode prediction result. Comparing with MatrixElement, The field Value in MatrixElement is
        // renamed to Score because Score is the default name of matrix factorization's output.
        internal class MatrixElementForScore
        {
            [KeyType(Count = _synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            [KeyType(Count = _synthesizedMatrixRowCount + _synthesizedMatrixFirstRowIndex)]
            public uint MatrixRowIndex;
            public float Score;
        }

        // This example first creates in-memory data and then use it to train a matrix factorization model. Afterward, quality metrics are reported.
        public static void MatrixFactorizationInMemoryData()
        {
            // Create an in-memory matrix as a list of tuples (column index, row index, value).
            var dataMatrix = new List<MatrixElement>();
            for (uint i = _synthesizedMatrixFirstColumnIndex; i < _synthesizedMatrixFirstColumnIndex + _synthesizedMatrixColumnCount; ++i)
                for (uint j = _synthesizedMatrixFirstRowIndex; j < _synthesizedMatrixFirstRowIndex + _synthesizedMatrixRowCount; ++j)
                    dataMatrix.Add(new MatrixElement() { MatrixColumnIndex = i, MatrixRowIndex = j, Value = (i + j) % 5 });

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging,
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext(seed: 0, conc: 1);

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var dataView = mlContext.Data.ReadFromEnumerable(dataMatrix);

            // Create a matrix factorization trainer which may consume "Value" as the training label, "MatrixColumnIndex" as the
            // matrix's column index, and "MatrixRowIndex" as the matrix's row index. Here nameof(...) is used to extract field
            // names' in MatrixElement class.

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(MatrixElement.MatrixColumnIndex),
                MatrixRowIndexColumnName = nameof(MatrixElement.MatrixRowIndex),
                LabelColumnName = nameof(MatrixElement.Value),
                NumIterations = 10,
                NumThreads = 1,
                K = 32,
            };

            var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(options);

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            // Apply the trained model to the training set.
            var prediction = model.Transform(dataView);

            // Calculate regression matrices for the prediction result.
            var metrics = mlContext.Recommendation().Evaluate(prediction,
                label: nameof(MatrixElement.Value), score: nameof(MatrixElementForScore.Score));

            // Print out some metrics for checking the model's quality.
            Console.WriteLine($"L1 - {metrics.L1}");
            Console.WriteLine($"L2 - {metrics.L2}");
            Console.WriteLine($"LossFunction - {metrics.LossFn}");
            Console.WriteLine($"RMS - {metrics.Rms}");
            Console.WriteLine($"RSquared - {metrics.RSquared}");

            // Create two two entries for making prediction. Of course, the prediction value, Score, is unknown so it can be anything
            // (here we use Score=0 and it will be overwritten by the true prediction). If any of row and column indexes are out-of-range
            // (e.g., MatrixColumnIndex=99999), the prediction value will be NaN.
            var testMatrix = new List<MatrixElementForScore>() {
                new MatrixElementForScore() { MatrixColumnIndex = 1, MatrixRowIndex = 7, Score = 0 },
                new MatrixElementForScore() { MatrixColumnIndex = 3, MatrixRowIndex = 6, Score = 0 } };

            // Again, convert the test data to a format supported by ML.NET.
            var testDataView = mlContext.Data.ReadFromEnumerable(testMatrix);

            // Feed the test data into the model and then iterate through all predictions.
            foreach (var pred in mlContext.CreateEnumerable<MatrixElementForScore>(testDataView, false))
                Console.WriteLine($"Predicted value at row {pred.MatrixRowIndex} and column {pred.MatrixColumnIndex} is {pred.Score}");
        }
    }
}
