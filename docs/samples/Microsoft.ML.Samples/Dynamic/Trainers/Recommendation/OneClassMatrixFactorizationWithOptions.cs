using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Samples.Dynamic.Trainers.Recommendation
{
    public static class OneClassMatrixFactorizationWithOptions
    {
        // This example shows the use of ML.NET's one-class matrix factorization
        // module which implements a coordinate descent method described in
        // Algorithm 1 in the paper found at 
        // https://www.csie.ntu.edu.tw/~cjlin/papers/one-class-mf/biased-mf-sdm-with-supp.pdf
        // See page 28 in of the slides
        // at https://www.csie.ntu.edu.tw/~cjlin/talks/facebook.pdf for a brief 
        // introduction to one-class matrix factorization.
        // In this example we will create in-memory data and then use it to train a
        // one-class matrix factorization model. Afterward, prediction values are
        // reported. To run this example, it requires installation of additional
        // nuget package Microsoft.ML.Recommender found at
        // https://www.nuget.org/packages/Microsoft.ML.Recommender/
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness.
            var mlContext = new MLContext(seed: 0);

            // Get a small in-memory dataset.
            GetOneClassMatrix(out List<MatrixElement> data,
                out List<MatrixElement> testData);

            // Convert the in-memory matrix into an IDataView so that ML.NET
            // components can consume it.
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // Create a matrix factorization trainer which takes "Value" as the
            // training label, "MatrixColumnIndex" as the matrix's column index, and
            // "MatrixRowIndex" as the matrix's row index. Here nameof(...) is used
            // to extract field
            // names' in MatrixElement class.
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(
                    MatrixElement.MatrixColumnIndex),
                MatrixRowIndexColumnName = nameof(MatrixElement.MatrixRowIndex),
                LabelColumnName = nameof(MatrixElement.Value),
                NumberOfIterations = 20,
                NumberOfThreads = 8,
                ApproximationRank = 32,
                Alpha = 1,

                // The desired values of matrix elements not specified in the
                // training set. If the training set doesn't tell the value at the
                // u -th row and v-th column, its desired value would be set 0.15.
                // In other words, this parameter determines the value of all
                // missing matrix elements.
                C = 0.15,
                // This argument enables one-class matrix factorization.
                LossFunction = MatrixFactorizationTrainer.LossFunctionType
                    .SquareLossOneClass
            };

            var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(
                options);

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            // Apply the trained model to the test set. Notice that training is a
            // partial 
            var prediction = model.Transform(mlContext.Data.LoadFromEnumerable(
                testData));

            var results = mlContext.Data.CreateEnumerable<MatrixElement>(prediction,
                false).ToList();
            // Feed the test data into the model and then iterate through a few
            // predictions.
            foreach (var pred in results.Take(15))
                Console.WriteLine($"Predicted value at row " +
                    $"{pred.MatrixRowIndex - 1} and column " +
                    $"{pred.MatrixColumnIndex - 1} is {pred.Score} and its " +
                    $"expected value is {pred.Value}.");

            // Expected output similar to:
            // Predicted value at row 0 and column 0 is 0.9873335 and its expected value is 1.
            // Predicted value at row 1 and column 0 is 0.1499522 and its expected value is 0.15.
            // Predicted value at row 2 and column 0 is 0.1499791 and its expected value is 0.15.
            // Predicted value at row 3 and column 0 is 0.1499254 and its expected value is 0.15.
            // Predicted value at row 4 and column 0 is 0.1499074 and its expected value is 0.15.
            // Predicted value at row 5 and column 0 is 0.1499968 and its expected value is 0.15.
            // Predicted value at row 6 and column 0 is 0.1499791 and its expected value is 0.15.
            // Predicted value at row 7 and column 0 is 0.1499805 and its expected value is 0.15.
            // Predicted value at row 8 and column 0 is 0.1500055 and its expected value is 0.15.
            // Predicted value at row 9 and column 0 is 0.1499199 and its expected value is 0.15.
            // Predicted value at row 10 and column 0 is 0.9873335 and its expected value is 1.
            // Predicted value at row 11 and column 0 is 0.1499522 and its expected value is 0.15.
            // Predicted value at row 12 and column 0 is 0.1499791 and its expected value is 0.15.
            // Predicted value at row 13 and column 0 is 0.1499254 and its expected value is 0.15.
            // Predicted value at row 14 and column 0 is 0.1499074 and its expected value is 0.15.
            //
            // Note: use the advanced options constructor to set the number of
            // threads to 1 for a deterministic behavior.

            // Assume that row index is user ID and column index game ID, the
            // following list contains the games recommended by the trained model.
            // Note that sometime, you may want to exclude training data from your
            // predicted results because those would represent games that were
            // already purchased. The variable topColumns stores two matrix elements
            // with the highest predicted scores on the 1st row.
            var topColumns = results.Where(element => element.MatrixRowIndex == 1)
                .OrderByDescending(element => element.Score).Take(2);

            Console.WriteLine("Top 2 predictions on the 1st row:");
            foreach (var top in topColumns)
                Console.WriteLine($"Predicted value at row " +
                    $"{top.MatrixRowIndex - 1} and column " +
                    $"{top.MatrixColumnIndex - 1} is {top.Score} and its " +
                    $"expected value is {top.Value}.");

            // Expected output similar to:
            // Top 2 predictions at the 2nd row:
            // Predicted value at row 0 and column 0 is 0.9871138 and its expected value is 1.
            // Predicted value at row 0 and column 10 is 0.9871138 and its expected value is 1.
        }

        // The following variables defines the shape of a matrix. Its shape is 
        // _synthesizedMatrixRowCount-by-_synthesizedMatrixColumnCount.
        // Because in ML.NET key type's minimal value is zero, the first row index
        // is always zero in C# data structure (e.g., MatrixColumnIndex=0 and 
        // MatrixRowIndex=0 in MatrixElement below specifies the value at the
        // upper-left corner in the training matrix). If user's row index
        // starts with 1, their row index 1 would be mapped to the 2nd row in matrix
        // factorization module and their first row may contain no values.
        // This behavior is also true to column index.
        private const uint _synthesizedMatrixColumnCount = 60;
        private const uint _synthesizedMatrixRowCount = 100;

        // A data structure used to encode a single value in matrix
        private class MatrixElement
        {
            // Matrix column index. Its allowed range is from 0 to
            // _synthesizedMatrixColumnCount - 1.
            [KeyType(_synthesizedMatrixColumnCount)]
            public uint MatrixColumnIndex { get; set; }
            // Matrix row index. Its allowed range is from 0 to
            // _synthesizedMatrixRowCount - 1.
            [KeyType(_synthesizedMatrixRowCount)]
            public uint MatrixRowIndex { get; set; }
            // The value at the MatrixColumnIndex-th column and the
            // MatrixRowIndex-th row.
            public float Value { get; set; }
            // The predicted value at the MatrixColumnIndex-th column and the
            // MatrixRowIndex-th row.
            public float Score { get; set; }
        }

        // Create an in-memory matrix as a list of tuples (column index, row index,
        // value). Notice that one-class matrix factorization handle scenerios where
        // only positive signals (e.g., on Facebook, only likes are recorded and no
        // dislike before) can be observed so that all values are set to 1.
        private static void GetOneClassMatrix(
            out List<MatrixElement> observedMatrix,
            out List<MatrixElement> fullMatrix)
        {
            // The matrix factorization model will be trained only using
            // observedMatrix but we will see it can learn all information carried
            // sin fullMatrix.
            observedMatrix = new List<MatrixElement>();
            fullMatrix = new List<MatrixElement>();
            for (uint i = 0; i < _synthesizedMatrixColumnCount; ++i)
                for (uint j = 0; j < _synthesizedMatrixRowCount; ++j)
                {
                    if ((i + j) % 10 == 0)
                    {
                        // Set observed elements' values to 1 (means like).
                        observedMatrix.Add(new MatrixElement()
                        {
                            MatrixColumnIndex = i,
                            MatrixRowIndex = j,
                            Value = 1,
                            Score = 0
                        });
                        fullMatrix.Add(new MatrixElement()
                        {
                            MatrixColumnIndex = i,
                            MatrixRowIndex = j,
                            Value = 1,
                            Score = 0
                        });
                    }
                    else
                        // Set unobserved elements' values to 0.15, a value smaller
                        // than observed values (means dislike).
                        fullMatrix.Add(new MatrixElement()
                        {
                            MatrixColumnIndex = i,
                            MatrixRowIndex = j,
                            Value = 0.15f,
                            Score = 0
                        });
                }
        }
    }
}
