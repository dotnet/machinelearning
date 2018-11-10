// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;

// NOTE: WHEN ADDING TO THE FILE, ALWAYS APPEND TO THE END OF IT. 
// If you change the existinc content, check that the files referencing it in the XML documentation are still correct, as they reference
// line by line. 
namespace Microsoft.ML.Samples.Dynamic
{
    public partial class TrainerSamples
    {
        // The following variables defines the shape of a matrix. Its shape is _synthesizedMatrixRowCount-by-_synthesizedMatrixColumnCount.
        // The variable _synthesizedMatrixFirstRowIndex indicates the integer that would be mapped to the first row index. If user data uses
        // 0-based indices for rows, _synthesizedMatrixFirstRowIndex can be set to 0. Similarly, for 1-based indices, _synthesizedMatrixFirstRowIndex
        // could be 1.
        const int _synthesizedMatrixFirstColumnIndex = 1;
        const int _synthesizedMatrixFirstRowIndex = 1;
        const int _synthesizedMatrixColumnCount = 60;
        const int _synthesizedMatrixRowCount = 100;

        // A data structure used to encode a single value in matrix
        internal class MatrixElement
        {
            // Matrix column index starts from _synthesizedMatrixFirstColumnIndex and is at most
            // _synthesizedMatrixFirstColumnIndex + _synthesizedMatrixColumnCount - 1.
            // Contieuous=true means that all values between the min and max indexes are all allowed.
            [KeyType(Contiguous = true, Count = _synthesizedMatrixColumnCount, Min = _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            // Matrix row index starts from _synthesizedMatrixFirstRowIndex and is at most
            // _synthesizedMatrixFirstRowIndex + _synthesizedMatrixRowCount - 1.
            // Contieuous=true means that all values between the min and max indexes are all allowed.
            [KeyType(Contiguous = true, Count = _synthesizedMatrixRowCount, Min = _synthesizedMatrixFirstRowIndex)]
            public uint MatrixRowIndex;
            // The value at the column MatrixColumnIndex and row MatrixRowIndex.
            public float Value;
        }

        // A data structure used to encode prediction result. Comparing with MatrixElement, The field Value in MatrixElement is
        // renamed to Score because Score is the default name of matrix factorization's output.
        internal class MatrixElementForScore
        {
            [KeyType(Contiguous = true, Count = _synthesizedMatrixColumnCount, Min = _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            [KeyType(Contiguous = true, Count = _synthesizedMatrixRowCount, Min = _synthesizedMatrixFirstRowIndex)]
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
            var dataView = ComponentCreation.CreateDataView(mlContext, dataMatrix);

            // Create a matrix factorization trainer which may consume "Value" as the training label, "MatrixColumnIndex" as the
            // matrix's column index, and "MatrixRowIndex" as the matrix's row index. Here nameof(...) is used to extract field
            // names' in MatrixElement class.
            var pipeline = new MatrixFactorizationTrainer(mlContext, 
                nameof(MatrixElement.MatrixColumnIndex),
                nameof(MatrixElement.MatrixRowIndex),
                nameof(MatrixElement.Value),
                advancedSettings: s =>
                {
                    s.NumIterations = 10;
                    s.NumThreads = 1; // To eliminate randomness, # of threads must be 1.
                    s.K = 32;
                });

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            // Apply the trained model to the training set.
            var prediction = model.Transform(dataView);

            // Calculate regression matrices for the prediction result.
            var metrics = mlContext.Regression.Evaluate(prediction,
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
            var testDataView = ComponentCreation.CreateDataView(mlContext, testMatrix);

            // Feed the test data into the model and then iterate through all predictions.
            foreach (var pred in model.Transform(testDataView).AsEnumerable<MatrixElementForScore>(mlContext, false))
                Console.WriteLine($"Predicted value at row {pred.MatrixRowIndex} and column {pred.MatrixColumnIndex} is {pred.Score}");
        }
    }
}
