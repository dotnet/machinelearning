// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// the alignment of the usings with the methods is intentional so they can display on the same level in the docs site. 
        using Microsoft.ML.Runtime.Api;
        using Microsoft.ML.Runtime.Data;
        using Microsoft.ML.Trainers;
        using System.Collections.Generic;

// NOTE: WHEN ADDING TO THE FILE, ALWAYS APPEND TO THE END OF IT. 
// If you change the existinc content, check that the files referencing it in the XML documentation are still correct, as they reference
// line by line. 
namespace Microsoft.ML.Samples.Static
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
            // The value at the column MatrixColumnIndex and row MatrixRowIndexin.
            public float Value;
        }

        // This example first creates in-memory data and then use it to train a matrix factorization model. Afterward, quality metrics are reported.
        public static void MatrixFactorizationInMemoryData()
        {
            // Create an in-memory matrix as a list of tuples (column index, row index, value).
            var dataMatrix = new List<MatrixElement>();
            for (uint i = _synthesizedMatrixFirstColumnIndex; i < _synthesizedMatrixFirstColumnIndex + _synthesizedMatrixColumnCount; ++i)
                for (uint j = _synthesizedMatrixFirstRowIndex; j < _synthesizedMatrixFirstRowIndex + _synthesizedMatrixRowCount; ++j)
                    dataMatrix.Add(new MatrixElement() { MatrixColumnIndex = i, MatrixRowIndex = j, Value = (i + j) % 5 });

            // Creating the ML.Net IHostEnvironment object, needed for the pipeline
            var mlContext = new MLContext(seed: 0, conc: 1);

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var dataView = ComponentCreation.CreateDataView(mlContext, dataMatrix);

            // Create a matrix factorization trainer which may consume "Value" as the training label, "MatrixColumnIndex" as the
            // matrix's column index, and "MatrixRowIndex" as the matrix's row index.
            var pipeline = new MatrixFactorizationTrainer(mlContext, "Value", "MatrixColumnIndex", "MatrixRowIndex",
                advancedSettings: s =>
                {
                    s.NumIterations = 10;
                    s.NumThreads = 1; // To eliminate randomness, # of threads must be 1.
                    s.K = 32;
                });

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            // Apply the trained model to the training set
            var prediction = model.Transform(dataView);

            // Calculate regression matrices for the prediction result
            var metrics = mlContext.Regression.Evaluate(prediction, label: "Value", score: "Score");
        }
    }
}
