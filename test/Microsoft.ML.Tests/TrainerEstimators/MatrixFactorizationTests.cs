// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // This test is being fixed as part of issue #1441.
        public void MatrixFactorization_Estimator()
        {
            string labelColumnName = "Label";
            string matrixColumnIndexColumnName = "Col";
            string matrixRowIndexColumnName = "Row";

            // This data contains three columns, Label, Col, and Row where Col and Row will be treated as the expected input names
            // of the trained matrix factorization model.
            var data = new TextLoader(Env, GetLoaderArgs(labelColumnName, matrixColumnIndexColumnName, matrixRowIndexColumnName))
                    .Read(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename)));

            // "invalidData" is not compatible to "data" because it contains columns Label, ColRenamed, and RowRenamed (no column is Col or Row).
            var invalidData = new TextLoader(Env, GetLoaderArgs(labelColumnName, matrixColumnIndexColumnName + "Renamed", matrixRowIndexColumnName + "Renamed"))
                    .Read(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.testFilename)));

            var est = new MatrixFactorizationTrainer(Env, labelColumnName, matrixColumnIndexColumnName, matrixRowIndexColumnName,
                advancedSettings: s =>
                {
                    s.NumIterations = 3;
                    s.NumThreads = 1;
                    s.K = 4;
                });

            TestEstimatorCore(est, data, invalidInput: invalidData);

            Done();
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // This test is being fixed as part of issue #1441.
        public void MatrixFactorizationSimpleTrainAndPredict()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Specific column names of the considered data set
            string labelColumnName = "Label";
            string userColumnName = "User";
            string itemColumnName = "Item";
            string scoreColumnName = "Score";

            // Create reader for both of training and test data sets
            var reader = new TextLoader(mlContext, GetLoaderArgs(labelColumnName, userColumnName, itemColumnName));

            // Read training data as an IDataView object
            var data = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename)));

            // Create a pipeline with a single operator.
            var pipeline = new MatrixFactorizationTrainer(mlContext, labelColumnName, userColumnName, itemColumnName,
                advancedSettings: s =>
                {
                    s.NumIterations = 3;
                    s.NumThreads = 1; // To eliminate randomness, # of threads must be 1.
                    s.K = 7;
                });

            // Train a matrix factorization model.
            var model = pipeline.Fit(data);

            // Read the test data set as an IDataView
            var testData = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.testFilename)));

            // Apply the trained model to the test set
            var prediction = model.Transform(testData);

            // Get output schema and check its column names
            var outputSchema = model.GetOutputSchema(data.Schema);
            var expectedOutputNames = new string[] { labelColumnName, userColumnName, itemColumnName, scoreColumnName };
            foreach (var (i, col) in outputSchema.GetColumns())
                Assert.True(col.Name == expectedOutputNames[i]);

            // Retrieve label column's index from the test IDataView
            testData.Schema.TryGetColumnIndex(labelColumnName, out int labelColumnId);

            // Retrieve score column's index from the IDataView produced by the trained model
            prediction.Schema.TryGetColumnIndex(scoreColumnName, out int scoreColumnId);

            // Compute prediction errors
            var metrices = mlContext.Regression.Evaluate(prediction, label: labelColumnName, score: scoreColumnName);

            // Determine if the selected metric is reasonable for different platforms
            double tolerance = Math.Pow(10, -7);
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                // Linux case
                var expectedUnixL2Error = 0.616821448679879; // Linux baseline
                Assert.InRange(metrices.L2, expectedUnixL2Error - tolerance, expectedUnixL2Error + tolerance);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))

            {
                // Mac case
                var expectedMacL2Error = 0.61192207960271; // Mac baseline
                Assert.InRange(metrices.L2, expectedMacL2Error - 5e-3, expectedMacL2Error + 5e-3); // 1e-7 is too small for Mac so we try 1e-5
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Windows case
                var expectedWindowsL2Error = 0.61528733643754685; // Windows baseline
                Assert.InRange(metrices.L2, expectedWindowsL2Error - tolerance, expectedWindowsL2Error + tolerance);
            }
        }

        private TextLoader.Arguments GetLoaderArgs(string labelColumnName, string matrixColumnIndexColumnName, string matrixRowIndexColumnName)
        {
            return new TextLoader.Arguments()
            {
                Separator = "\t",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column(labelColumnName, DataKind.R4, new [] { new TextLoader.Range(0) }),
                    new TextLoader.Column(matrixColumnIndexColumnName, DataKind.U4, new [] { new TextLoader.Range(1) }, new KeyRange(0, 19)),
                    new TextLoader.Column(matrixRowIndexColumnName, DataKind.U4, new [] { new TextLoader.Range(2) }, new KeyRange(0, 39)),
                }
            };
        }

        // The following variables defines the shape of a matrix. Its shape is _synthesizedMatrixRowCount-by-_synthesizedMatrixColumnCount.
        // The variable _synthesizedMatrixFirstRowIndex indicates the integer that would be mapped to the first row index. If user data uses
        // 0-based indices for rows, _synthesizedMatrixFirstRowIndex can be set to 0. Similarly, for 1-based indices, _synthesizedMatrixFirstRowIndex
        // could be 1.
        const int _synthesizedMatrixFirstColumnIndex = 1;
        const int _synthesizedMatrixFirstRowIndex = 1;
        const int _synthesizedMatrixColumnCount = 60;
        const int _synthesizedMatrixRowCount = 100;

        internal class MatrixElement
        {
            // Matrix column index starts from 1 and is at most _synthesizedMatrixColumnCount.
            // Contieuous=true means that all values from 1 to _synthesizedMatrixColumnCount are allowed keys.
            [KeyType(Contiguous = true, Count = _synthesizedMatrixColumnCount, Min = _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            // Matrix row index starts from 1 and is at most _synthesizedMatrixRowCount.
            // Contieuous=true means that all values from 1 to _synthesizedMatrixRowCount are allowed keys.
            [KeyType(Contiguous = true, Count = _synthesizedMatrixRowCount, Min = _synthesizedMatrixFirstRowIndex)]
            public uint MatrixRowIndex;
            // The value at the MatrixColumnIndex-th column and the MatrixRowIndex-th row in the considered matrix.
            public float Value;
        }

        internal class MatrixElementForScore
        {
            [KeyType(Contiguous = true, Count = _synthesizedMatrixColumnCount, Min = _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            [KeyType(Contiguous = true, Count = _synthesizedMatrixRowCount, Min = _synthesizedMatrixFirstRowIndex)]
            public uint MatrixRowIndex;
            public float Score;
        }


        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // This test is being fixed as part of issue #1441.
        public void MatrixFactorizationInMemoryData()
        {
            // Create an in-memory matrix as a list of tuples (column index, row index, value).
            var dataMatrix = new List<MatrixElement>();
            for (uint i = _synthesizedMatrixFirstColumnIndex; i < _synthesizedMatrixFirstColumnIndex + _synthesizedMatrixColumnCount; ++i)
                for (uint j = _synthesizedMatrixFirstRowIndex; j < _synthesizedMatrixFirstRowIndex + _synthesizedMatrixRowCount; ++j)
                    dataMatrix.Add(new MatrixElement() { MatrixColumnIndex = i, MatrixRowIndex = j, Value = (i + j) % 5 });

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var dataView = ComponentCreation.CreateDataView(Env, dataMatrix);

            // Create a matrix factorization trainer which may consume "Value" as the training label, "MatrixColumnIndex" as the
            // matrix's column index, and "MatrixRowIndex" as the matrix's row index.
            var mlContext = new MLContext(seed: 1, conc: 1);
            var pipeline = new MatrixFactorizationTrainer(mlContext, "Value", "MatrixColumnIndex", "MatrixRowIndex",
                advancedSettings: s =>
                {
                    s.NumIterations = 10;
                    s.NumThreads = 1; // To eliminate randomness, # of threads must be 1.
                    s.K = 32;
                });

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            Assert.True(model.MatrixColumnIndexColumnName == "MatrixColumnIndex");
            Assert.True(model.MatrixRowIndexColumnName == "MatrixRowIndex");
            Assert.True(model.MatrixColumnIndexColumnType.IsKey);
            Assert.True(model.MatrixRowIndexColumnType.IsKey);
            var matColKeyType = model.MatrixColumnIndexColumnType.AsKey;
            Assert.True(matColKeyType.Min == _synthesizedMatrixFirstColumnIndex);
            Assert.True(matColKeyType.Count == _synthesizedMatrixColumnCount);
            var matRowKeyType = model.MatrixRowIndexColumnType.AsKey;
            Assert.True(matRowKeyType.Min == _synthesizedMatrixFirstRowIndex);
            Assert.True(matRowKeyType.Count == _synthesizedMatrixRowCount);

            // Apply the trained model to the training set
            var prediction = model.Transform(dataView);

            // Calculate regression matrices for the prediction result
            var metrics = mlContext.Regression.Evaluate(prediction, label: "Value", score: "Score");

            // Native test. Just check the pipeline runs.
            Assert.True(metrics.L2 < 0.1);

            // Create two two entries for making prediction. Of course, the prediction value, Score, is unknown so it's default.
            var testMatrix = new List<MatrixElementForScore>() {
                new MatrixElementForScore() { MatrixColumnIndex = 1000, MatrixRowIndex = 7, Score = default },
                new MatrixElementForScore() { MatrixColumnIndex = 3, MatrixRowIndex = 6, Score = default } };

            // Again, convert the test data to a format supported by ML.NET.
            var testDataView = ComponentCreation.CreateDataView(mlContext, testMatrix);

            // Feed the test data into the model and then iterate through all predictions.
            foreach (var pred in model.Transform(testDataView).AsEnumerable<MatrixElementForScore>(mlContext, false))
                Console.WriteLine("Predicted value at row {0} and column {1} is {2}", pred.MatrixRowIndex, pred.MatrixColumnIndex, pred.Score);
        }
    }
}