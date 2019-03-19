// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        [MatrixFactorizationFact]
        public void MatrixFactorization_Estimator()
        {
            string labelColumnName = "Label";
            string matrixColumnIndexColumnName = "Col";
            string matrixRowIndexColumnName = "Row";

            // This data contains three columns, Label, Col, and Row where Col and Row will be treated as the expected input names
            // of the trained matrix factorization model.
            var data = new TextLoader(Env, GetLoaderArgs(labelColumnName, matrixColumnIndexColumnName, matrixRowIndexColumnName))
                    .Load(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename)));

            // "invalidData" is not compatible to "data" because it contains columns Label, ColRenamed, and RowRenamed (no column is Col or Row).
            var invalidData = new TextLoader(Env, GetLoaderArgs(labelColumnName, matrixColumnIndexColumnName + "Renamed", matrixRowIndexColumnName + "Renamed"))
                    .Load(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.testFilename)));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = matrixColumnIndexColumnName,
                MatrixRowIndexColumnName = matrixRowIndexColumnName,
                LabelColumnName = labelColumnName,
                NumberOfIterations = 3,
                NumberOfThreads = 1,
                ApproximationRank = 4,
            };

            var est = ML.Recommendation().Trainers.MatrixFactorization(options);

            TestEstimatorCore(est, data, invalidInput: invalidData);

            Done();
        }

        [MatrixFactorizationFact]
        public void MatrixFactorizationSimpleTrainAndPredict()
        {
            var mlContext = new MLContext(seed: 1);

            // Specific column names of the considered data set
            string labelColumnName = "Label";
            string userColumnName = "User";
            string itemColumnName = "Item";
            string scoreColumnName = "Score";

            // Create reader for both of training and test data sets
            var reader = new TextLoader(mlContext, GetLoaderArgs(labelColumnName, userColumnName, itemColumnName));

            // Read training data as an IDataView object
            var data = reader.Load(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename)));

            // Create a pipeline with a single operator.
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = userColumnName,
                MatrixRowIndexColumnName = itemColumnName,
                LabelColumnName = labelColumnName,
                NumberOfIterations = 3,
                NumberOfThreads = 1, // To eliminate randomness, # of threads must be 1.
                ApproximationRank = 7,
            };

            var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(options);

            // Train a matrix factorization model.
            var model = pipeline.Fit(data);

            // Let's validate content of the model.
            Assert.Equal(model.Model.ApproximationRank, options.ApproximationRank);
            var leftMatrix = model.Model.LeftFactorMatrix;
            var rightMatrix = model.Model.RightFactorMatrix;
            Assert.Equal(leftMatrix.Count, model.Model.NumberOfRows * model.Model.ApproximationRank);
            Assert.Equal(rightMatrix.Count, model.Model.NumberOfColumns * model.Model.ApproximationRank);
            // MF produce different matrixes on different platforms, so at least test thier content on windows.
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                Assert.Equal(leftMatrix[0], (double)0.3091519, 5);
                Assert.Equal(leftMatrix[leftMatrix.Count - 1], (double)0.5639161, 5);
                Assert.Equal(rightMatrix[0], (double)0.243584976, 5);
                Assert.Equal(rightMatrix[rightMatrix.Count - 1], (double)0.380032182, 5);
            }
            // Read the test data set as an IDataView
            var testData = reader.Load(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.testFilename)));

            // Apply the trained model to the test set
            var prediction = model.Transform(testData);

            // Get output schema and check its column names
            var outputSchema = model.GetOutputSchema(data.Schema);
            var expectedOutputNames = new string[] { labelColumnName, userColumnName, itemColumnName, scoreColumnName };
            foreach (var col in outputSchema)
                Assert.True(col.Name == expectedOutputNames[col.Index]);

            // Retrieve label column's index from the test IDataView
            testData.Schema.TryGetColumnIndex(labelColumnName, out int labelColumnId);

            // Retrieve score column's index from the IDataView produced by the trained model
            prediction.Schema.TryGetColumnIndex(scoreColumnName, out int scoreColumnId);

            // Compute prediction errors
            var metrices = mlContext.Recommendation().Evaluate(prediction, labelColumnName: labelColumnName, scoreColumnName: scoreColumnName);

            // Determine if the selected metric is reasonable for different platforms
            double tolerance = Math.Pow(10, -7);
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                // Linux case
                var expectedUnixL2Error = 0.616821448679879; // Linux baseline
                Assert.InRange(metrices.MeanSquaredError, expectedUnixL2Error - tolerance, expectedUnixL2Error + tolerance);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                // The Mac case is just broken. Should be fixed later. Re-enable when done.
                // Mac case
                //var expectedMacL2Error = 0.61192207960271; // Mac baseline
                //Assert.InRange(metrices.L2, expectedMacL2Error - 5e-3, expectedMacL2Error + 5e-3); // 1e-7 is too small for Mac so we try 1e-5
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Windows case
                var expectedWindowsL2Error = 0.61528733643754685; // Windows baseline
                Assert.InRange(metrices.MeanSquaredError, expectedWindowsL2Error - tolerance, expectedWindowsL2Error + tolerance);
            }

            var modelWithValidation = pipeline.Fit(data, testData);
        }

        private TextLoader.Options GetLoaderArgs(string labelColumnName, string matrixColumnIndexColumnName, string matrixRowIndexColumnName)
        {
            return new TextLoader.Options()
            {
                Separator = "\t",
                HasHeader = true,
                Columns = new[]
                {
                    new TextLoader.Column(labelColumnName, DataKind.Single, new [] { new TextLoader.Range(0) }),
                    new TextLoader.Column(matrixColumnIndexColumnName, DataKind.UInt32, new [] { new TextLoader.Range(1) }, new KeyCount(20)),
                    new TextLoader.Column(matrixRowIndexColumnName, DataKind.UInt32, new [] { new TextLoader.Range(2) }, new KeyCount(40)),
                }
            };
        }

        // The following variables defines the shape of a matrix. Its shape is _synthesizedMatrixRowCount-by-_synthesizedMatrixColumnCount.
        // Because in ML.NET key type's minimal value is zero, the first row index is always zero in C# data structure (e.g., MatrixColumnIndex=0
        // and MatrixRowIndex=0 in MatrixElement below specifies the value at the upper-left corner in the training matrix). If user's row index 
        // starts with 1, their row index 1 would be mapped to the 2nd row in matrix factorization module and their first row may contain no values.
        // This behavior is also true to column index.
        const int _synthesizedMatrixFirstColumnIndex = 1;
        const int _synthesizedMatrixFirstRowIndex = 1;
        const int _synthesizedMatrixColumnCount = 60;
        const int _synthesizedMatrixRowCount = 100;

        internal class MatrixElement
        {
            // Matrix column index starts from 0 and is at most _synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex.
            [KeyType(_synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            // Matrix row index starts from 0 and is at most _synthesizedMatrixRowCount + _synthesizedMatrixRowCount.
            [KeyType(_synthesizedMatrixRowCount + _synthesizedMatrixRowCount)]
            public uint MatrixRowIndex;
            // The value at the MatrixColumnIndex-th column and the MatrixRowIndex-th row in the considered matrix.
            public float Value;
        }

        internal class MatrixElementForScore
        {
            [KeyType(_synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex)]
            public uint MatrixColumnIndex;
            [KeyType(_synthesizedMatrixRowCount + _synthesizedMatrixRowCount)]
            public uint MatrixRowIndex;
            public float Score;
        }

        [MatrixFactorizationFact]
        public void MatrixFactorizationInMemoryData()
        {
            // Create an in-memory matrix as a list of tuples (column index, row index, value).
            var dataMatrix = new List<MatrixElement>();
            for (uint i = _synthesizedMatrixFirstColumnIndex; i < _synthesizedMatrixFirstColumnIndex + _synthesizedMatrixColumnCount; ++i)
                for (uint j = _synthesizedMatrixFirstRowIndex; j < _synthesizedMatrixFirstRowIndex + _synthesizedMatrixRowCount; ++j)
                    dataMatrix.Add(new MatrixElement() { MatrixColumnIndex = i, MatrixRowIndex = j, Value = (i + j) % 5 });

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var dataView = ML.Data.LoadFromEnumerable(dataMatrix);

            // Create a matrix factorization trainer which may consume "Value" as the training label, "MatrixColumnIndex" as the
            // matrix's column index, and "MatrixRowIndex" as the matrix's row index.
            var mlContext = new MLContext(seed: 1);

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(MatrixElement.MatrixColumnIndex),
                MatrixRowIndexColumnName = nameof(MatrixElement.MatrixRowIndex),
                LabelColumnName = nameof(MatrixElement.Value),
                NumberOfIterations = 10,
                NumberOfThreads = 1, // To eliminate randomness, # of threads must be 1.
                ApproximationRank = 32,
            };

            var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(options);

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            // Check if the expected types in the trained model are expected.
            Assert.True(model.MatrixColumnIndexColumnName == "MatrixColumnIndex");
            Assert.True(model.MatrixRowIndexColumnName == "MatrixRowIndex");
            Assert.True(model.MatrixColumnIndexColumnType is KeyType);
            Assert.True(model.MatrixRowIndexColumnType is KeyType);
            var matColKeyType = (KeyType)model.MatrixColumnIndexColumnType;
            Assert.True(matColKeyType.Count == _synthesizedMatrixColumnCount + _synthesizedMatrixFirstColumnIndex);
            var matRowKeyType = (KeyType)model.MatrixRowIndexColumnType;
            Assert.True(matRowKeyType.Count == _synthesizedMatrixRowCount + _synthesizedMatrixRowCount);

            // Apply the trained model to the training set
            var prediction = model.Transform(dataView);

            // Calculate regression matrices for the prediction result
            var metrics = mlContext.Recommendation().Evaluate(prediction, labelColumnName: nameof(MatrixElement.Value),
                scoreColumnName: nameof(MatrixElementForScore.Score));

            // Native test. Just check the pipeline runs.
            Assert.True(metrics.MeanSquaredError < 0.1);

            // Create two two entries for making prediction. Of course, the prediction value, Score, is unknown so it's default.
            var testMatrix = new List<MatrixElementForScore>() {
                new MatrixElementForScore() { MatrixColumnIndex = 10, MatrixRowIndex = 7, Score = default },
                new MatrixElementForScore() { MatrixColumnIndex = 3, MatrixRowIndex = 6, Score = default } };

            // Again, convert the test data to a format supported by ML.NET.
            var testDataView = mlContext.Data.LoadFromEnumerable(testMatrix);

            // Feed the test data into the model and then iterate through all predictions.
            foreach (var pred in mlContext.Data.CreateEnumerable<MatrixElementForScore>(model.Transform(testDataView), false))
                Assert.True(pred.Score != 0);
        }

        internal class MatrixElementZeroBased
        {
            // Matrix column index starts from 0 and is at most _synthesizedMatrixColumnCount.
            [KeyType(_synthesizedMatrixColumnCount)]
            public uint MatrixColumnIndex;
            // Matrix row index starts from 0 and is at most _synthesizedMatrixRowCount.
            [KeyType(_synthesizedMatrixRowCount)]
            public uint MatrixRowIndex;
            // The value at the MatrixColumnIndex-th column and the MatrixRowIndex-th row in the considered matrix.
            public float Value;
        }

        internal class MatrixElementZeroBasedForScore
        {
            // Matrix column index starts from 0 and is at most _synthesizedMatrixColumnCount.
            // Contieuous=true means that all values from 0 to _synthesizedMatrixColumnCount are allowed keys.
            [KeyType(_synthesizedMatrixColumnCount)]
            public uint MatrixColumnIndex;
            // Matrix row index starts from 0 and is at most _synthesizedMatrixRowCount.
            // Contieuous=true means that all values from 0 to _synthesizedMatrixRowCount are allowed keys.
            [KeyType(_synthesizedMatrixRowCount)]
            public uint MatrixRowIndex;
            public float Score;
        }

        [MatrixFactorizationFact]
        public void MatrixFactorizationInMemoryDataZeroBaseIndex()
        {
            // Create an in-memory matrix as a list of tuples (column index, row index, value).
            // Iterators i and j are column and row indexes, respectively.
            var dataMatrix = new List<MatrixElementZeroBased>();
            for (uint i = 0; i < _synthesizedMatrixColumnCount; ++i)
                for (uint j = 0; j < _synthesizedMatrixRowCount; ++j)
                    dataMatrix.Add(new MatrixElementZeroBased() { MatrixColumnIndex = i, MatrixRowIndex = j, Value = (i + j) % 5 });

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var dataView = ML.Data.LoadFromEnumerable(dataMatrix);

            // Create a matrix factorization trainer which may consume "Value" as the training label, "MatrixColumnIndex" as the
            // matrix's column index, and "MatrixRowIndex" as the matrix's row index.
            var mlContext = new MLContext(seed: 1);

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(MatrixElement.MatrixColumnIndex),
                MatrixRowIndexColumnName = nameof(MatrixElement.MatrixRowIndex),
                LabelColumnName = nameof(MatrixElement.Value),
                NumberOfIterations = 100,
                NumberOfThreads = 1, // To eliminate randomness, # of threads must be 1.
                ApproximationRank = 32,
                LearningRate = 0.5,
            };

            var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(options);

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            // Check if the expected types in the trained model are expected.
            Assert.True(model.MatrixColumnIndexColumnName == nameof(MatrixElementZeroBased.MatrixColumnIndex));
            Assert.True(model.MatrixRowIndexColumnName == nameof(MatrixElementZeroBased.MatrixRowIndex));
            var matColKeyType = model.MatrixColumnIndexColumnType as KeyType;
            Assert.NotNull(matColKeyType);
            var matRowKeyType = model.MatrixRowIndexColumnType as KeyType;
            Assert.NotNull(matRowKeyType);
            Assert.True(matColKeyType.Count == _synthesizedMatrixColumnCount);
            Assert.True(matRowKeyType.Count == _synthesizedMatrixRowCount);

            // Apply the trained model to the training set
            var prediction = model.Transform(dataView);

            // Calculate regression matrices for the prediction result. It's a global
            var metrics = mlContext.Recommendation().Evaluate(prediction, labelColumnName: "Value", scoreColumnName: "Score");

            // Make sure the prediction error is not too large.
            Assert.InRange(metrics.MeanSquaredError, 0, 0.1);

            foreach (var pred in mlContext.Data.CreateEnumerable<MatrixElementZeroBasedForScore>(prediction, false))
                // Test data contains no out-of-range indexes (i.e., all indexes can be found in the training matrix),
                // so NaN should never happen.
                Assert.True(!float.IsNaN(pred.Score));

            // Create out-of-range examples and make sure their predictions are all NaN
            var invalidTestMatrix = new List<MatrixElementZeroBasedForScore>()
            {
                // An example with a matrix column index just greater than the maximum allowed value
                new MatrixElementZeroBasedForScore() { MatrixColumnIndex = _synthesizedMatrixFirstColumnIndex + _synthesizedMatrixColumnCount, MatrixRowIndex = _synthesizedMatrixFirstRowIndex, Score = default },
                // An example with a matrix row index just greater than the maximum allowed value
                new MatrixElementZeroBasedForScore() { MatrixColumnIndex = _synthesizedMatrixFirstColumnIndex, MatrixRowIndex = _synthesizedMatrixFirstRowIndex + _synthesizedMatrixRowCount, Score = default }
            };

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var invalidTestDataView = mlContext.Data.LoadFromEnumerable(invalidTestMatrix);

            // Apply the trained model to the examples with out-of-range indexes. 
            var invalidPrediction = model.Transform(invalidTestDataView);

            foreach (var pred in mlContext.Data.CreateEnumerable<MatrixElementZeroBasedForScore>(invalidPrediction, false))
                // The presence of out-of-range indexes may lead to NaN
                Assert.True(float.IsNaN(pred.Score));
        }

        // The following ingredients are used to define a 3-by-2 one-class
        // matrix used in a test, OneClassMatrixFactorizationInMemoryDataZeroBaseIndex,
        // for one-class matrix factorization. One-class matrix means that all
        // the available elements in the training matrix are 1. Such a matrix
        // is common. Let's use online game store as an example. Assume that
        // user IDs are row indexes and game IDs are column indexes. By
        // encoding all users' purchase history as a matrix (i.e., if the value
        // at the u-th row and the v-th column is 1, then the u-th user owns
        // the v-th game), a one-class matrix gets created because all matrix
        // elements are 1. If you train a prediction model from that matrix
        // using standard collaborative filtering, all your predictions would
        // be 1! One-class matrix factorization assumes unspecified matrix
        // entries are all 0 (or a small constant value selected by the user)
        // so that the trainined model can assign purchased itemas higher
        // scores than those not purchased.
        private const int _oneClassMatrixColumnCount = 2;
        private const int _oneClassMatrixRowCount = 3;

        private class OneClassMatrixElementZeroBased
        {
            [KeyType(_oneClassMatrixColumnCount)]
            public uint MatrixColumnIndex;
            [KeyType(_oneClassMatrixRowCount)]
            public uint MatrixRowIndex;
            public float Value;
        }

        private class OneClassMatrixElementZeroBasedForScore
        {
            [KeyType(_oneClassMatrixColumnCount)]
            public uint MatrixColumnIndex;
            [KeyType(_oneClassMatrixRowCount)]
            public uint MatrixRowIndex;
            public float Value;
            public float Score;
        }

        [MatrixFactorizationFact]
        public void OneClassMatrixFactorizationInMemoryDataZeroBaseIndex()
        {
            // Create an in-memory matrix as a list of tuples (column index, row index, value). For one-class matrix
            // factorization problem, unspecified matrix elements are all a constant provided by user. If that constant is 0.15,
            // the following list means a 3-by-2 training matrix with elements:
            //   (0, 0, 1), (1, 1, 1), (0, 2, 1), (0, 1, 0.15), (1, 0, 0.15), (1, 2, 0.15).
            // because matrix elements at (0, 1), (1, 0), and (1, 2) are not specified. Below is a visualization of the training matrix.
            //   [1, ?]
            //   |?, 1| where ? will be set to 0.15 by user when creating the trainer.
            //   [1, ?]
            var dataMatrix = new List<OneClassMatrixElementZeroBased>();
            dataMatrix.Add(new OneClassMatrixElementZeroBased() { MatrixColumnIndex = 0, MatrixRowIndex = 0, Value = 1 });
            dataMatrix.Add(new OneClassMatrixElementZeroBased() { MatrixColumnIndex = 1, MatrixRowIndex = 1, Value = 1 });
            dataMatrix.Add(new OneClassMatrixElementZeroBased() { MatrixColumnIndex = 0, MatrixRowIndex = 2, Value = 1 });

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var dataView = ML.Data.LoadFromEnumerable(dataMatrix);

            // Create a matrix factorization trainer which may consume "Value" as the training label, "MatrixColumnIndex" as the
            // matrix's column index, and "MatrixRowIndex" as the matrix's row index.
            var mlContext = new MLContext(seed: 1);

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(MatrixElement.MatrixColumnIndex),
                MatrixRowIndexColumnName = nameof(MatrixElement.MatrixRowIndex),
                LabelColumnName = nameof(MatrixElement.Value),
                LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
                NumberOfIterations = 100,
                NumberOfThreads = 1, // To eliminate randomness, # of threads must be 1.
                Lambda = 0.025, // Let's test non-default regularization coefficient.
                ApproximationRank = 16,
                Alpha = 0.01, // Importance coefficient of loss function over matrix elements not specified in the input matrix.
                C = 0.15, // Desired value for matrix elements not specified in the input matrix.
            };

            var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(options);

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            // Apply the trained model to the training set.
            var prediction = model.Transform(dataView);

            // Calculate regression matrices for the prediction result.
            var metrics = mlContext.Recommendation().Evaluate(prediction, labelColumnName: "Value", scoreColumnName: "Score");

            // Make sure the prediction error is not too large.
            Assert.InRange(metrics.MeanSquaredError, 0, 0.0016);

            // Create data for testing. Note that the 2nd element is not specified in the training data so it should
            // be close to the constant specified by s.C = 0.15. Comparing with the data structure used in training phase,
            // one extra float is added into OneClassMatrixElementZeroBasedForScore for storing the prediction result. Note
            // that the prediction engine may ignore Value and assign the predicted value to Score.
            var testDataMatrix = new List<OneClassMatrixElementZeroBasedForScore>();
            testDataMatrix.Add(new OneClassMatrixElementZeroBasedForScore() { MatrixColumnIndex = 0, MatrixRowIndex = 0, Value = 0, Score = 0 });
            testDataMatrix.Add(new OneClassMatrixElementZeroBasedForScore() { MatrixColumnIndex = 1, MatrixRowIndex = 2, Value = 0, Score = 0 });

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var testDataView = ML.Data.LoadFromEnumerable(testDataMatrix);

            // Apply the trained model to the test data.
            var testPrediction = model.Transform(testDataView);

            var testResults = mlContext.Data.CreateEnumerable<OneClassMatrixElementZeroBasedForScore>(testPrediction, false).ToList();
            // Positive example (i.e., examples can be found in dataMatrix) is close to 1.
            CompareNumbersWithTolerance(0.982391, testResults[0].Score, digitsOfPrecision: 5);
            // Negative example (i.e., examples can not be found in dataMatrix) is close to 0.15 (specified by s.C = 0.15 in the trainer).
            CompareNumbersWithTolerance(0.141411, testResults[1].Score, digitsOfPrecision: 5);
        }

        [MatrixFactorizationFact]
        public void MatrixFactorizationBackCompat()
        {
            // This test is meant to check backwards compatibility after the change that removed Min and Contiguous from KeyType.
            // The model that we are loading in this test was generated using ML.Model.Save() on:

            //var dataMatrix = new List<OneClassMatrixElementZeroBased>();
            //dataMatrix.Add(new OneClassMatrixElementZeroBased() { MatrixColumnIndex = 0, MatrixRowIndex = 0, Value = 1 });
            //dataMatrix.Add(new OneClassMatrixElementZeroBased() { MatrixColumnIndex = 1, MatrixRowIndex = 1, Value = 1 });
            //dataMatrix.Add(new OneClassMatrixElementZeroBased() { MatrixColumnIndex = 0, MatrixRowIndex = 2, Value = 1 });
            //// Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            //var dataView = ML.Data.ReadFromEnumerable(dataMatrix);
            //var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(
            //    nameof(OneClassMatrixElementZeroBased.MatrixColumnIndex),
            //    nameof(OneClassMatrixElementZeroBased.MatrixRowIndex),
            //    nameof(OneClassMatrixElementZeroBased.Value),
            //    advancedSettings: s =>
            //    {
            //        s.LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass;
            //        s.NumIterations = 100;
            //        s.NumThreads = 1; // To eliminate randomness, # of threads must be 1.
            //        // Let's test non-default regularization coefficient.
            //        s.Lambda = 0.025;
            //        s.K = 16;
            //        // Importance coefficient of loss function over matrix elements not specified in the input matrix.
            //        s.Alpha = 0.01;
            //        // Desired value for matrix elements not specified in the input matrix.
            //        s.C = 0.15;
            //    });
            // Train a matrix factorization model.
            //var model = pipeline.Fit(dataView);

            var mlContext = new MLContext(seed: 1);

            // Test that we can load model after KeyType change (removed Min and Contiguous).            
            var modelPath = GetDataPath("backcompat", "matrix-factorization-model.zip");
            ITransformer model;
            using (var ch = Env.Start("load"))
            {
                using (var fs = File.OpenRead(modelPath))
                {
                    model = ML.Model.Load(fs, out var schema);
                    // This model was saved without the input schema.
                    Assert.Null(schema);
                }
            }

            // Create data for testing. Note that the 2nd element is not specified in the training data so it should
            // be close to the constant specified by s.C = 0.15. Comparing with the data structure used in training phase,
            // one extra float is added into OneClassMatrixElementZeroBasedForScore for storing the prediction result. Note
            // that the prediction engine may ignore Value and assign the predicted value to Score.
            var testDataMatrix = new List<OneClassMatrixElementZeroBasedForScore>();
            testDataMatrix.Add(new OneClassMatrixElementZeroBasedForScore() { MatrixColumnIndex = 0, MatrixRowIndex = 0, Value = 0, Score = 0 });
            testDataMatrix.Add(new OneClassMatrixElementZeroBasedForScore() { MatrixColumnIndex = 1, MatrixRowIndex = 2, Value = 0, Score = 0 });

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var testDataView = ML.Data.LoadFromEnumerable(testDataMatrix);

            // Apply the trained model to the test data.
            var testPrediction = model.Transform(testDataView);

            var testResults = mlContext.Data.CreateEnumerable<OneClassMatrixElementZeroBasedForScore>(testPrediction, false).ToList();
            // Positive example (i.e., examples can be found in dataMatrix) is close to 1.
            CompareNumbersWithTolerance(0.982391, testResults[0].Score, digitsOfPrecision: 5);
            // Negative example (i.e., examples can not be found in dataMatrix) is close to 0.15 (specified by s.C = 0.15 in the trainer).
            CompareNumbersWithTolerance(0.141411, testResults[1].Score, digitsOfPrecision: 5);
        }

        [MatrixFactorizationFact]
        public void OneClassMatrixFactorizationWithUnseenColumnAndRow()
        {
            // Create an in-memory matrix as a list of tuples (column index, row index, value). For one-class matrix
            // factorization problem, unspecified matrix elements are all a constant provided by user. If that constant is 0.15,
            // the following list means a 3-by-2 training matrix with elements:
            //   (0, 0, 1), (0, 1, 1), (1, 0, 0.15), (1, 1, 0.15), (0, 2, 0.15), (1, 2, 0.15).
            // because matrix elements at (1, 0), (1, 1), (0, 2), and (1, 2) are not specified. Below is a visualization of the training matrix.
            //   [1, ?]
            //   |1, ?| where ? will be set to 0.15 by user when creating the trainer.
            //   [?, ?]
            // Note that the second column and the third row are called unseen because they contain no training element (i.e., all its values are "?"s).
            var dataMatrix = new List<OneClassMatrixElementZeroBased>();
            dataMatrix.Add(new OneClassMatrixElementZeroBased() { MatrixColumnIndex = 0, MatrixRowIndex = 0, Value = 1 });
            dataMatrix.Add(new OneClassMatrixElementZeroBased() { MatrixColumnIndex = 0, MatrixRowIndex = 1, Value = 1 });

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var dataView = ML.Data.LoadFromEnumerable(dataMatrix);

            // Create a matrix factorization trainer which may consume "Value" as the training label, "MatrixColumnIndex" as the
            // matrix's column index, and "MatrixRowIndex" as the matrix's row index.
            var mlContext = new MLContext(seed: 1);

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(MatrixElement.MatrixColumnIndex),
                MatrixRowIndexColumnName = nameof(MatrixElement.MatrixRowIndex),
                LabelColumnName = nameof(MatrixElement.Value),
                LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
                NumberOfIterations = 100,
                NumberOfThreads = 1, // To eliminate randomness, # of threads must be 1.
                Lambda = 0.025, // Let's test non-default regularization coefficient.
                ApproximationRank = 16,
                Alpha = 0.01, // Importance coefficient of loss function over matrix elements not specified in the input matrix.
                C = 0.15, // Desired value for matrix elements not specified in the input matrix.
            };

            var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(options);

            // Train a matrix factorization model.
            var model = pipeline.Fit(dataView);

            // Apply the trained model to the training set.
            var prediction = model.Transform(dataView);

            // Calculate regression matrices for the prediction result.
            var metrics = mlContext.Recommendation().Evaluate(prediction, labelColumnName: "Value", scoreColumnName: "Score");

            // Make sure the prediction error is not too large.
            Assert.InRange(metrics.MeanSquaredError, 0, 0.0016);

            // Create data for testing.
            var testDataMatrix = new List<OneClassMatrixElementZeroBasedForScore>();
            testDataMatrix.Add(new OneClassMatrixElementZeroBasedForScore() { MatrixColumnIndex = 0, MatrixRowIndex = 0, Value = 0, Score = 0 });
            testDataMatrix.Add(new OneClassMatrixElementZeroBasedForScore() { MatrixColumnIndex = 1, MatrixRowIndex = 0, Value = 0, Score = 0 });
            testDataMatrix.Add(new OneClassMatrixElementZeroBasedForScore() { MatrixColumnIndex = 1, MatrixRowIndex = 2, Value = 0, Score = 0 });

            // Convert the in-memory matrix into an IDataView so that ML.NET components can consume it.
            var testDataView = ML.Data.LoadFromEnumerable(testDataMatrix);

            // Apply the trained model to the test data.
            var testPrediction = model.Transform(testDataView);

            var testResults = mlContext.Data.CreateEnumerable<OneClassMatrixElementZeroBasedForScore>(testPrediction, false).ToList();
            // Positive example (i.e., examples can be found in dataMatrix) is close to 1.
            CompareNumbersWithTolerance(0.9823623, testResults[0].Score, digitsOfPrecision: 5);
            // Negative examples' scores (i.e., examples can not be found in dataMatrix) are closer
            // to 0.15 (specified by s.C = 0.15 in the trainer) than positive example's score.
            CompareNumbersWithTolerance(0.05511549, testResults[1].Score, digitsOfPrecision: 5);
            CompareNumbersWithTolerance(0.00316973357, testResults[2].Score, digitsOfPrecision: 5);
        }
    }
}