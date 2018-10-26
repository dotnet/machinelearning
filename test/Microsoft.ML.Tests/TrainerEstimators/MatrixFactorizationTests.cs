// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Recommender;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using System;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        [Fact]
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
            var invalidData = new TextLoader(Env, GetLoaderArgs(labelColumnName, matrixColumnIndexColumnName + "Renamed", matrixRowIndexColumnName+"Renamed"))
                    .Read(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.testFilename)));

            var est = new MatrixFactorizationTrainer(Env, labelColumnName, matrixColumnIndexColumnName, matrixRowIndexColumnName, 
                advancedSettings:s=>
                {
                    s.NumIterations = 3;
                    s.NumThreads = 1;
                    s.K = 4;
                });

            TestEstimatorCore(est, data, invalidInput: invalidData);

            Done();
        }

        [Fact]
        public void MatrixFactorizationSimpleTrainAndPredict()
        {
            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                // Specific column names of the considered data set
                string labelColumnName = "Label";
                string userColumnName = "User";
                string itemColumnName = "Item";
                string scoreColumnName = "Score";

                // Create reader for both of training and test data sets
                var reader = new TextLoader(env, GetLoaderArgs(labelColumnName, userColumnName, itemColumnName));

                // Read training data as an IDataView object
                var data = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename)));

                // Create a pipeline with a single operator.
                var pipeline = new MatrixFactorizationTrainer(env, labelColumnName, userColumnName, itemColumnName, 
                    advancedSettings:s=>
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
                var mlContext = new MLContext();
                var metrices = mlContext.Regression.Evaluate(prediction, label: labelColumnName, score: scoreColumnName);

                // Determine if the selected metric is reasonable for different platforms
                double tolerance = Math.Pow(10, -6); // -7 sometime leads to a test failure on Mac
                if (Environment.OSVersion.Platform == PlatformID.Unix)
                {
                    // Unix case
                    var expectedUnixL2Error = 0.616821448679879; // Unix baseline
                    Assert.InRange(metrices.L2, expectedUnixL2Error - tolerance, expectedUnixL2Error + tolerance);
                }
                else if (Environment.OSVersion.Platform == PlatformID.MacOSX)
                {
                    // Mac case
                    var expectedMacL2Error = 0.61192207960271; // Mac baseline
                    Assert.InRange(metrices.L2, expectedMacL2Error - tolerance, expectedMacL2Error + tolerance);
                }
                else
                {
                    // Windows case
                    var expectedWindowsL2Error = 0.61528733643754685; // Windows baseline
                    Assert.InRange(metrices.L2, expectedWindowsL2Error - tolerance, expectedWindowsL2Error + tolerance);
                }
            }
        }

        [Fact]
        public void MatrixFactorizationStatic()
        {
            using (var env = new LocalEnvironment(seed: 1, conc: 1))
            {
                var dataPath = GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename);
                var dataSource = new MultiFileSource(dataPath);
                var ctx = new RegressionContext(env);
                var reader = TextLoader.CreateReader(env, c => (label: c.LoadFloat(0), matrixColumnIndex: c.LoadUInt32Key(1, 0, 19), matrixRowIndex: c.LoadUInt32Key(2, 0, 39)));

                var testData = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.testFilename)));

                // The parameter that will be into the onFit method below. The obtained predictor will be assigned to this variable
                // so that we will be able to touch it.
                MatrixFactorizationPredictor pred = null;

                // Create a statically-typed matrix factorization estimator. The MatrixFactorization's input and output defined in MatrixFactorizationStatic
                // tell what (aks a Scalar<float>) is expected. Notice that only one thread is used for deterministic outcome.
                var matrixFactorizationEstimator = reader.MakeNewEstimator()
                    .Append(r => (r.label, score: ctx.Trainers.MatrixFactorization(r.label, r.matrixRowIndex, r.matrixColumnIndex, onFit: p => pred = p,
                    advancedSettings: args => { args.NumThreads = 1; })));

                // Create a pipeline from the reader (the 1st step) and the matrix factorization estimator (the 2nd step).
                var pipe = reader.Append(matrixFactorizationEstimator);

                // pred will be assigned by the onFit method once the training process is finished, so pred must be null before training.
                Assert.Null(pred);

                // Train the pipeline on the given data file. Steps in the pipeline are sequentially fitted (by calling their Fit function).
                var model = pipe.Fit(dataSource);

                // pred got assigned so that one can inspect the predictor trained in pipeline.
                Assert.NotNull(pred);

                // Feed the data file into the trained pipeline. In the pipeline, it would be loaded by TextLoader (the 1st step) and then the output of the
                // TextLoader would be fed into MatrixFactorizationEstimator.
                var estimatedData = model.Read(dataSource);

                // After the training process, the metrics for regression problems can be computed.
                var metrics = ctx.Evaluate(estimatedData, r => r.label, r => r.score);

                // Naive test. Just make sure the pipeline runs.
                Assert.True(metrics.L2 > 0);
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
    }
}
