using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Recommender;
using Microsoft.ML.Runtime.RunTests;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        [Fact]
        public void MatrixFactorization_Estimator()
        {
            string labelColumnName = "Label";
            string xColumnName = "X";
            string yColumnName = "Y";
            var data = new TextLoader(Env, GetMfTrivialLoaderArgs(labelColumnName, xColumnName, yColumnName))
                    .Read(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename)));

            var est = new MatrixFactorizationTrainer(Env, labelColumnName, xColumnName, yColumnName, 
                advancedSettings:s=>
                {
                    s.NumIterations = 3;
                    s.NumThreads = 1;
                    s.K = 4;
                });

            TestEstimatorCore(est, data);

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
                var reader = new TextLoader(env, GetMfTrivialLoaderArgs(labelColumnName, userColumnName, itemColumnName));

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

                // Go through the whole test set and calculate the expected statistics.
                // TODO: use regression evaluator
                var mlContext = new MLContext();
                var expectedL2Error = 0.61528733643754685;
                double tolerance = System.Math.Pow(10, -DigitsOfPrecision);
                var metrices = mlContext.Regression.Evaluate(prediction, label: labelColumnName, score: scoreColumnName);
                Assert.InRange(metrices.L2, expectedL2Error - tolerance, expectedL2Error + tolerance);
            }
        }

        private TextLoader.Arguments GetMfTrivialLoaderArgs(string labelColumnName, string xColumnName, string yColumnName)
        {
            return new TextLoader.Arguments()
            {
                Separator = "\t",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column(labelColumnName, DataKind.R4, new [] { new TextLoader.Range(0) }),
                    new TextLoader.Column(xColumnName, DataKind.U4, new [] { new TextLoader.Range(1) }, new KeyRange(0, 19)),
                    new TextLoader.Column(yColumnName, DataKind.U4, new [] { new TextLoader.Range(2) }, new KeyRange(0, 39)),
                }
            };
        }
    }
}
