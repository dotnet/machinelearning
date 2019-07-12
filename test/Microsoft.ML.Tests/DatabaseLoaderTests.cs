// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data;
using System.Data.SqlClient;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class DatabaseLoaderTests : BaseTestClass
    {
        public DatabaseLoaderTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void TryItOut()
        {
            var mlContext = new MLContext(seed: 1);

            var connectionString = @"Server=(localdb)\mssqllocaldb;Database=EFGetStarted.AspNetCore.NewDb;Trusted_Connection=True;ConnectRetryCount=0";
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                connection.Open();

                using (SqlCommand command = new SqlCommand("SELECT SepalLength, SepalWidth, PetalLength, PetalWidth, Label FROM IrisData", connection))
                {
                    var loader = new DatabaseLoader(mlContext, new DatabaseLoader.Options()
                    {
                        Columns = new[]
                        {
                            new DatabaseLoader.Column() { Name = "SepalLength", Source = 0},
                            new DatabaseLoader.Column() { Name = "SepalWidth", Source = 1},
                            new DatabaseLoader.Column() { Name = "PetalLength", Source = 2},
                            new DatabaseLoader.Column() { Name = "PetalWidth", Source = 3},
                            new DatabaseLoader.Column() { Name = "Label", Type = DbType.Int32, Source = 4},
                        }
                    });

                    var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                        .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                        //.AppendCacheCheckpoint(mlContext)
                        .Append(mlContext.MulticlassClassification.Trainers.LightGbm())
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                    IDataView trainingData = loader.Load(() => command.ExecuteReader());
                    //trainingData = mlContext.Data.Cache(trainingData, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Label");
                    var model = pipeline.Fit(trainingData);

                    var engine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);

                    var prediction = engine.Predict(new IrisData()
                    {
                        SepalLength = 4.5f,
                        SepalWidth = 5.6f,
                        PetalLength = 0.5f,
                        PetalWidth = 0.5f,
                    });

                    Assert.Equal(2, prediction.PredictedLabel);
                }
            }
        }

        public class IrisData
        {
            public float SepalLength;
            public float SepalWidth;
            public float PetalLength;
            public float PetalWidth;
            public int Label;
        }

        public class IrisPrediction
        {
            public int PredictedLabel;
            public float[] Score;
        }

    }
}
