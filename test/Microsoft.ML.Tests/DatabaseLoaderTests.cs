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

                using (SqlCommand command = new SqlCommand(
                    "SELECT SepalLength, SepalWidth, PetalLength, PetalWidth, Label FROM IrisData",
                    connection))
                {
                    DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader(
                        new DatabaseLoader.Column() { Name = "SepalLength", Type = DbType.Single },
                        new DatabaseLoader.Column() { Name = "SepalWidth", Type = DbType.Single },
                        new DatabaseLoader.Column() { Name = "PetalLength", Type = DbType.Single },
                        new DatabaseLoader.Column() { Name = "PetalWidth", Type = DbType.Single },
                        new DatabaseLoader.Column() { Name = "Label", Type = DbType.Int32 }
                    );

                    IDataView trainingData = loader.Load(() => command.ExecuteReader());
                    //trainingData = mlContext.Data.Cache(trainingData, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Label");


                    var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                        .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                        //.AppendCacheCheckpoint(mlContext)
                        .Append(mlContext.MulticlassClassification.Trainers.LightGbm())
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                    var model = pipeline.Fit(trainingData);

                    var engine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);

                    Assert.Equal(1, engine.Predict(new IrisData()
                    {
                        SepalLength = 4.5f,
                        SepalWidth = 5.6f,
                        PetalLength = 0.5f,
                        PetalWidth = 0.5f,
                    }).PredictedLabel);
                    Assert.Equal(2, engine.Predict(new IrisData()
                    {
                        SepalLength = 4.9f,
                        SepalWidth = 2.4f,
                        PetalLength = 3.3f,
                        PetalWidth = 1.0f,
                    }).PredictedLabel);
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
