// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data;
using System.Data.SqlClient;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
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

        [LightGBMFact]
        public void IrisLightGbm()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // https://github.com/dotnet/machinelearning/issues/4156
                return;
            }

            var mlContext = new MLContext(seed: 1);

            var connectionString = GetConnectionString(TestDatasets.irisDb.name);
            var commandText = $@"SELECT * FROM ""{TestDatasets.irisDb.trainFilename}""";

            var loaderColumns = new DatabaseLoader.Column[]
            {
                new DatabaseLoader.Column() { Name = "Label", Type = DbType.Int32 },
                new DatabaseLoader.Column() { Name = "SepalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "SepalWidth", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalWidth", Type = DbType.Single }
            };

            var loader = mlContext.Data.CreateDatabaseLoader(loaderColumns);

            var databaseSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, commandText);

            var trainingData = loader.Load(databaseSource);

            IEstimator<ITransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainingData);

            var engine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);

            Assert.Equal(0, engine.Predict(new IrisData()
            {
                SepalLength = 4.5f,
                SepalWidth = 5.6f,
                PetalLength = 0.5f,
                PetalWidth = 0.5f,
            }).PredictedLabel);

            Assert.Equal(1, engine.Predict(new IrisData()
            {
                SepalLength = 4.9f,
                SepalWidth = 2.4f,
                PetalLength = 3.3f,
                PetalWidth = 1.0f,
            }).PredictedLabel);
        }

        [LightGBMFact]
        public void IrisLightGbmWithLoadColumnName()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // https://github.com/dotnet/machinelearning/issues/4156
                return;
            }

            var mlContext = new MLContext(seed: 1);

            var connectionString = GetConnectionString(TestDatasets.irisDb.name);
            var commandText = $@"SELECT Label as [My Label], SepalLength, SepalWidth, PetalLength, PetalWidth FROM ""{TestDatasets.irisDb.trainFilename}""";

            var loader = mlContext.Data.CreateDatabaseLoader<IrisDataWithLoadColumnName>();

            var databaseSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, commandText);

            var trainingData = loader.Load(databaseSource);

            IEstimator<ITransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainingData);

            var engine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);

            Assert.Equal(0, engine.Predict(new IrisData()
            {
                SepalLength = 4.5f,
                SepalWidth = 5.6f,
                PetalLength = 0.5f,
                PetalWidth = 0.5f,
            }).PredictedLabel);

            Assert.Equal(1, engine.Predict(new IrisData()
            {
                SepalLength = 4.9f,
                SepalWidth = 2.4f,
                PetalLength = 3.3f,
                PetalWidth = 1.0f,
            }).PredictedLabel);
        }

        [LightGBMFact]
        public void IrisVectorLightGbm()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // https://github.com/dotnet/machinelearning/issues/4156
                return;
            }

            var mlContext = new MLContext(seed: 1);

            var connectionString = GetConnectionString(TestDatasets.irisDb.name);
            var commandText = $@"SELECT * FROM ""{TestDatasets.irisDb.trainFilename}""";

            var loader = mlContext.Data.CreateDatabaseLoader<IrisVectorData>();

            var databaseSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, commandText);

            var trainingData = loader.Load(databaseSource);

            IEstimator<ITransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalInfo", "PetalInfo"))
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainingData);

            var engine = mlContext.Model.CreatePredictionEngine<IrisVectorData, IrisPrediction>(model);

            Assert.Equal(0, engine.Predict(new IrisVectorData()
            {
                SepalInfo = new float[] { 4.5f, 5.6f },
                PetalInfo = new float[] { 0.5f, 0.5f },
            }).PredictedLabel);

            Assert.Equal(1, engine.Predict(new IrisVectorData()
            {
                SepalInfo = new float[] { 4.9f, 2.4f },
                PetalInfo = new float[] { 3.3f, 1.0f },
            }).PredictedLabel);
        }

        [LightGBMFact]
        public void IrisVectorLightGbmWithLoadColumnName()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // https://github.com/dotnet/machinelearning/issues/4156
                return;
            }

            var mlContext = new MLContext(seed: 1);

            var connectionString = GetConnectionString(TestDatasets.irisDb.name);
            var commandText = $@"SELECT * FROM ""{TestDatasets.irisDb.trainFilename}""";

            var loader = mlContext.Data.CreateDatabaseLoader<IrisVectorDataWithLoadColumnName>();

            var databaseSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, commandText);

            var trainingData = loader.Load(databaseSource);

            IEstimator<ITransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalInfo", "PetalInfo"))
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainingData);

            var engine = mlContext.Model.CreatePredictionEngine<IrisVectorData, IrisPrediction>(model);

            Assert.Equal(0, engine.Predict(new IrisVectorData()
            {
                SepalInfo = new float[] { 4.5f, 5.6f },
                PetalInfo = new float[] { 0.5f, 0.5f },
            }).PredictedLabel);

            Assert.Equal(1, engine.Predict(new IrisVectorData()
            {
                SepalInfo = new float[] { 4.9f, 2.4f },
                PetalInfo = new float[] { 3.3f, 1.0f },
            }).PredictedLabel);
        }

        [Fact]
        public void IrisSdcaMaximumEntropy()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // https://github.com/dotnet/machinelearning/issues/4156
                return;
            }

            var mlContext = new MLContext(seed: 1);

            var connectionString = GetConnectionString(TestDatasets.irisDb.name);
            var commandText = $@"SELECT * FROM ""{TestDatasets.irisDb.trainFilename}""";

            var loader = mlContext.Data.CreateDatabaseLoader<IrisData>();

            var databaseSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, commandText);

            var trainingData = loader.Load(databaseSource);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainingData);

            var engine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);

            Assert.Equal(0, engine.Predict(new IrisData()
            {
                SepalLength = 4.5f,
                SepalWidth = 5.6f,
                PetalLength = 0.5f,
                PetalWidth = 0.5f,
            }).PredictedLabel);

            Assert.Equal(1, engine.Predict(new IrisData()
            {
                SepalLength = 4.9f,
                SepalWidth = 2.4f,
                PetalLength = 3.3f,
                PetalWidth = 1.0f,
            }).PredictedLabel);
        }

        private string GetTestDatabasePath(string databaseName)
        {
            return Path.GetFullPath(Path.Combine("TestDatabases", $"{databaseName}.mdf"));
        }

        private string GetConnectionString(string databaseName)
        {
            var databaseFile = GetTestDatabasePath(databaseName);
            return $@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename={databaseFile};Database={databaseName};Integrated Security=True;Connect Timeout=120";
        }

        public class IrisData
        {
            public int Label;

            public float SepalLength;

            public float SepalWidth;

            public float PetalLength;

            public float PetalWidth;
        }

        public class IrisDataWithLoadColumnName
        {
            [LoadColumnName("My Label")]
            [ColumnName("Label")]
            public int Kind;

            public float SepalLength;

            public float SepalWidth;

            public float PetalLength;

            public float PetalWidth;
        }

        public class IrisVectorData
        {
            public int Label;

            [LoadColumn(1, 2)]
            [VectorType(2)]
            public float[] SepalInfo;

            [LoadColumn(3, 4)]
            [VectorType(2)]
            public float[] PetalInfo;
        }

        public class IrisVectorDataWithLoadColumnName
        {
            public int Label;

            [LoadColumnName("SepalLength", "SepalWidth")]
            [VectorType(2)]
            public float[] SepalInfo;

            [LoadColumnName("PetalLength", "PetalWidth")]
            [VectorType(2)]
            public float[] PetalInfo;
        }

        public class IrisPrediction
        {
            public int PredictedLabel;

            public float[] Score;
        }
    }
}
