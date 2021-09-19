// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Data;
using System.Data.SqlClient;
using System.Data.SQLite;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon;
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
            DatabaseSource dbs = GetIrisDatabaseSource("SELECT * FROM {0}");
            IrisLightGbmImpl(dbs);
        }

        [LightGBMFact]
        public void IrisLightGbmWithTimeout()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) //sqlite does not have built-in command for sleep
                return;
            DatabaseSource dbs = GetIrisDatabaseSource("WAITFOR DELAY '00:00:01'; SELECT * FROM {0}", 1);
            var ex = Assert.Throws<System.Reflection.TargetInvocationException>(() => IrisLightGbmImpl(dbs));
            Assert.Contains("Timeout", ex.InnerException.Message);
        }

        private void IrisLightGbmImpl(DatabaseSource dbs)
        {
            var mlContext = new MLContext(seed: 1);

            var loaderColumns = new DatabaseLoader.Column[]
            {
                new DatabaseLoader.Column() { Name = "Label", Type = DbType.Int32 },
                new DatabaseLoader.Column() { Name = "SepalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "SepalWidth", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalWidth", Type = DbType.Single }
            };

            var loader = mlContext.Data.CreateDatabaseLoader(loaderColumns);

            var trainingData = loader.Load(dbs);

            IEstimator<ITransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
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
            var mlContext = new MLContext(seed: 1);

            var loader = mlContext.Data.CreateDatabaseLoader<IrisDataWithLoadColumnName>();

            var trainingData = loader.Load(GetIrisDatabaseSource("SELECT Label as [My Label], SepalLength, SepalWidth, PetalLength, PetalWidth FROM {0}"));

            IEstimator<ITransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
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
            var mlContext = new MLContext(seed: 1);

            var loader = mlContext.Data.CreateDatabaseLoader<IrisVectorData>();

            var trainingData = loader.Load(GetIrisDatabaseSource("SELECT * FROM {0}"));

            IEstimator<ITransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalInfo", "PetalInfo"))
                .AppendCacheCheckpoint(mlContext)
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
            var mlContext = new MLContext(seed: 1);

            var loader = mlContext.Data.CreateDatabaseLoader<IrisVectorDataWithLoadColumnName>();

            var trainingData = loader.Load(GetIrisDatabaseSource("SELECT * FROM {0}"));

            IEstimator<ITransformer> pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalInfo", "PetalInfo"))
                .AppendCacheCheckpoint(mlContext)
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

        [X86X64FactAttribute("The SQLite un-managed code, SQLite.interop, only supports x86/x64 architectures.")]
        public void IrisSdcaMaximumEntropy()
        {
            var mlContext = new MLContext(seed: 1);

            var loader = mlContext.Data.CreateDatabaseLoader<IrisData>();

            var trainingData = loader.Load(GetIrisDatabaseSource("SELECT * FROM {0}"));

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
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

        /// <summary>
        /// Non-Windows builds do not support SqlClientFactory/MSSQL databases. Hence, an equivalent
        /// SQLite database is used on Linux and MacOS builds.
        /// </summary>
        /// <returns>Return the appropiate Iris DatabaseSource according to build OS.</returns>
        private DatabaseSource GetIrisDatabaseSource(string command, int commandTimeoutInSeconds = 30)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return new DatabaseSource(
                    SqlClientFactory.Instance,
                    GetMSSQLConnectionString(TestDatasets.irisDb.name),
                    String.Format(command, $@"""{TestDatasets.irisDb.trainFilename}"""),
                    commandTimeoutInSeconds);
            else
                return new DatabaseSource(
                    SQLiteFactory.Instance,
                    GetSQLiteConnectionString(TestDatasets.irisDbSQLite.name),
                    String.Format(command, TestDatasets.irisDbSQLite.trainFilename),
                    commandTimeoutInSeconds);
        }

        private string GetMSSQLConnectionString(string databaseName)
        {
            var databaseFile = Path.GetFullPath(Path.Combine("TestDatabases", $"{databaseName}.mdf"));
            return $@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename={databaseFile};Database={databaseName};Integrated Security=True;Connect Timeout=120";
        }

        private string GetSQLiteConnectionString(string databaseName)
        {
            var databaseFile = Path.GetFullPath(Path.Combine("TestDatabases", $"{databaseName}.sqlite"));
            return $@"Data Source={databaseFile};Version=3;Read Only=True;Timeout=120;";
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
            [LoadColumn(0)]
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
            [LoadColumnName("Label")]
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
