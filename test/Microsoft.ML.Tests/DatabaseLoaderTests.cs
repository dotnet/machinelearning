// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Data;
using System.Data.Common;
using System.Linq;
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
            var mlContext = new MLContext(seed: 1);
            var connectionString = GetDataPath(TestDatasets.iris.trainFilename);
            var commandText = "Label;SepalLength;SepalWidth;PetalLength;PetalWidth";

            var loaderColumns = new DatabaseLoader.Column[]
            {
                new DatabaseLoader.Column() { Name = "Label", Type = DbType.Int32 },
                new DatabaseLoader.Column() { Name = "SepalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "SepalWidth", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalWidth", Type = DbType.Single }
            };

            var loader = mlContext.Data.CreateDatabaseLoader(loaderColumns);

            var mockProviderFactory = new MockProviderFactory(mlContext, loaderColumns);
            var databaseSource = new DatabaseSource(mockProviderFactory, connectionString, commandText);

            var trainingData = loader.Load(databaseSource);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
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

        [Fact]
        public void IrisSdcaMaximumEntropy()
        {
            var mlContext = new MLContext(seed: 1);
            var connectionString = GetDataPath(TestDatasets.iris.trainFilename);
            var commandText = "Label;SepalLength;SepalWidth;PetalLength;PetalWidth";

            var loaderColumns = new DatabaseLoader.Column[]
            {
                new DatabaseLoader.Column() { Name = "Label", Type = DbType.Int32 },
                new DatabaseLoader.Column() { Name = "SepalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "SepalWidth", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalWidth", Type = DbType.Single }
            };

            var loader = mlContext.Data.CreateDatabaseLoader(loaderColumns);

            var mockProviderFactory = new MockProviderFactory(mlContext, loaderColumns);
            var databaseSource = new DatabaseSource(mockProviderFactory, connectionString, commandText);

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

    internal sealed class MockProviderFactory : DbProviderFactory
    {
        private MLContext _context;
        private DatabaseLoader.Column[] _columns;

        public MockProviderFactory(MLContext context, DatabaseLoader.Column[] columns)
        {
            _context = context;
            _columns = columns;
        }

        public override DbConnection CreateConnection() => new MockConnection(_context, _columns);
    }

    internal sealed class MockConnection : DbConnection
    {
        private string _dataPath;
        private TextLoader _reader;

        public MockConnection(MLContext context, DatabaseLoader.Column[] columns)
        {
            Columns = columns;

            var readerColumns = new TextLoader.Column[columns.Length];

            for (int i = 0; i < columns.Length; i++)
            {
                var column = columns[i];
                var columnType = column.Type.ToType();

                Assert.True(columnType.TryGetDataKind(out var internalDataKind));
                readerColumns[i] = new TextLoader.Column(column.Name, internalDataKind.ToDataKind(), i);
            }

            _reader = context.Data.CreateTextLoader(readerColumns);
        }

        public DatabaseLoader.Column[] Columns { get; }

        public override string ConnectionString
        {
            get
            {
                return _dataPath;
            }

            set
            {
                _dataPath = value;
            }
        }

        public override string Database => throw new NotImplementedException();

        public override string DataSource => throw new NotImplementedException();

        public IDataView DataView { get; private set; }

        public override string ServerVersion => throw new NotImplementedException();

        public override ConnectionState State => throw new NotImplementedException();

        public override void ChangeDatabase(string databaseName) => throw new NotImplementedException();

        public override void Close() => throw new NotImplementedException();

        public override void Open()
        {
            DataView = _reader.Load(_dataPath);
        }

        protected override DbTransaction BeginDbTransaction(IsolationLevel isolationLevel) => throw new NotImplementedException();

        protected override DbCommand CreateDbCommand() => new MockCommand(this);
    }

    internal sealed class MockCommand : DbCommand
    {
        public MockCommand(MockConnection connection)
        {
            CommandText = string.Empty;
            Connection = connection;
        }

        public override string CommandText { get; set; }

        public override int CommandTimeout
        {
            get => throw new NotImplementedException();
            set => throw new NotImplementedException();
        }

        public override CommandType CommandType
        {
            get
            {
                throw new NotImplementedException();
            }

            set
            {
                throw new NotImplementedException();
            }
        }

        public override bool DesignTimeVisible
        {
            get => throw new NotImplementedException();
            set => throw new NotImplementedException();
        }

        public override UpdateRowSource UpdatedRowSource
        {
            get => throw new NotImplementedException();
            set => throw new NotImplementedException();
        }

        protected override DbConnection DbConnection { get; set; }

        protected override DbParameterCollection DbParameterCollection => throw new NotImplementedException();

        protected override DbTransaction DbTransaction
        {
            get => throw new NotImplementedException();
            set => throw new NotImplementedException();
        }

        public override void Cancel() => throw new NotImplementedException();

        public override int ExecuteNonQuery() => throw new NotImplementedException();

        public override object ExecuteScalar() => throw new NotImplementedException();

        public override void Prepare() => throw new NotImplementedException();

        protected override DbParameter CreateDbParameter() => throw new NotImplementedException();

        protected override DbDataReader ExecuteDbDataReader(CommandBehavior behavior) => new MockDbDataReader(this);
    }

    internal sealed class MockDbDataReader : DbDataReader
    {
        private MockCommand _command;
        private DataViewRowCursor _rowCursor;
        private IDataView _dataView;

        public MockDbDataReader(MockCommand command)
        {
            _command = command;

            var connection = (MockConnection)_command.Connection;
            _dataView = connection.DataView;

            var inputColumns = _dataView.Schema.Where((column) => {
                var inputColumnNames = command.CommandText.Split(';');
                return inputColumnNames.Any((columnName) => column.Name.Equals(column.Name));
            });
            _rowCursor = _dataView.GetRowCursor(inputColumns);
        }

        public override object this[int ordinal] => throw new NotImplementedException();

        public override object this[string name] => throw new NotImplementedException();

        public override int Depth => throw new NotImplementedException();

        public override int FieldCount => throw new NotImplementedException();

        public override bool HasRows => throw new NotImplementedException();

        public override bool IsClosed => throw new NotImplementedException();

        public override int RecordsAffected => throw new NotImplementedException();

        public override bool GetBoolean(int ordinal) => throw new NotImplementedException();

        public override byte GetByte(int ordinal) => throw new NotImplementedException();

        public override long GetBytes(int ordinal, long dataOffset, byte[] buffer, int bufferOffset, int length) => throw new NotImplementedException();

        public override char GetChar(int ordinal) => throw new NotImplementedException();

        public override long GetChars(int ordinal, long dataOffset, char[] buffer, int bufferOffset, int length) => throw new NotImplementedException();

        public override string GetDataTypeName(int ordinal) => throw new NotImplementedException();

        public override DateTime GetDateTime(int ordinal) => throw new NotImplementedException();

        public override decimal GetDecimal(int ordinal) => throw new NotImplementedException();

        public override double GetDouble(int ordinal) => throw new NotImplementedException();

        public override IEnumerator GetEnumerator() => throw new NotImplementedException();

        public override Type GetFieldType(int ordinal) => throw new NotImplementedException();

        public override float GetFloat(int ordinal)
        {
            float result = 0;
            _rowCursor.GetGetter<float>(_dataView.Schema[ordinal])(ref result);
            return result;
        }

        public override Guid GetGuid(int ordinal) => throw new NotImplementedException();

        public override short GetInt16(int ordinal) => throw new NotImplementedException();

        public override int GetInt32(int ordinal)
        {
            int result = 0;
            _rowCursor.GetGetter<int>(_dataView.Schema[ordinal])(ref result);
            return result;
        }

        public override long GetInt64(int ordinal) => throw new NotImplementedException();

        public override string GetName(int ordinal) => throw new NotImplementedException();

        public override int GetOrdinal(string name)
        {
            var connection = (MockConnection)_command.Connection;
            var columns = connection.Columns;

            for (int i = 0; i < columns.Length; i++)
            {
                var column = columns[i];

                if (column.Name.Equals(name))
                {
                    return i;
                }
            }

            return -1;
        }

        public override string GetString(int ordinal) => throw new NotImplementedException();

        public override object GetValue(int ordinal) => throw new NotImplementedException();

        public override int GetValues(object[] values) => throw new NotImplementedException();

        public override bool IsDBNull(int ordinal) => false;

        public override bool NextResult() => throw new NotImplementedException();

        public override bool Read() => _rowCursor.MoveNext();
    }
}
