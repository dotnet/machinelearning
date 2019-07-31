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

            var loaderColumns = new DatabaseLoader.Column[]
            {
                new DatabaseLoader.Column() { Name = "Label", Type = DbType.Int32 },
                new DatabaseLoader.Column() { Name = "SepalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "SepalWidth", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalLength", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "PetalWidth", Type = DbType.Single }
            };

            using (var connection = new MockConnection(mlContext, GetDataPath(TestDatasets.iris.trainFilename), loaderColumns))
            {
                connection.Open();

                using (var command = new MockCommand(connection, "Label", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                {
                    var loader = mlContext.Data.CreateDatabaseLoader(loaderColumns);

                    var trainingData = loader.Load(() => command.ExecuteReader());
                    //trainingData = mlContext.Data.Cache(trainingData, "Label", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth");

                    var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                        .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                        //.AppendCacheCheckpoint(mlContext)
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

    internal sealed class MockConnection : DbConnection
    {
        private string _dataPath;
        private DatabaseLoader.Column[] _columns;
        private TextLoader _reader;
        private IDataView _dataView;

        public MockConnection(MLContext mlContext, string dataPath, DatabaseLoader.Column[] columns)
        {
            _dataPath = dataPath;
            _columns = columns;

            var readerColumns = new TextLoader.Column[columns.Length];

            for (int i = 0; i < columns.Length; i++)
            {
                var column = columns[i];
                var columnType = column.Type.ToType();

                Assert.True(columnType.TryGetDataKind(out var internalDataKind));
                readerColumns[i] = new TextLoader.Column(column.Name, internalDataKind.ToDataKind(), i);
            }

            _reader = mlContext.Data.CreateTextLoader(readerColumns);
        }

        public DatabaseLoader.Column[] Columns => _columns;

        public override string ConnectionString
        {
            get
            {
                return _dataPath;
            }

            set
            {
                throw new NotImplementedException();
            }
        }

        public override string Database
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override string DataSource
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public IDataView DataView => _dataView;

        public override string ServerVersion
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override ConnectionState State
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override void ChangeDatabase(string databaseName)
        {
            throw new NotImplementedException();
        }

        public override void Close()
        {
            throw new NotImplementedException();
        }

        public override void Open()
        {
            _dataView = _reader.Load(_dataPath);
        }

        protected override DbTransaction BeginDbTransaction(IsolationLevel isolationLevel)
        {
            throw new NotImplementedException();
        }

        protected override DbCommand CreateDbCommand()
        {
            throw new NotImplementedException();
        }
    }

    internal sealed class MockCommand : DbCommand
    {
        private MockConnection _connection;
        private string[] _inputColumnNames;

        public MockCommand(MockConnection connection, params string[] inputColumnNames)
        {
            _connection = connection;
            _inputColumnNames = inputColumnNames;
        }

        public override string CommandText
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

        public override int CommandTimeout
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

        public string[] InputColumnNames => _inputColumnNames;

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
            get
            {
                throw new NotImplementedException();
            }

            set
            {
                throw new NotImplementedException();
            }
        }

        public override UpdateRowSource UpdatedRowSource
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

        protected override DbConnection DbConnection
        {
            get
            {
                return _connection;
            }

            set
            {
                throw new NotImplementedException();
            }
        }

        protected override DbParameterCollection DbParameterCollection
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        protected override DbTransaction DbTransaction
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

        public override void Cancel()
        {
            throw new NotImplementedException();
        }

        public override int ExecuteNonQuery()
        {
            throw new NotImplementedException();
        }

        public override object ExecuteScalar()
        {
            throw new NotImplementedException();
        }

        public override void Prepare()
        {
            throw new NotImplementedException();
        }

        protected override DbParameter CreateDbParameter()
        {
            throw new NotImplementedException();
        }

        protected override DbDataReader ExecuteDbDataReader(CommandBehavior behavior)
        {
            return new MockDbDataReader(this);
        }
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

            var inputColumns = _dataView.Schema.Where((column) => 
                command.InputColumnNames.Any((columnName) => column.Name.Equals(column.Name))
            );
            _rowCursor = _dataView.GetRowCursor(inputColumns);
        }

        public override object this[int ordinal]
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override object this[string name]
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override int Depth
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override int FieldCount
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override bool HasRows
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override bool IsClosed
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override int RecordsAffected
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public override bool GetBoolean(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override byte GetByte(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override long GetBytes(int ordinal, long dataOffset, byte[] buffer, int bufferOffset, int length)
        {
            throw new NotImplementedException();
        }

        public override char GetChar(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override long GetChars(int ordinal, long dataOffset, char[] buffer, int bufferOffset, int length)
        {
            throw new NotImplementedException();
        }

        public override string GetDataTypeName(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override DateTime GetDateTime(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override decimal GetDecimal(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override double GetDouble(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override IEnumerator GetEnumerator()
        {
            throw new NotImplementedException();
        }

        public override Type GetFieldType(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override float GetFloat(int ordinal)
        {
            float result = 0;
            _rowCursor.GetGetter<float>(_dataView.Schema[ordinal])(ref result);
            return result;
        }

        public override Guid GetGuid(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override short GetInt16(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override int GetInt32(int ordinal)
        {
            int result = 0;
            _rowCursor.GetGetter<int>(_dataView.Schema[ordinal])(ref result);
            return result;
        }

        public override long GetInt64(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override string GetName(int ordinal)
        {
            throw new NotImplementedException();
        }

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

        public override string GetString(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override object GetValue(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override int GetValues(object[] values)
        {
            throw new NotImplementedException();
        }

        public override bool IsDBNull(int ordinal)
        {
            throw new NotImplementedException();
        }

        public override bool NextResult()
        {
            throw new NotImplementedException();
        }

        public override bool Read()
        {
            return _rowCursor.MoveNext();
        }
    }
}
