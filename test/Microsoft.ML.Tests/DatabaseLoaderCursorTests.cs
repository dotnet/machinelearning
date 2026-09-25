// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;
using System.Data;
using System.Data.SQLite;
using Microsoft.ML.TestFramework.Attributes;

namespace Microsoft.ML.Tests
{
    public class DatabaseLoaderCursorTests(ITestOutputHelper output) : BaseTestClass(output)
    {
        [X86X64Fact("The SQLite un-managed code, SQLite.interop, only supports x86/x64 architectures.")]
        public void DatabaseLoaderCursor_WhenCursorIsNotUsed_DisposeDoesNotThrow()
        {
            var mlContext = new MLContext(seed: 1);
            var loader = mlContext.Data.CreateDatabaseLoader([
                new DatabaseLoader.Column("Id", DbType.Int32, 1),
                new DatabaseLoader.Column("Value", DbType.Double, 2)
            ]);

            var connectionString = "DataSource=Dummy;Mode=Memory;Version=3;Timeout=120;Cache=Shared";
            using (var setup = new SQLiteConnection(connectionString))
            {
                setup.Open();
                using var command = setup.CreateCommand();
                command.CommandText = "BEGIN; DROP TABLE IF EXISTS CursorTestsTable; COMMIT;";
                command.ExecuteNonQuery();

                command.CommandText = "CREATE TABLE CursorTestsTable (Id INT, Value DOUBLE)";
                command.ExecuteNonQuery();

                command.CommandText = "INSERT INTO CursorTestsTable (Id, Value) VALUES (1, 1.1), (2, 2.2)";
                command.ExecuteNonQuery();
            }

            var connection = new SQLiteConnection(connectionString);
            var view = loader.Load(connection, "SELECT * FROM CursorTestsTable");

            // Create the cursor but don't use it
            // This validates that Dispose doesn't throw when the cursor hasn't been used; in this situation, the command and reader can be null, and the null propogation operator led to coverage complaints.
            var cursor = view.GetRowCursor(view.Schema);
            var exception = Record.Exception(() => cursor.Dispose());
            Assert.Null(exception);
        }
    }
}
