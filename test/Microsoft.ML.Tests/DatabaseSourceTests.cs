// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data.Common;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Moq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class DatabaseSourceTests(ITestOutputHelper output) : BaseTestClass(output)
    {
        [Fact]
        public void DatabaseSource_WhenCreatedUsingDbConnection_ConnectionIsNotNull()
        {
            var mockConnection = Mock.Of<DbConnection>();
            DatabaseSource source = new(mockConnection, "");

            Assert.NotNull(source.Connection);
        }

        [Fact]
        public void DatabaseSource_WhenCreatedUsingDbConnection_ProviderFactoryIsNull()
        {
            var mockConnection = Mock.Of<DbConnection>();
            DatabaseSource source = new(mockConnection, "");

            Assert.Null(source.ProviderFactory);
        }

        [Fact]
        public void DatabaseSource_WhenCreatedUsingDbConnection_ConnectionStringIsNull()
        {
            var mockConnection = Mock.Of<DbConnection>();
            DatabaseSource source = new(mockConnection, "");

            Assert.Null(source.ConnectionString);
        }

        [Fact]
        public void DatabaseSource_WhenCreatedUsingDbProviderFactory_ConnectionIsNull()
        {
            var mockFactory = Mock.Of<DbProviderFactory>();
            DatabaseSource source = new(mockFactory, "", "");

            Assert.Null(source.Connection);
        }

        [Fact]
        public void DatabaseSource_WhenCreatedUsingDbProviderFactory_ProviderFactoryIsNotNull()
        {
            var mockFactory = Mock.Of<DbProviderFactory>();
            DatabaseSource source = new(mockFactory, "", "");

            Assert.NotNull(source.ProviderFactory);
        }

        [Fact]
        public void DatabaseSource_WhenCreatedUsingDbProviderFactory_ConnectionStringIsNotNull()
        {
            var mockFactory = Mock.Of<DbProviderFactory>();
            DatabaseSource source = new(mockFactory, "", "");

            Assert.NotNull(source.ConnectionString);
        }
    }
}
