// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.Data.Analysis.Tests
{
    public class StringDataFrameColumnTests : BaseTestClass
    {
        public StringDataFrameColumnTests(ITestOutputHelper output) : base(output, true)
        {
        }

        [Fact]
        public void TestColumnClone()
        {
            var stringColumn = new StringDataFrameColumn("Test", new[] { "Zero", "One", "Two", null, "Four", "Five" });
            var clonedColumn = stringColumn.Clone();

            Assert.NotSame(stringColumn, clonedColumn);
            Assert.Equal(stringColumn.Name, clonedColumn.Name);
            Assert.Equal(stringColumn.Length, clonedColumn.Length);
            Assert.Equal(stringColumn.NullCount, clonedColumn.NullCount);

            for (int i = 0; i < stringColumn.Length; i++)
                Assert.Equal(stringColumn[i], clonedColumn[i]);
        }

        [Fact]
        public void TestColumnClone_WithIntMapIndices()
        {
            var mapIndices = new[] { 0, 1, 2, 2, 3, 4, 5 };
            var stringColumn = new StringDataFrameColumn("Test", ["Zero", "One", null, "Three", "Four", "Five"]);
            var clonedColumn = stringColumn.Clone(new Int32DataFrameColumn("Map Indices", mapIndices));

            Assert.NotSame(stringColumn, clonedColumn);
            Assert.Equal(stringColumn.Name, clonedColumn.Name);
            Assert.Equal(mapIndices.Length, clonedColumn.Length);
            Assert.Equal(2, clonedColumn.NullCount);

            for (int i = 0; i < mapIndices.Length; i++)
                Assert.Equal(stringColumn[mapIndices[i]], clonedColumn[i]);
        }

        [Fact]
        public void TestColumnClone_WithIntMapIndices_InvertIndices()
        {
            var mapIndices = new[] { 0, 1, 2, 2, 3, 4, 5 };
            var stringColumn = new StringDataFrameColumn("Test", ["Zero", "One", null, "Three", "Four", "Five"]);
            var clonedColumn = stringColumn.Clone(new Int32DataFrameColumn("Map Indices", mapIndices), true);

            Assert.NotSame(stringColumn, clonedColumn);
            Assert.Equal(stringColumn.Name, clonedColumn.Name);
            Assert.Equal(mapIndices.Length, clonedColumn.Length);
            Assert.Equal(2, clonedColumn.NullCount);

            for (int i = 0; i < mapIndices.Length; i++)
                Assert.Equal(stringColumn[mapIndices[mapIndices.Length - 1 - i]], clonedColumn[i]);
        }

        [Fact]
        public void TestColumnClone_WithLongMapIndices()
        {
            var mapIndices = new long[] { 0, 1, 2, 2, 3, 4, 5 };
            var stringColumn = new StringDataFrameColumn("Test", ["Zero", "One", null, "Three", "Four", "Five"]);
            var clonedColumn = stringColumn.Clone(new Int64DataFrameColumn("Map Indices", mapIndices));

            Assert.NotSame(stringColumn, clonedColumn);
            Assert.Equal(stringColumn.Name, clonedColumn.Name);
            Assert.Equal(mapIndices.Length, clonedColumn.Length);
            Assert.Equal(2, clonedColumn.NullCount);

            for (int i = 0; i < mapIndices.Length; i++)
                Assert.Equal(stringColumn[mapIndices[i]], clonedColumn[i]);
        }

        [Fact]
        public void TestColumnClone_WithLongMapIndices_InvertIndices()
        {
            var mapIndices = new long[] { 0, 1, 2, 2, 3, 4, 5 };
            var stringColumn = new StringDataFrameColumn("Test", ["Zero", "One", "Two", null, "Four", "Five"]);
            var clonedColumn = stringColumn.Clone(new Int64DataFrameColumn("Map Indices", mapIndices), true);
            Assert.Equal(1, clonedColumn.NullCount);

            Assert.NotSame(stringColumn, clonedColumn);
            Assert.Equal(stringColumn.Name, clonedColumn.Name);
            Assert.Equal(mapIndices.Length, clonedColumn.Length);

            for (int i = 0; i < mapIndices.Length; i++)
                Assert.Equal(stringColumn[mapIndices[mapIndices.Length - 1 - i]], clonedColumn[i]);
        }
    }
}
