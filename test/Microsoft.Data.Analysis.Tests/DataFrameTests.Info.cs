// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameTests
    {
        [Theory]
        [InlineData(0, 0)]
        [InlineData(3, 0)]
        [InlineData(3, 1)]
        [InlineData(3, 3)]
        public void TestColumnInfo(int length, int nullCount)
        {
            var intColumn = new Int32DataFrameColumn("Int", Enumerable.Range(0, length).Select(i => i < nullCount ? (int?)null : i));
            var stringColumn = new StringDataFrameColumn("String", Enumerable.Range(0, length).Select(i => i < nullCount ? null : i.ToString()));

            foreach (DataFrameColumn column in new DataFrameColumn[] { intColumn, stringColumn })
            {
                AssertColumnInfo(column);
            }
        }

        [Fact]
        public void TestArrowStringColumnInfo()
        {
            AssertColumnInfo(new ArrowStringDataFrameColumn("Empty"));

            ReadOnlyMemory<byte> values = new byte[] { 102, 111, 111 };
            ReadOnlyMemory<byte> offsets = new byte[] { 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0 };
            ReadOnlyMemory<byte> nullBits = new byte[] { 1 };
            var column = new ArrowStringDataFrameColumn("ArrowString", values, offsets, nullBits, 2, 1);

            AssertColumnInfo(column);
        }

        private static void AssertColumnInfo(DataFrameColumn column)
        {
            StringDataFrameColumn info = column.Info();

            Assert.Equal(column.Name, info.Name);
            Assert.Equal(2, info.Length);
            Assert.Equal($"{Strings.DataType}: {column.DataType}", info[0]);
            Assert.Equal($"{Strings.DescriptionMethodLength}: {column.Length - column.NullCount}", info[1]);

            DataFrame dataFrameInfo = new DataFrame(column).Info();

            Assert.Equal(2, dataFrameInfo.Columns.Count);
            Assert.Equal(2, dataFrameInfo.Rows.Count);
            Assert.Equal("Info", dataFrameInfo.Columns[0].Name);
            Assert.Equal(column.Name, dataFrameInfo.Columns[1].Name);
            Assert.Equal(Strings.DataType, dataFrameInfo[0, 0]);
            Assert.Equal(Strings.DescriptionMethodLength, dataFrameInfo[1, 0]);
            Assert.Equal(column.DataType.ToString(), dataFrameInfo[0, 1]);
            Assert.Equal((column.Length - column.NullCount).ToString(), dataFrameInfo[1, 1]);
        }

        [Fact]
        public void TestEmptyDataFrameInfo()
        {
            DataFrame info = new DataFrame().Info();

            Assert.Empty(info.Columns);
            Assert.Equal(0, info.Rows.Count);
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void TestColumnInfoOverride(bool useBaseInfo)
        {
            var column = new CustomInfoStringDataFrameColumn("Custom", useBaseInfo);
            StringDataFrameColumn originalInfo = column.Info();
            string[] originalValues = originalInfo.ToArray();

            DataFrame info = new DataFrame(column).Info();

            Assert.Equal("Custom", info.Columns[1].Name);
            Assert.Equal(useBaseInfo ? "System.String: custom" : "custom: type", info[0, 1]);
            Assert.Equal(useBaseInfo ? "1: custom" : null, info[1, 1]);
            Assert.Equal(originalValues, originalInfo.ToArray());
            Assert.Same(originalInfo, column.Info());
        }

        private sealed class CustomInfoStringDataFrameColumn : StringDataFrameColumn
        {
            private readonly bool _useBaseInfo;
            private StringDataFrameColumn _info;

            public CustomInfoStringDataFrameColumn(string name, bool useBaseInfo) : base(name, new string[] { "value" })
            {
                _useBaseInfo = useBaseInfo;
            }

            public override StringDataFrameColumn Info()
            {
                if (_info == null)
                {
                    if (_useBaseInfo)
                    {
                        _info = base.Info();
                        _info[0] += ": custom";
                        _info[1] += ": custom";
                    }
                    else
                    {
                        _info = new StringDataFrameColumn(Name, new string[] { "custom: type", null });
                    }
                }
                return _info;
            }
        }
    }
}
