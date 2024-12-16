// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Apache.Arrow;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.Data.Analysis.Tests
{
    public class ArrowStringColumnTests : BaseTestClass
    {
        public ArrowStringColumnTests(ITestOutputHelper output) : base(output, true)
        {
        }

        [Fact]
        public void TestBasicArrowStringColumn()
        {
            using StringArray strArray = new StringArray.Builder().Append("foo").Append("bar").Build();
            Memory<byte> dataMemory = new byte[] { 102, 111, 111, 98, 97, 114 };
            Memory<byte> nullMemory = new byte[] { 0, 0, 0, 0 };
            Memory<byte> offsetMemory = new byte[] { 0, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0 };

            ArrowStringDataFrameColumn stringColumn = new ArrowStringDataFrameColumn("String", dataMemory, offsetMemory, nullMemory, strArray.Length, strArray.NullCount);
            Assert.Equal(2, stringColumn.Length);
            Assert.Equal("foo", stringColumn[0]);
            Assert.Equal("bar", stringColumn[1]);
        }

        [Fact]
        public void TestArrowStringColumnWithNulls()
        {
            string data = "joemark";
            byte[] bytes = Encoding.UTF8.GetBytes(data);
            Memory<byte> dataMemory = new Memory<byte>(bytes);
            Memory<byte> nullMemory = new byte[] { 0b1101 };
            Memory<byte> offsetMemory = new byte[] { 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0 };
            ArrowStringDataFrameColumn stringColumn = new ArrowStringDataFrameColumn("String", dataMemory, offsetMemory, nullMemory, 4, 1);

            Assert.Equal(4, stringColumn.Length);
            Assert.Equal("joe", stringColumn[0]);
            Assert.Null(stringColumn[1]);
            Assert.Equal("mark", stringColumn[2]);
            Assert.Equal("", stringColumn[3]);

            List<string> ret = stringColumn[0, 4];
            Assert.Equal("joe", ret[0]);
            Assert.Null(ret[1]);
            Assert.Equal("mark", ret[2]);
            Assert.Equal("", ret[3]);
        }

        [Fact]
        public void TestArrowStringColumnClone()
        {
            using StringArray strArray = new StringArray.Builder().Append("foo").Append("bar").Build();
            Memory<byte> dataMemory = new byte[] { 102, 111, 111, 98, 97, 114 };
            Memory<byte> nullMemory = new byte[] { 0, 0, 0, 0 };
            Memory<byte> offsetMemory = new byte[] { 0, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0 };

            ArrowStringDataFrameColumn stringColumn = new ArrowStringDataFrameColumn("String", dataMemory, offsetMemory, nullMemory, strArray.Length, strArray.NullCount);

            DataFrameColumn clone = stringColumn.Clone(numberOfNullsToAppend: 5);
            Assert.Equal(7, clone.Length);
            Assert.Equal(stringColumn[0], clone[0]);
            Assert.Equal(stringColumn[1], clone[1]);
            for (int i = 2; i < 7; i++)
                Assert.Null(clone[i]);
        }

        [Fact]
        public void TestArrowStringApply()
        {
            ArrowStringDataFrameColumn column = DataFrameTests.CreateArrowStringColumn(10);
            ArrowStringDataFrameColumn ret = column.Apply((string cur) =>
            {
                if (cur != null)
                {
                    return cur + "123";
                }
                return null;
            });
            for (long i = 0; i < column.Length; i++)
            {
                if (column[i] != null)
                {
                    Assert.Equal(column[i] + "123", ret[i]);
                }
                else
                {
                    Assert.Null(ret[i]);
                }
            }
            Assert.Equal(1, ret.NullCount);

            // Test null counts
            ret = column.Apply((string cur) =>
            {
                return null;
            });
            Assert.Equal(column.Length, ret.NullCount);
        }
    }
}
