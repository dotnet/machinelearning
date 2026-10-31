// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Xunit;
using Xunit.Abstractions;


namespace Microsoft.Data.Analysis.Tests
{
    public class VBufferColumnTests : BaseTestClass
    {
        public VBufferColumnTests(ITestOutputHelper output) : base(output, true)
        {
        }

        [Fact]
        public void TestVBufferColumn_Creation()
        {
            var buffers = Enumerable.Repeat(new VBuffer<int>(5, new[] { 0, 1, 2, 3, 4 }), 10).ToArray();
            var vBufferColumn = new VBufferDataFrameColumn<int>("VBuffer", buffers);

            Assert.Equal(10, vBufferColumn.Length);
            Assert.Equal(5, vBufferColumn[0].GetValues().Length);
            Assert.Equal(0, vBufferColumn[0].GetValues()[0]);
        }

        [Fact]
        public void TestVBufferColumn_Indexer()
        {
            var buffer = new VBuffer<int>(5, new[] { 4, 3, 2, 1, 0 });

            var vBufferColumn = new VBufferDataFrameColumn<int>("VBuffer", 1);
            vBufferColumn[0] = buffer;

            Assert.Equal(1, vBufferColumn.Length);
            Assert.Equal(5, vBufferColumn[0].GetValues().Length);
            Assert.Equal(0, vBufferColumn[0].GetValues()[4]);
        }

        [Fact]
        public void TestVBufferColumn_CloneAppendsDefaultValues()
        {
            var first = new VBuffer<int>(2, new[] { 1, 2 });
            var second = new VBuffer<int>(2, new[] { 3, 4 });
            var column = new VBufferDataFrameColumn<int>("VBuffer", new[] { first, second });
            var mapIndices = new Int32DataFrameColumn("Indices", new[] { 1 });

            VBufferDataFrameColumn<int> clone = column.Clone(numberOfNullsToAppend: 2);
            VBufferDataFrameColumn<int> mappedClone = column.Clone(mapIndices, invertMapIndices: false, numberOfNullsToAppend: 2);

            Assert.Equal(4, clone.Length);
            Assert.Equal(first.GetValues().ToArray(), clone[0].GetValues().ToArray());
            Assert.Equal(second.GetValues().ToArray(), clone[1].GetValues().ToArray());
            Assert.Equal(0, clone[2].GetValues().Length);
            Assert.Equal(0, clone[3].GetValues().Length);

            Assert.Equal(3, mappedClone.Length);
            Assert.Equal(second.GetValues().ToArray(), mappedClone[0].GetValues().ToArray());
            Assert.Equal(0, mappedClone[1].GetValues().Length);
            Assert.Equal(0, mappedClone[2].GetValues().Length);
        }

        [X64Fact("32-bit doesn't allow to allocate more than 2 Gb")]
        public void TestVBufferColumn_Indexer_MoreThanMaxInt()
        {
            var originalValues = new[] { 4, 3, 2, 1, 0 };

            var length = VBufferDataFrameColumn<int>.MaxCapacity + 3;

            var vBufferColumn = new VBufferDataFrameColumn<int>("VBuffer", length);
            long index = length - 2;

            vBufferColumn[index] = new VBuffer<int>(5, originalValues);

            var values = vBufferColumn[index].GetValues();

            Assert.Equal(length, vBufferColumn.Length);
            Assert.Equal(5, values.Length);

            for (int i = 0; i < values.Length; i++)
            {
                Assert.Equal(originalValues[i], values[i]);
            }

            vBufferColumn = null;
        }
    }
}
