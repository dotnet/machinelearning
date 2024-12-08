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
        public VBufferColumnTests(ITestOutputHelper output) : base(output)
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
