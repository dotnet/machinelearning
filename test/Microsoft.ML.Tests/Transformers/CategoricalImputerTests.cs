// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;
using Microsoft.ML.RunTests;
using System;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;
using System.Linq;
using Microsoft.ML.TestFramework.Attributes;

namespace Microsoft.ML.Tests.Transformers
{
    public class CategoricalImputerTests : TestDataPipeBase
    {
        public CategoricalImputerTests(ITestOutputHelper output) : base(output)
        {
        }

        private class SchemaAllTypes
        {
            public byte uint8_t;
            public sbyte int8_t;
            public short int16_t;
            public ushort uint16_t;
            public int int32_t;
            public uint uint32_t;
            public long int64_t;
            public ulong uint64_t;
            public float float_t;
            public double double_t;
            public string str;

            internal SchemaAllTypes(byte num_i, float num_f, string s)
            {
                uint8_t = num_i;
                int8_t = (sbyte)num_i;
                int16_t = num_i;
                uint16_t = num_i;
                int32_t = num_i;
                uint32_t = num_i;
                int64_t = num_i;
                uint64_t = num_i;
                float_t = num_f;
                double_t = num_f;
                str = s;
            }
        }

        private IDataView GetIDataView()
        {
            List<SchemaAllTypes> dataList = new List<SchemaAllTypes>();
            dataList.Add(new SchemaAllTypes(1, 1.5f, "one"));
            dataList.Add(new SchemaAllTypes(1, 1.5f, "one"));
            dataList.Add(new SchemaAllTypes(2, 2.5f, "two"));
            dataList.Add(new SchemaAllTypes(0, Single.NaN, null));
            dataList.Add(new SchemaAllTypes(1, 1.5f, "one"));
            dataList.Add(new SchemaAllTypes(1, 1.5f, "one"));
            dataList.Add(new SchemaAllTypes(2, 2.5f, "two"));
            dataList.Add(new SchemaAllTypes(0, Single.NaN, null));
            dataList.Add(new SchemaAllTypes(1, 1.5f, "one"));
            dataList.Add(new SchemaAllTypes(1, 1.5f, "one"));
            dataList.Add(new SchemaAllTypes(2, 2.5f, "two"));
            dataList.Add(new SchemaAllTypes(0, Single.NaN, null));

            MLContext mlContext = new MLContext(1);
            IDataView data = mlContext.Data.LoadFromEnumerable(dataList);
            return data;
        }
        private void Test<T>(string columnName, bool addNewCol, T mostFrequentValue)
        {

            string outputColName = addNewCol ? columnName + "_output" : columnName;
            string inputColName = addNewCol ? columnName : null;

            MLContext mlContext = new MLContext(1);
            var data = GetIDataView();
            var pipeline = mlContext.Transforms.ImputeCategories(outputColName, inputColName);
            TestEstimatorCore(pipeline, data);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);

            List<T> transformedColData = output.GetColumn<T>(outputColName).ToList();
            Assert.Equal(mostFrequentValue, transformedColData[3]);
            Assert.Equal(mostFrequentValue, transformedColData[7]);
            Assert.Equal(mostFrequentValue, transformedColData[11]);
        }

        [NotCentOS7Fact]
        public void TestAllTypes()
        {
            Test("float_t", false, 1.5f);
            Test("float_t", true, 1.5f);
            Test("double_t", false, 1.5);
            Test("double_t", true, 1.5);
            Test("str", false, "one");
            Test("str", true, "one");

            Done();
        }

    }
}
