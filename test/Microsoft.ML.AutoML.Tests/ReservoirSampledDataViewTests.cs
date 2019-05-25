// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class ReservoirSampledDataViewTests
    {
        /// <summary>
        /// Test case where requested sample size is greater than # of rows in original data view.
        /// </summary>
        [TestMethod]
        public void SmallDataViewTest()
        {
            var dataView = BuildTestDataView(5);
            var preview = dataView.Preview();
            var sampledDataView = new ReservoirSampledDataView(dataView, 10);

            var cursor = sampledDataView.GetRowCursor(sampledDataView.Schema);
            var numericColGetter = cursor.GetGetter<float>(sampledDataView.Schema[0]);
            var numericVectorColGetter = cursor.GetGetter<VBuffer<float>>(sampledDataView.Schema[1]);
            var stringColGetter = cursor.GetGetter<ReadOnlyMemory<char>>(sampledDataView.Schema[2]);

            var rowIdx = 0;
            while (cursor.MoveNext())
            {
                float numericVal = 0;
                numericColGetter.Invoke(ref numericVal);
                var numericVectorVal = default(VBuffer<float>);
                numericVectorColGetter.Invoke(ref numericVectorVal);
                var stringVal = default(ReadOnlyMemory<char>);
                stringColGetter.Invoke(ref stringVal);

                // Assert row integrity
                Assert.AreEqual(rowIdx, numericVal);
                Assert.AreEqual(rowIdx.ToString(), stringVal.ToString());
                Assert.AreEqual(rowIdx, numericVectorVal.GetValues()[0]);
                Assert.AreEqual(rowIdx, -1 * numericVectorVal.GetValues()[1]);

                rowIdx++;
            }

            // Assert number of rows is size of original data view
            Assert.AreEqual(5, rowIdx);
        }

        [TestMethod]
        public void Test()
        {
            var dataView = BuildTestDataView(1000);
            var preview = dataView.Preview();
            var sampledDataView = new ReservoirSampledDataView(dataView, 100);

            var cursor = sampledDataView.GetRowCursor(sampledDataView.Schema);
            var numericColGetter = cursor.GetGetter<float>(sampledDataView.Schema[0]);
            var numericVectorColGetter = cursor.GetGetter<VBuffer<float>>(sampledDataView.Schema[1]);
            var stringColGetter = cursor.GetGetter<ReadOnlyMemory<char>>(sampledDataView.Schema[2]);

            var rowIdx = 0;
            var seenNumericValues = new HashSet<float>();
            while (cursor.MoveNext())
            {
                float numericVal = 0;
                numericColGetter.Invoke(ref numericVal);
                var numericVectorVal = default(VBuffer<float>);
                numericVectorColGetter.Invoke(ref numericVectorVal);
                var stringVal = default(ReadOnlyMemory<char>);
                stringColGetter.Invoke(ref stringVal);

                // Assert row integrity
                Assert.AreEqual(numericVal.ToString(), stringVal.ToString());
                Assert.AreEqual(numericVal, numericVectorVal.GetValues()[0]);
                Assert.AreEqual(numericVal, -1 * numericVectorVal.GetValues()[1]);

                seenNumericValues.Add(numericVal);

                rowIdx++;
            }

            // Assert number of rows = of original data view
            Assert.AreEqual(100, rowIdx);
            // Assert all sampled numeric values are distinct
            Assert.AreEqual(100, seenNumericValues.Count);
            // Assert sampled numeric values has at least one # greater than 100
            Assert.IsTrue(seenNumericValues.Any(x => x > 100));
        }

        private static IDataView BuildTestDataView(int numRows)
        {
            var mlContext = new MLContext();
            var dataViewBuilder = new ArrayDataViewBuilder(mlContext);
            AddNumericColumn(dataViewBuilder, numRows);
            AddNumericVectorColumn(dataViewBuilder, numRows);
            AddStringColumn(dataViewBuilder, numRows);
            return dataViewBuilder.GetDataView();
        }

        private static void AddNumericColumn(ArrayDataViewBuilder dataViewBuilder, int numRows)
        {
            var values = new float[numRows];
            for (var i = 0; i < numRows; i++)
            {
                values[i] = i;
            }
            dataViewBuilder.AddColumn<float>("Numeric", NumberDataViewType.Single, values);
        }

        private static void AddNumericVectorColumn(ArrayDataViewBuilder dataViewBuilder, int numRows)
        {
            var slotNames = new[] { "0", "1" };
            var values = new float[numRows][];
            for (var i = 0; i < numRows; i++)
            {
                values[i] = new float[] { i, -1 * i };
            }
            dataViewBuilder.AddColumn("NumericVector", Util.GetKeyValueGetter(slotNames), NumberDataViewType.Single, values);
        }

        private static void AddStringColumn(ArrayDataViewBuilder dataViewBuilder, int numRows)
        {
            var values = new string[numRows];
            for (var i = 0; i < numRows; i++)
            {
                values[i] = i.ToString();
            }
            dataViewBuilder.AddColumn("String", values);
        }
    }
}
