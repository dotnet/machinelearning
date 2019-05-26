// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class DatasetDimensionsTests
    {
        public object DatasetDimensionUtil { get; private set; }

        [TestMethod]
        public void StringColumnDimensionsTest()
        {
            var context = new MLContext();
            var dataBuilder = new ArrayDataViewBuilder(context);
            dataBuilder.AddColumn("String", new string[] { "0", "1", "0", "1", "0", "1", "2", "2", "0", "1", "two words", "" });
            var data = dataBuilder.GetDataView();
            var dimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data);
            Assert.IsNotNull(dimensions);
            Assert.AreEqual(1, dimensions.Length);
            Assert.AreEqual(5, dimensions[0].Cardinality);
            Assert.IsTrue(dimensions[0].HasMissingValues());
            Assert.AreEqual(1, dimensions[0].MissingValueCount);
            Assert.IsTrue(dimensions[0].SummaryStatistics.Mean > 0);
        }

        [TestMethod]
        public void FloatColumnDimensionsTest()
        {
            var context = new MLContext();
            var dataBuilder = new ArrayDataViewBuilder(context);
            dataBuilder.AddColumn("Float", NumberDataViewType.Single, new float[] { 0, 1, 0, 1, 0 });
            dataBuilder.AddColumn("NaN", NumberDataViewType.Single, new float[] { 0, 1, 0, 1, float.NaN });
            var data = dataBuilder.GetDataView();
            var dimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data);
            Assert.IsNotNull(dimensions);
            Assert.AreEqual(2, dimensions.Length);
            Assert.AreEqual(2, dimensions[0].Cardinality);
            Assert.AreEqual(3, dimensions[1].Cardinality);
            Assert.AreEqual(false, dimensions[0].HasMissingValues());
            Assert.AreEqual(true, dimensions[1].HasMissingValues());
            Assert.AreEqual(0.4, dimensions[0].SummaryStatistics.Mean);
            Assert.AreEqual(0.5, dimensions[1].SummaryStatistics.Mean);
        }

        [TestMethod]
        public void FloatVectorColumnHasNaNTest()
        {
            var context = new MLContext();
            var dataBuilder = new ArrayDataViewBuilder(context);
            var slotNames = new[] { "Col1", "Col2" };
            var colValues = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 1, 1 },
            };
            dataBuilder.AddColumn("Vector", Util.GetKeyValueGetter(slotNames), NumberDataViewType.Single, colValues);
            colValues = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 1, float.NaN },
            };
            dataBuilder.AddColumn("NaN", Util.GetKeyValueGetter(slotNames), NumberDataViewType.Single, colValues);
            var data = dataBuilder.GetDataView();
            var dimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data);
            Assert.IsNotNull(dimensions);
            Assert.AreEqual(2, dimensions.Length);
            Assert.AreEqual(null, dimensions[0].Cardinality);
            Assert.AreEqual(null, dimensions[1].Cardinality);
            Assert.AreEqual(false, dimensions[0].HasMissingValues());
            Assert.AreEqual(true, dimensions[1].HasMissingValues());
        }
    }
}
