// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{

    public class DatasetDimensionsTests : BaseTestClass
    {
        public DatasetDimensionsTests(ITestOutputHelper output) : base(output)
        {
        }

        public object DatasetDimensionUtil { get; private set; }

        [Fact]
        public void TextColumnDimensionsTest()
        {
            var context = new MLContext(1);
            var dataBuilder = new ArrayDataViewBuilder(context);
            dataBuilder.AddColumn("categorical", new string[] { "0", "1", "0", "1", "0", "1", "2", "2", "0", "1" });
            dataBuilder.AddColumn("text", new string[] { "0", "1", "0", "1", "0", "1", "2", "2", "0", "1" });
            var data = dataBuilder.GetDataView();
            var dimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data, new[] {
                new PurposeInference.Column(0, ColumnPurpose.CategoricalFeature),
                new PurposeInference.Column(0, ColumnPurpose.TextFeature),
            });
            Assert.NotNull(dimensions);
            Assert.Equal(2, dimensions.Length);
            Assert.Equal(3, dimensions[0].Cardinality);
            Assert.Null(dimensions[1].Cardinality);
            Assert.Null(dimensions[0].HasMissing);
            Assert.Null(dimensions[1].HasMissing);
        }

        [Fact]
        public void FloatColumnDimensionsTest()
        {
            var context = new MLContext(1);
            var dataBuilder = new ArrayDataViewBuilder(context);
            dataBuilder.AddColumn("NoNan", NumberDataViewType.Single, new float[] { 0, 1, 0, 1, 0 });
            dataBuilder.AddColumn("Nan", NumberDataViewType.Single, new float[] { 0, 1, 0, 1, float.NaN });
            var data = dataBuilder.GetDataView();
            var dimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data, new[] {
                new PurposeInference.Column(0, ColumnPurpose.NumericFeature),
                new PurposeInference.Column(1, ColumnPurpose.NumericFeature),
            });
            Assert.NotNull(dimensions);
            Assert.Equal(2, dimensions.Length);
            Assert.Null(dimensions[0].Cardinality);
            Assert.Null(dimensions[1].Cardinality);
            Assert.False(dimensions[0].HasMissing);
            Assert.True(dimensions[1].HasMissing);
        }

        [Fact]
        public void FloatVectorColumnHasNanTest()
        {
            var context = new MLContext(1);
            var dataBuilder = new ArrayDataViewBuilder(context);
            var slotNames = new[] { "Col1", "Col2" };
            var colValues = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 1, 1 },
            };
            dataBuilder.AddColumn("NoNan", Util.GetKeyValueGetter(slotNames), NumberDataViewType.Single, colValues);
            colValues = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 1, float.NaN },
            };
            dataBuilder.AddColumn("Nan", Util.GetKeyValueGetter(slotNames), NumberDataViewType.Single, colValues);
            var data = dataBuilder.GetDataView();
            var dimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data, new[] {
                new PurposeInference.Column(0, ColumnPurpose.NumericFeature),
                new PurposeInference.Column(1, ColumnPurpose.NumericFeature),
            });
            Assert.NotNull(dimensions);
            Assert.Equal(2, dimensions.Length);
            Assert.Null(dimensions[0].Cardinality);
            Assert.Null(dimensions[1].Cardinality);
            Assert.False(dimensions[0].HasMissing);
            Assert.True(dimensions[1].HasMissing);
        }
    }
}
