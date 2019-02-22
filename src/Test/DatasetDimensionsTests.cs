using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class DatasetDimensionsTests
    {
        public object DatasetDimensionUtil { get; private set; }

        [TestMethod]
        public void TextColumnDimensionsTest()
        {
            var context = new MLContext();
            var dataBuilder = new ArrayDataViewBuilder(context);
            dataBuilder.AddColumn("categorical", new string[] { "0", "1", "0", "1", "0", "1", "2", "2", "0", "1" });
            dataBuilder.AddColumn("text", new string[] { "0", "1", "0", "1", "0", "1", "2", "2", "0", "1" });
            var data = dataBuilder.GetDataView();
            var dimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data, new[] {
                new PurposeInference.Column(0, ColumnPurpose.CategoricalFeature),
                new PurposeInference.Column(0, ColumnPurpose.TextFeature),
            });
            Assert.IsNotNull(dimensions);
            Assert.AreEqual(2, dimensions.Length);
            Assert.AreEqual(3, dimensions[0].Cardinality);
            Assert.AreEqual(null, dimensions[1].Cardinality);
            Assert.IsNull(dimensions[0].HasMissing);
            Assert.IsNull(dimensions[1].HasMissing);
        }

        [TestMethod]
        public void FloatColumnDimensionsTest()
        {
            var context = new MLContext();
            var dataBuilder = new ArrayDataViewBuilder(context);
            dataBuilder.AddColumn("NoNan", NumberType.R4, new float[] { 0, 1, 0, 1, 0 });
            dataBuilder.AddColumn("Nan", NumberType.R4, new float[] { 0, 1, 0, 1, float.NaN });
            var data = dataBuilder.GetDataView();
            var dimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data, new[] {
                new PurposeInference.Column(0, ColumnPurpose.NumericFeature),
                new PurposeInference.Column(1, ColumnPurpose.NumericFeature),
            });
            Assert.IsNotNull(dimensions);
            Assert.AreEqual(2, dimensions.Length);
            Assert.AreEqual(null, dimensions[0].Cardinality);
            Assert.AreEqual(null, dimensions[1].Cardinality);
            Assert.AreEqual(false, dimensions[0].HasMissing);
            Assert.AreEqual(true, dimensions[1].HasMissing);
        }

        [TestMethod]
        public void FloatVectorColumnHasNanTest()
        {
            var context = new MLContext();
            var dataBuilder = new ArrayDataViewBuilder(context);
            var slotNames = new[] { "Col1", "Col2" };
            var colValues = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 1, 1 },
            };
            dataBuilder.AddColumn("NoNan", Util.GetKeyValueGetter(slotNames), NumberType.R4, colValues);
            colValues = new float[][]
            {
                new float[] { 0, 0 },
                new float[] { 1, float.NaN },
            };
            dataBuilder.AddColumn("Nan", Util.GetKeyValueGetter(slotNames), NumberType.R4, colValues);
            var data = dataBuilder.GetDataView();
            var dimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data, new[] {
                new PurposeInference.Column(0, ColumnPurpose.NumericFeature),
                new PurposeInference.Column(1, ColumnPurpose.NumericFeature),
            });
            Assert.IsNotNull(dimensions);
            Assert.AreEqual(2, dimensions.Length);
            Assert.AreEqual(null, dimensions[0].Cardinality);
            Assert.AreEqual(null, dimensions[1].Cardinality);
            Assert.AreEqual(false, dimensions[0].HasMissing);
            Assert.AreEqual(true, dimensions[1].HasMissing);
        }
    }
}
