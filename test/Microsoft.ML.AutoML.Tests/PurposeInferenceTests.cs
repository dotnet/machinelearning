using System.Linq;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class PurposeInferenceTests
    {
        [TestMethod]
        public void PurposeInferenceHiddenColumnsTest()
        {
            var context = new MLContext();

            // build basic data view
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(DefaultColumnNames.Label, BooleanDataViewType.Instance);
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            IDataView data = new EmptyDataView(context, schema);

            // normalize 'Features' column. this has the effect of creating 2 columns named
            // 'Features' in the data view, the first of which gets marked as 'Hidden'
            var normalizer = context.Transforms.NormalizeMinMax(DefaultColumnNames.Features);
            data = normalizer.Fit(data).Transform(data);

            // infer purposes
            var purposes = PurposeInference.InferPurposes(context, data, new ColumnInformation());

            Assert.AreEqual(3, purposes.Count());
            Assert.AreEqual(ColumnPurpose.Label, purposes[0].Purpose);
            // assert first 'Features' purpose (hidden column) is Ignore
            Assert.AreEqual(ColumnPurpose.Ignore, purposes[1].Purpose);
            // assert second 'Features' purpose is NumericFeature
            Assert.AreEqual(ColumnPurpose.NumericFeature, purposes[2].Purpose);
        }
    }
}
