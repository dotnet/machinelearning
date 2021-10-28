// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class PurposeInferenceTests : BaseTestClass
    {
        public PurposeInferenceTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void PurposeInferenceHiddenColumnsTest()
        {
            var context = new MLContext(1);

            // build basic data view
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(DefaultColumnNames.Label, BooleanDataViewType.Instance);
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            IDataView data = DataViewTestFixture.BuildDummyDataView(schema);

            // normalize 'Features' column. this has the effect of creating 2 columns named
            // 'Features' in the data view, the first of which gets marked as 'Hidden'
            var normalizer = context.Transforms.NormalizeMinMax(DefaultColumnNames.Features);
            data = normalizer.Fit(data).Transform(data);

            // infer purposes
            var purposes = PurposeInference.InferPurposes(context, data, new ColumnInformation());

            Assert.Equal(3, purposes.Count());
            Assert.Equal(ColumnPurpose.Label, purposes[0].Purpose);
            // assert first 'Features' purpose (hidden column) is Ignore
            Assert.Equal(ColumnPurpose.Ignore, purposes[1].Purpose);
            // assert second 'Features' purpose is NumericFeature
            Assert.Equal(ColumnPurpose.NumericFeature, purposes[2].Purpose);
        }
    }
}
