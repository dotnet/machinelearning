// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Data.DataLoadSave;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class FakeSchemaTest : BaseTestClass
    {
        public FakeSchemaTest(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        void SimpleTest()
        {
            var metadataBuilder = new DataViewSchema.Annotations.Builder();
            metadataBuilder.Add("M", NumberDataViewType.Single, (ref float v) => v = 484f);
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("A", new VectorType(NumberDataViewType.Single, 94));
            schemaBuilder.AddColumn("B", new KeyType(typeof(uint), 17));
            schemaBuilder.AddColumn("C", NumberDataViewType.Int32, metadataBuilder.ToAnnotations());

            var shape = SchemaShape.Create(schemaBuilder.ToSchema());

            var fakeSchema = FakeSchemaFactory.Create(shape);

            var columnA = fakeSchema[0];
            var columnB = fakeSchema[1];
            var columnC = fakeSchema[2];

            Assert.Equal("A", columnA.Name);
            Assert.Equal(NumberDataViewType.Single, columnA.Type.GetItemType());
            Assert.Equal(10, columnA.Type.GetValueCount());

            Assert.Equal("B", columnB.Name);
            Assert.Equal(InternalDataKind.U4, columnB.Type.GetRawKind());
            Assert.Equal(10u, columnB.Type.GetKeyCount());

            Assert.Equal("C", columnC.Name);
            Assert.Equal(NumberDataViewType.Int32, columnC.Type);

            var metaC = columnC.Annotations;
            Assert.Single(metaC.Schema);

            float mValue = -1;
            metaC.GetValue("M", ref mValue);
            Assert.Equal(default, mValue);
        }
    }
}
