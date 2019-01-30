// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
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
            var metadataBuilder = new MetadataBuilder();
            metadataBuilder.Add("M", NumberType.R4, (ref float v) => v = 484f);
            var schemaBuilder = new SchemaBuilder();
            schemaBuilder.AddColumn("A", new VectorType(NumberType.R4, 94));
            schemaBuilder.AddColumn("B", new KeyType(typeof(uint), 17));
            schemaBuilder.AddColumn("C", NumberType.I4, metadataBuilder.GetMetadata());

            var shape = SchemaShape.Create(schemaBuilder.GetSchema());

            var fakeSchema = FakeSchemaFactory.Create(shape);

            var columnA = fakeSchema[0];
            var columnB = fakeSchema[1];
            var columnC = fakeSchema[2];

            Assert.Equal("A", columnA.Name);
            Assert.Equal(NumberType.R4, columnA.Type.GetItemType());
            Assert.Equal(10, columnA.Type.GetValueCount());

            Assert.Equal("B", columnB.Name);
            Assert.Equal(DataKind.U4, columnB.Type.GetRawKind());
            Assert.Equal(10u, columnB.Type.GetKeyCount());

            Assert.Equal("C", columnC.Name);
            Assert.Equal(NumberType.I4, columnC.Type);

            var metaC = columnC.Metadata;
            Assert.Single(metaC.Schema);

            float mValue = -1;
            metaC.GetValue("M", ref mValue);
            Assert.Equal(default, mValue);
        }
    }
}
