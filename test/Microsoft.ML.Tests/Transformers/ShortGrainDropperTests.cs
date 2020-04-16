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
    public class ShortGrainDropperTests : TestDataPipeBase
    {
        public ShortGrainDropperTests(ITestOutputHelper output) : base(output)
        {
        }

        [NotCentOS7Fact]
        public void TestInvalidGrainTypes()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new { GrainA = 4 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.DropShortGrains(new string[] { "GrainA" }, 2);

            Assert.Throws<InvalidOperationException>(() => pipeline.Fit(data));
            Assert.Throws<InvalidOperationException>(() => pipeline.GetOutputSchema(SchemaShape.Create(data.Schema)));

            Done();
        }

        [NotCentOS7Fact]
        public void ConstructorParamterTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new { GrainA = 4 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Make sure invalid constructor args throw.
            Assert.Throws<InvalidOperationException>(() => mlContext.Transforms.DropShortGrains(new string[] { "GrainA" }, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => mlContext.Transforms.DropShortGrains(new string[] { }, 2));
            Assert.Throws<ArgumentNullException>(() => mlContext.Transforms.DropShortGrains(null, 2));

            Done();
        }

        [NotCentOS7Fact]
        public void SimpleDropTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new { GrainA = "Grain", Value = 0 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            var pipeline = mlContext.Transforms.DropShortGrains(new string[] { "GrainA" }, 2);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 2 and we only have 1, should have no rows back.
            Assert.True(rows.Length == 0);

            // Also make sure that the value column was dropped correctly.
            Assert.True(cols[0].Values.Length == 0);
            Assert.True(cols[1].Values.Length == 0);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void SimpleKeepTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", Value = 0 },
                new { GrainA = "Grain", Value = 1 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            var pipeline = mlContext.Transforms.DropShortGrains(new string[] { "GrainA" }, 2);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 2 and we have 2, should have all rows back.
            Assert.True(rows.Length == 2);

            // Also make sure that the value column was kept correctly.
            Assert.True(cols[0].Values.Length == 2);
            Assert.True(cols[1].Values.Length == 2);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void MultipleGrainsKeepTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "GrainA", Value = 0 },
                new { GrainA = "GrainB", Value = 1 },
                new { GrainA = "GrainA", Value = 2 },
                new { GrainA = "GrainB", Value = 3 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            var pipeline = mlContext.Transforms.DropShortGrains(new string[] { "GrainA" }, 2);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 2 and we have 2 of grain A and 2 of grain B, should have all rows back.
            Assert.True(rows.Length == 4);

            // Also make sure that the value column was kept correctly.
            Assert.True(cols[0].Values.Length == 4);
            Assert.True(cols[1].Values.Length == 4);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void MultipleGrainsPartialDropTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "GrainA", Value = 0 },
                new { GrainA = "GrainB", Value = 0 },
                new { GrainA = "GrainA", Value = 1 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            var pipeline = mlContext.Transforms.DropShortGrains(new string[] { "GrainA" }, 2);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 2 and we have 2 of grain A and 1 of grain B, should have all GrainA rows back and not GrainB.
            Assert.True(rows.Length == 2);

            // Also make sure that the value column was kept correctly.
            Assert.True(cols[0].Values.Length == 2);
            Assert.True(cols[1].Values.Length == 2);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void MultipleGrainsAllDropTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "GrainA", Value = 0 },
                new { GrainA = "GrainB", Value = 0 },
                new { GrainA = "GrainA", Value = 1 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            var pipeline = mlContext.Transforms.DropShortGrains(new string[] { "GrainA" }, 3);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 3 and we have 3 of grain A and 1 of grain B, should have no rows back.
            Assert.True(rows.Length == 0);

            // Also make sure that the value column was kept correctly.
            Assert.True(cols[0].Values.Length == 0);
            Assert.True(cols[1].Values.Length == 0);

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}
