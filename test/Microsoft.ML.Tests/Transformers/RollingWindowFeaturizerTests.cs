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
    public class RollingWindowFeaturizerTests : TestDataPipeBase
    {
        public RollingWindowFeaturizerTests(ITestOutputHelper output) : base(output)
        {
        }

        [NotCentOS7Fact]
        public void TestInvalidType()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new { GrainA = "Grain", ColA = "Invalid Type" } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 2, 2, 2);

            Assert.Throws<InvalidOperationException>(() => pipeline.Fit(data));
            Assert.Throws<InvalidOperationException>(() => pipeline.GetOutputSchema(SchemaShape.Create(data.Schema)));

            Done();
        }

        [NotCentOS7Fact]
        public void SimpleSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 },
                new { GrainA = "Grain", ColA = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 1);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 1);
            Assert.True(columnType.Dimensions[1] == 1);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> columnName = default;

            var annotations = addedColumn.Annotations;
            var columnAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(columnAnnotationName, ref columnName);

            Assert.Equal("ColA_Mean_MinWin1_MaxWin1", columnName.ToString());

            Done();
        }

        [NotCentOS7Fact]
        public void ComplexSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 },
                new { GrainA = "Grain", ColA = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "NewInputColumn", RollingWindowEstimator.RollingWindowCalculation.Mean, 4, 3, 2, "ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["NewInputColumn"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 1);
            Assert.True(columnType.Dimensions[1] == 4);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> columnName = default;

            var annotations = addedColumn.Annotations;
            var columnAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(columnAnnotationName, ref columnName);

            Assert.Equal("NewInputColumn_Mean_MinWin2_MaxWin3", columnName.ToString());

            Done();
        }

        [NotCentOS7Fact]
        public void ConstructorParameterTest() {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Make sure invalid constructor args throw.
            Assert.Throws<InvalidOperationException>(() => mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 0, 1));
            Assert.Throws<InvalidOperationException>(() => mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 0));
            Assert.Throws<InvalidOperationException>(() => mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 1, 0));
            Assert.Throws<InvalidOperationException>(() => mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 1, 2));
            Assert.Throws<ArgumentOutOfRangeException>(() => mlContext.Transforms.RollingWindow(new string[] { }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 1, 2));
            Assert.Throws<ArgumentNullException>(() => mlContext.Transforms.RollingWindow(null, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 1, 2));

            Done();
        }

        [NotCentOS7Fact]
        public void SimpleMinTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 },
                new { GrainA = "Grain", ColA = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Min, 1, 1);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA"];
            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { double.NaN }, new[] { 1d }, new[] { 2d }, new[] { 3d } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);

            VBuffer<double> buffer = default;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index++][0], bufferValues[0]);
            }

            // Verify annotations are correct.
            ReadOnlyMemory<char> columnName = default;

            var annotations = addedColumn.Annotations;
            var columnAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(columnAnnotationName, ref columnName);

            Assert.Equal("ColA_Min_MinWin1_MaxWin1", columnName.ToString());

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void SimpleMaxTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 },
                new { GrainA = "Grain", ColA = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Max, 1, 1);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA"];
            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { double.NaN }, new[] { 1d }, new[] { 2d }, new[] { 3d } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);

            VBuffer<double> buffer = default;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index++][0], bufferValues[0]);
            }

            // Verify annotations are correct.
            ReadOnlyMemory<char> columnName = default;

            var annotations = addedColumn.Annotations;
            var columnAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(columnAnnotationName, ref columnName);

            Assert.Equal("ColA_Max_MinWin1_MaxWin1", columnName.ToString());

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void SimpleMeanTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 },
                new { GrainA = "Grain", ColA = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 1);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA"];
            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { double.NaN }, new[] { 1d }, new[] { 2d }, new[] { 3d } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);

            VBuffer<double> buffer = default;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index++][0], bufferValues[0]);
            }

            // Verify annotations are correct.
            ReadOnlyMemory<char> columnName = default;

            var annotations = addedColumn.Annotations;
            var columnAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(columnAnnotationName, ref columnName);

            Assert.Equal("ColA_Mean_MinWin1_MaxWin1", columnName.ToString());

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void MultipleGrains()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "GrainOne", ColA = 1.0 },
                new { GrainA = "GrainOne", ColA = 2.0 },
                new { GrainA = "GrainTwo", ColA = 1.0 },
                new { GrainA = "GrainTwo", ColA = 2.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 1);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA"];
            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { double.NaN }, new[] { 1d }, new[] { double.NaN }, new[] { 1d } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);

            VBuffer<double> buffer = default;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index++][0], bufferValues[0]);
            }

            // Verify annotations are correct.
            ReadOnlyMemory<char> columnName = default;

            var annotations = addedColumn.Annotations;
            var columnAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(columnAnnotationName, ref columnName);

            Assert.Equal("ColA_Mean_MinWin1_MaxWin1", columnName.ToString());

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}
