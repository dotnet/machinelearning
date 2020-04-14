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

        [Fact]
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

        [Fact]
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
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var calculationAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Calculation")).First().Name;
            var minWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MinWindowSize")).First().Name;
            var maxWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MaxWindowSize")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>(calculationAnnotationName, ref calculation);
            annotations.GetValue<UInt32>(minWindowSizeAnnotationName, ref minWindowSize);
            annotations.GetValue<UInt32>(maxWindowSizeAnnotationName, ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Mean", calculation.ToString());
            Assert.Equal((UInt32)1, minWindowSize);
            Assert.Equal((UInt32)1, maxWindowSize);

            Done();
        }

        [Fact]
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
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var calculationAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Calculation")).First().Name;
            var minWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MinWindowSize")).First().Name;
            var maxWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MaxWindowSize")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>(calculationAnnotationName, ref calculation);
            annotations.GetValue<UInt32>(minWindowSizeAnnotationName, ref minWindowSize);
            annotations.GetValue<UInt32>(maxWindowSizeAnnotationName, ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Mean", calculation.ToString());
            Assert.Equal((UInt32)2, minWindowSize);
            Assert.Equal((UInt32)3, maxWindowSize);

            Done();
        }

        [Fact]
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

        [Fact]
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
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var calculationAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Calculation")).First().Name;
            var minWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MinWindowSize")).First().Name;
            var maxWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MaxWindowSize")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>(calculationAnnotationName, ref calculation);
            annotations.GetValue<UInt32>(minWindowSizeAnnotationName, ref minWindowSize);
            annotations.GetValue<UInt32>(maxWindowSizeAnnotationName, ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Min", calculation.ToString());
            Assert.Equal((UInt32)1, minWindowSize);
            Assert.Equal((UInt32)1, maxWindowSize);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [Fact]
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
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var calculationAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Calculation")).First().Name;
            var minWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MinWindowSize")).First().Name;
            var maxWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MaxWindowSize")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>(calculationAnnotationName, ref calculation);
            annotations.GetValue<UInt32>(minWindowSizeAnnotationName, ref minWindowSize);
            annotations.GetValue<UInt32>(maxWindowSizeAnnotationName, ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Max", calculation.ToString());
            Assert.Equal((UInt32)1, minWindowSize);
            Assert.Equal((UInt32)1, maxWindowSize);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [Fact]
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
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var calculationAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Calculation")).First().Name;
            var minWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MinWindowSize")).First().Name;
            var maxWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MaxWindowSize")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>(calculationAnnotationName, ref calculation);
            annotations.GetValue<UInt32>(minWindowSizeAnnotationName, ref minWindowSize);
            annotations.GetValue<UInt32>(maxWindowSizeAnnotationName, ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Mean", calculation.ToString());
            Assert.Equal((UInt32)1, minWindowSize);
            Assert.Equal((UInt32)1, maxWindowSize);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [Fact]
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
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var calculationAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Calculation")).First().Name;
            var minWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MinWindowSize")).First().Name;
            var maxWindowSizeAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("MaxWindowSize")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>(calculationAnnotationName, ref calculation);
            annotations.GetValue<UInt32>(minWindowSizeAnnotationName, ref minWindowSize);
            annotations.GetValue<UInt32>(maxWindowSizeAnnotationName, ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Mean", calculation.ToString());
            Assert.Equal((UInt32)1, minWindowSize);
            Assert.Equal((UInt32)1, maxWindowSize);

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}
