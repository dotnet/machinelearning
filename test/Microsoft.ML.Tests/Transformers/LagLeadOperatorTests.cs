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
    public class LagLeadOperatorTests : TestDataPipeBase
    {
        public LagLeadOperatorTests(ITestOutputHelper output) : base(output)
        {
        }

        [NotCentOS7Fact]
        public void TestInvalidType()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new { GrainA = "Grain", ColA = "Invalid Type" } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 1, new long[] { -1 });

            Assert.Throws<InvalidOperationException>(() => pipeline.Fit(data));
            Assert.Throws<InvalidOperationException>(() => pipeline.GetOutputSchema(SchemaShape.Create(data.Schema)));

            Done();
        }

        [NotCentOS7Fact]
        public void ConstructorParameterTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Make sure invalid constructor args throw.
            Assert.Throws<ArgumentNullException>(() => mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, null, 1, new long[] { -1 }));
            Assert.Throws<InvalidOperationException>(() => mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 0, new long[] { -1 }));

            Assert.Throws<ArgumentOutOfRangeException>(() => mlContext.Transforms.CreateLagsAndLeads(new string[] { }, "ColA", 1, new long[] { -1 }));
            Assert.Throws<ArgumentNullException>(() => mlContext.Transforms.CreateLagsAndLeads(null, "ColA", 1, new long[] { -1 }));

            Assert.Throws<ArgumentOutOfRangeException>(() => mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 1, new long[] { }));
            Assert.Throws<ArgumentNullException>(() => mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 1, null));

            Done();
        }

        [NotCentOS7Fact]
        public void SimpleSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 1, new long[] { -1 });
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
            VBuffer<long> offsets = default;
            VBuffer<long> offsetsExpectedOutput = new VBuffer<long>(1, new long[] { -1 });

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var offsetsAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Offsets")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<VBuffer<long>>(offsetsAnnotationName, ref offsets);

            Assert.Equal("LagLead", featurizerName.ToString());
            Assert.Equal(offsetsExpectedOutput.DenseValues(), offsets.DenseValues());

            Done();
        }

        [NotCentOS7Fact]
        public void ComplexSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA_New", 3, new long[] { -1, 1, -2, 2 }, "ColA");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA_New"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 4);
            Assert.True(columnType.Dimensions[1] == 3);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            VBuffer<long> offsets = default;
            VBuffer<long> offsetsExpectedOutput = new VBuffer<long>(4, new long[] { -1, 1, -2, 2 });

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var offsetsAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Offsets")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<VBuffer<long>>(offsetsAnnotationName, ref offsets);

            Assert.Equal("LagLead", featurizerName.ToString());
            Assert.Equal(offsetsExpectedOutput.DenseValues(), offsets.DenseValues());

            Done();
        }

        [NotCentOS7Fact]
        public void SimpleLagTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0, ColB = 2 },
                new { GrainA = "Grain", ColA = 2.0, ColB = 2 },
                new { GrainA = "Grain", ColA = 3.0, ColB = 2 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 1, new long[] { -1 });
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
            VBuffer<long> offsets = default;
            VBuffer<long> offsetsExpectedOutput = new VBuffer<long>(1, new long[] { -1 });

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var offsetsAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Offsets")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<VBuffer<long>>(offsetsAnnotationName, ref offsets);

            Assert.Equal("LagLead", featurizerName.ToString());
            Assert.Equal(offsetsExpectedOutput.DenseValues(), offsets.DenseValues());

            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { double.NaN }, new[] { 1d }, new[] { 2d } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);
            //var getterB = cursor.GetGetter<int>(schema["ColB"]);

            VBuffer<double> buffer = default;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index++][0], bufferValues[0]);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void LagTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 2, new long[] { -2, -1 });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 2);
            Assert.True(columnType.Dimensions[1] == 2);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            VBuffer<long> offsets = default;
            VBuffer<long> offsetsExpectedOutput = new VBuffer<long>(2, new long[] { -2, -1 });

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var offsetsAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Offsets")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<VBuffer<long>>(offsetsAnnotationName, ref offsets);

            Assert.Equal("LagLead", featurizerName.ToString());
            Assert.Equal(offsetsExpectedOutput.DenseValues(), offsets.DenseValues());

            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { double.NaN, double.NaN, double.NaN, double.NaN }, new[] { double.NaN, double.NaN, double.NaN, 1d }, new[] { double.NaN, 1d, 1d, 2d } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);

            VBuffer<double> buffer = default;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();
                for (int i = 0; i < expectedOutput[0].Length; i++)
                {
                    Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                    Assert.Equal(expectedOutput[index][i], bufferValues[i]);
                }
                index += 1;
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void LargeNumberTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 500000, new long[] { 500000 });
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
            Assert.True(columnType.Dimensions[1] == 500000);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            VBuffer<long> offsets = default;
            VBuffer<long> offsetsExpectedOutput = new VBuffer<long>(1, new long[] { 500000 });

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var offsetsAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Offsets")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<VBuffer<long>>(offsetsAnnotationName, ref offsets);

            Assert.Equal("LagLead", featurizerName.ToString());
            Assert.Equal(offsetsExpectedOutput.DenseValues(), offsets.DenseValues());

            output.Preview();

            Done();
        }


        [NotCentOS7Fact]
        public void SimpleLeadTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0, ColB = 1 },
                new { GrainA = "Grain", ColA = 2.0, ColB = 2 },
                new { GrainA = "Grain", ColA = 3.0, ColB = 3 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 1, new long[] { 1 });
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
            VBuffer<long> offsets = default;
            VBuffer<long> offsetsExpectedOutput = new VBuffer<long>(1, new long[] { 1 });

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var offsetsAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Offsets")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<VBuffer<long>>(offsetsAnnotationName, ref offsets);

            Assert.Equal("LagLead", featurizerName.ToString());
            Assert.Equal(offsetsExpectedOutput.DenseValues(), offsets.DenseValues());

            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { 2d }, new[] { 3d }, new[] { double.NaN } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);
            var getterB = cursor.GetGetter<int>(schema["ColB"]);

            VBuffer<double> buffer = default;
            int colBbuffer = default;
            var colBValue = 1;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index++][0], bufferValues[0]);

                getterB(ref colBbuffer);
                Assert.Equal(colBValue++, colBbuffer);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void SimpleLeadOffset2Test()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0, ColB = 1 },
                new { GrainA = "Grain", ColA = 2.0, ColB = 2 },
                new { GrainA = "Grain", ColA = 3.0, ColB = 3 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 1, new long[] { 2 });
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
            VBuffer<long> offsets = default;
            VBuffer<long> offsetsExpectedOutput = new VBuffer<long>(1, new long[] { 2 });

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var offsetsAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Offsets")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<VBuffer<long>>(offsetsAnnotationName, ref offsets);

            Assert.Equal("LagLead", featurizerName.ToString());
            Assert.Equal(offsetsExpectedOutput.DenseValues(), offsets.DenseValues());

            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { 3d }, new[] { double.NaN }, new[] { double.NaN } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);
            var getterB = cursor.GetGetter<int>(schema["ColB"]);

            VBuffer<double> buffer = default;
            int colBbuffer = default;
            var colBValue = 1;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index++][0], bufferValues[0]);

                getterB(ref colBbuffer);
                Assert.Equal(colBValue++, colBbuffer);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void SimpleLagLeadOffset2Test()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0, ColB = 1 },
                new { GrainA = "Grain", ColA = 2.0, ColB = 2 },
                new { GrainA = "Grain", ColA = 3.0, ColB = 3 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 1, new long[] { -2, 2 });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 2);
            Assert.True(columnType.Dimensions[1] == 1);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            VBuffer<long> offsets = default;
            VBuffer<long> offsetsExpectedOutput = new VBuffer<long>(2, new long[] { -2, 2 });

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var offsetsAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Offsets")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<VBuffer<long>>(offsetsAnnotationName, ref offsets);

            Assert.Equal("LagLead", featurizerName.ToString());
            Assert.Equal(offsetsExpectedOutput.DenseValues(), offsets.DenseValues());

            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { double.NaN, 3d }, new[] { double.NaN, double.NaN }, new[] { 1d, double.NaN } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);
            var getterB = cursor.GetGetter<int>(schema["ColB"]);

            VBuffer<double> buffer = default;
            int colBbuffer = default;
            var colBValue = 1;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index][0], bufferValues[0]);
                Assert.Equal(expectedOutput[index++][1], bufferValues[1]);

                getterB(ref colBbuffer);
                Assert.Equal(colBValue++, colBbuffer);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void MoreComplexLagLeadOffsetTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0, ColB = 1 },
                new { GrainA = "Grain", ColA = 2.0, ColB = 2 },
                new { GrainA = "Grain", ColA = 3.0, ColB = 3 },
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            var offsets = new long[] { -2, -1, 1, 2 };
            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.CreateLagsAndLeads(new string[] { "GrainA" }, "ColA", 2, new long[] { -2, -1, 1, 2 });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 4);
            Assert.True(columnType.Dimensions[1] == 2);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            VBuffer<long> offsetsBuf = default;
            VBuffer<long> offsetsExpectedOutput = new VBuffer<long>(offsets.Length, offsets);

            var annotations = addedColumn.Annotations;
            var feautizerAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("FeaturizerName")).First().Name;
            var offsetsAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("Offsets")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(feautizerAnnotationName, ref featurizerName);
            annotations.GetValue<VBuffer<long>>(offsetsAnnotationName, ref offsetsBuf);

            Assert.Equal("LagLead", featurizerName.ToString());
            Assert.Equal(offsetsExpectedOutput.DenseValues(), offsetsBuf.DenseValues());

            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[]
            {
                new[] { double.NaN, double.NaN, double.NaN, double.NaN, 1d, 2d, 2d, 3d },
                new[] { double.NaN, double.NaN, double.NaN, 1d, 2d, 2d, 3d, double.NaN },
                new[] { double.NaN, 1d, 2d, 2d, 3d, double.NaN, double.NaN, double.NaN }
            };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);
            var getterB = cursor.GetGetter<int>(schema["ColB"]);

            VBuffer<double> buffer = default;
            int colBbuffer = default;
            var colBValue = 1;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index][0], bufferValues[0]);
                Assert.Equal(expectedOutput[index++][1], bufferValues[1]);

                getterB(ref colBbuffer);
                Assert.Equal(colBValue++, colBbuffer);
            }

            TestEstimatorCore(pipeline, data);
            Done();
        }
    }
}
