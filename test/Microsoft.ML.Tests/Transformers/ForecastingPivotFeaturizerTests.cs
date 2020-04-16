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
    public class ForecastingPivotFeaturizerTests : TestDataPipeBase
    {
        public ForecastingPivotFeaturizerTests(ITestOutputHelper output) : base(output)
        {
        }
        private class SimpleRWTestData
        {
            public double ColA { get; set; }

            [VectorType(1, 2)]
            public double[] ColA_Vec { get; set; }
        }
        private class SimpleLagLeadTestData
        {
            public double ColA { get; set; }

            [VectorType(2, 2)]
            public double[] ColA_Vec { get; set; }
        }

        private class Horizon2LagLeadRWTestData
        {
            public double ColA { get; set; }

            [VectorType(1, 2)]
            public double[] ColA_RW { get; set; }

            [VectorType(2, 2)]
            public double[] ColA_LL { get; set; }
        }

        [NotCentOS7Fact]
        public void TestInvalidType()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { ColA_RW_Mean_MinWin1_MaxWin1 = 1.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.PivotForecastingData(new string[] { "ColA_RW_Mean_MinWin1_MaxWin1" });

            Assert.Throws<InvalidOperationException>(() => pipeline.Fit(data));
            Assert.Throws<InvalidOperationException>(() => pipeline.GetOutputSchema(SchemaShape.Create(data.Schema)));

            Done();
        }

        [NotCentOS7Fact]
        public void SimpleSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = new [] { 1.0 } }
            };

            var annotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "ColA_Mean_MinWin1_MaxWin1".AsMemory();

            annotations.Add<ReadOnlyMemory<char>>("ColumnNames=ColA_Mean_MinWin1_MaxWin1", TextDataViewType.Instance, nameValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("GrainA", TextDataViewType.Instance);
            schemaBuilder.AddColumn("ColA", new VectorDataViewType(NumberDataViewType.Double, 1, 1), annotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            // Build the pipeline.
            var pipeline = mlContext.Transforms.PivotForecastingData(new string[] { "ColA" });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA_Mean_MinWin1_MaxWin1"];
            var columnType = addedColumn.Type;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType == NumberDataViewType.Double);

            addedColumn = schema["Horizon"];
            columnType = addedColumn.Type;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType == NumberDataViewType.UInt32);

            Done();
        }

        [NotCentOS7Fact]
        public void SimpleLagLeadSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new SimpleRWTestData { ColA = 1.0, ColA_Vec = new [] { double.NaN, double.NaN } },
                new SimpleRWTestData { ColA = 2.0, ColA_Vec = new [] { double.NaN, 1.0 } },
                new SimpleRWTestData { ColA = 3.0, ColA_Vec = new [] { 1.0, 2.0 } }
            };

            var annotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "ColA_Vec_Lag1,ColA_Vec_Lead1".AsMemory();

            annotations.Add<ReadOnlyMemory<char>>("ColumnNames=ColA_Vec_Lag1,ColA_Vec_Lead1", TextDataViewType.Instance, nameValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("ColA", NumberDataViewType.Double);
            schemaBuilder.AddColumn("ColA_Vec", new VectorDataViewType(NumberDataViewType.Double, 1, 2), annotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            // Build the pipeline. Starting with RollingWindow since this depends on RollingWindow or LagLead.
            var pipeline = mlContext.Transforms.PivotForecastingData(new string[] { "ColA_Vec" });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            // 3 columns have been added.
            // 1 Lag column, 1 Lead column, and 1 horizon column
            var addedLagColumn = schema["ColA_Vec_Lag1"];
            var lagColumnType = addedLagColumn.Type;

            var addedLeadColumn = schema["ColA_Vec_Lead1"];
            var leadColumnType = addedLeadColumn.Type;

            var addedHorizonColumn = schema["Horizon"];
            var horizonColumnType = addedHorizonColumn.Type;

            // Make sure the type and schema of the column are correct.
            Assert.True(lagColumnType == NumberDataViewType.Double);

            Assert.True(leadColumnType == NumberDataViewType.Double);

            Assert.True(horizonColumnType == NumberDataViewType.UInt32);

            Done();
        }

        [NotCentOS7Fact]
        public void SimpleTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = new [] { double.NaN } },
                new { GrainA = "Grain", ColA = new [] { 1.0 } },
                new { GrainA = "Grain", ColA = new [] { 2.0 } }
            };

            var annotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "ColA_Mean_MinWin1_MaxWin1".AsMemory();

            annotations.Add<ReadOnlyMemory<char>>("ColumnNames=ColA_Mean_MinWin1_MaxWin1", TextDataViewType.Instance, nameValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("GrainA", TextDataViewType.Instance);
            schemaBuilder.AddColumn("ColA", new VectorDataViewType(NumberDataViewType.Double, 1, 1), annotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            // Build the pipeline
            var pipeline = mlContext.Transforms.PivotForecastingData(new string[] { "ColA" });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var horizonCol = debugView.ColumnView[2].Values;
            var pivotCol = debugView.ColumnView[3].Values;

            // Length should be 2 because we drop the first row.
            Assert.True(pivotCol.Length == 2);

            // Make sure the values are correct.
            Assert.Equal(1.0, pivotCol[0]);
            Assert.Equal(2.0, pivotCol[1]);

            Assert.Equal((UInt32)1, horizonCol[0]);
            Assert.Equal((UInt32)1, horizonCol[1]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void DifferentHorizonNameTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = new [] { double.NaN } },
                new { GrainA = "Grain", ColA = new [] { 1.0 } },
                new { GrainA = "Grain", ColA = new [] { 2.0 } }
            };

            var annotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "ColA_Mean_MinWin1_MaxWin1".AsMemory();

            annotations.Add<ReadOnlyMemory<char>>("ColumnNames=ColA_Mean_MinWin1_MaxWin1", TextDataViewType.Instance, nameValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("GrainA", TextDataViewType.Instance);
            schemaBuilder.AddColumn("ColA", new VectorDataViewType(NumberDataViewType.Double, 1, 1), annotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            // Build the pipeline
            var pipeline = mlContext.Transforms.PivotForecastingData(new string[] { "ColA" }, "NotHorizon");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var horizonCol = debugView.ColumnView[2].Values;
            var pivotCol = debugView.ColumnView[3].Values;

            // Make sure the different horizon name is correct.
            Assert.Equal("NotHorizon", debugView.ColumnView[2].Column.Name);

            // Length should be 2 because we drop the first row.
            Assert.True(pivotCol.Length == 2);

            // Make sure the values are correct.
            Assert.Equal(1.0, pivotCol[0]);
            Assert.Equal(2.0, pivotCol[1]);

            Assert.Equal((UInt32)1, horizonCol[0]);
            Assert.Equal((UInt32)1, horizonCol[1]);

            TestEstimatorCore(pipeline, data);
            Done();
        }


        [NotCentOS7Fact]
        public void SimpleLagLeadTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new SimpleLagLeadTestData { ColA = 1.0, ColA_Vec = new [] { double.NaN, double.NaN, double.NaN, double.NaN } },
                new SimpleLagLeadTestData { ColA = 2.0, ColA_Vec = new [] { double.NaN, 1.0, double.NaN, 1.0 } },
                new SimpleLagLeadTestData { ColA = 3.0, ColA_Vec = new [] { 1.0, 2.0, 1.0, 2.0 } }
            };

            var annotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<ReadOnlyMemory<char>> lagLeadNameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "ColA_Vec_Lag1,ColA_Vec_Lead1".AsMemory();

            annotations.Add<ReadOnlyMemory<char>>("ColumnNames=ColA_LL_Lag1,ColA_LL_Lead1", TextDataViewType.Instance, lagLeadNameValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("ColA", NumberDataViewType.Double);
            schemaBuilder.AddColumn("ColA_Vec", new VectorDataViewType(NumberDataViewType.Double, 2, 2), annotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            // Build the pipeline
            var pipeline = mlContext.Transforms.PivotForecastingData(new string[] { "ColA_Vec" });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var colA = debugView.ColumnView[0].Values;
            var horizonCol = debugView.ColumnView[2].Values;
            var lagCol = debugView.ColumnView[3].Values;
            var leadCol = debugView.ColumnView[4].Values;

            // Correct output for:
            // ColA,    ColA_Lag_1, ColA_Lead_1,    Horizon
            // 2.0,     1.0,        1.0,            1
            // 3.0,     1.0,        1.0,            2
            // 3.0,     2.0,        2.0,            1

            Assert.True(leadCol.Length == 3);

            // Make sure the values are correct.
            Assert.Equal(2.0, colA[0]);
            Assert.Equal(3.0, colA[1]);
            Assert.Equal(3.0, colA[2]);

            Assert.Equal(1.0, leadCol[0]);
            Assert.Equal(1.0, leadCol[1]);
            Assert.Equal(2.0, leadCol[2]);

            Assert.Equal(1.0, lagCol[0]);
            Assert.Equal(1.0, lagCol[1]);
            Assert.Equal(2.0, lagCol[2]);

            Assert.Equal((UInt32)1, horizonCol[0]);
            Assert.Equal((UInt32)2, horizonCol[1]);
            Assert.Equal((UInt32)1, horizonCol[2]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void Horizon2Test()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new SimpleRWTestData { ColA = 1.0, ColA_Vec = new [] { double.NaN, double.NaN } },
                new SimpleRWTestData { ColA = 2.0, ColA_Vec = new [] { double.NaN, 1.0 } },
                new SimpleRWTestData { ColA = 3.0, ColA_Vec = new [] { 1.0, 2.0 } }
            };

            var annotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "ColA_Vec_Mean_MinWin1_MaxWin1".AsMemory();

            annotations.Add<ReadOnlyMemory<char>>("ColumnNames=ColA_Vec_Mean_MinWin1_MaxWin1", TextDataViewType.Instance, nameValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("ColA", NumberDataViewType.Double);
            schemaBuilder.AddColumn("ColA_Vec", new VectorDataViewType(NumberDataViewType.Double, 1, 2), annotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            // Build the pipeline
            var pipeline = mlContext.Transforms.PivotForecastingData(new string[] { "ColA_Vec" });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var colA = debugView.ColumnView[0].Values;
            var horizonCol = debugView.ColumnView[2].Values;
            var pivotCol = debugView.ColumnView[3].Values;

            // Correct output for:
            // ColA,    ColA_Vec_Mean_MinWin1_MaxWin1,  Horizon
            // 2.0,     1.0,                            1
            // 3.0,     1.0,                            2
            // 3.0,     2.0,                            1

            Assert.True(pivotCol.Length == 3);

            // Make sure the values are correct.
            Assert.Equal(2.0, colA[0]);
            Assert.Equal(3.0, colA[1]);
            Assert.Equal(3.0, colA[2]);

            Assert.Equal(1.0, pivotCol[0]);
            Assert.Equal(1.0, pivotCol[1]);
            Assert.Equal(2.0, pivotCol[2]);

            Assert.Equal((UInt32)1, horizonCol[0]);
            Assert.Equal((UInt32)2, horizonCol[1]);
            Assert.Equal((UInt32)1, horizonCol[2]);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7Fact]
        public void Horizon2LagLeadRWTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new Horizon2LagLeadRWTestData { ColA = 1.0, ColA_RW = new [] { double.NaN, double.NaN }, ColA_LL = new [] { double.NaN, double.NaN, double.NaN, double.NaN } },
                new Horizon2LagLeadRWTestData { ColA = 2.0, ColA_RW = new [] { double.NaN, 1.0 }, ColA_LL = new [] { double.NaN, 1.0, double.NaN, 2.0 } },
                new Horizon2LagLeadRWTestData { ColA = 3.0, ColA_RW = new [] { 1.0, 2.0 }, ColA_LL = new [] { 2.0, double.NaN, 3.0, double.NaN } }
            };

            // Make our annotations for rolling window and lag lead.
            var rolWinAnnotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<ReadOnlyMemory<char>> rolWinNameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "ColA_RW_Mean_MinWin1_MaxWin1".AsMemory();

            rolWinAnnotations.Add<ReadOnlyMemory<char>>("ColumnNames=ColA_RW_Mean_MinWin1_MaxWin1", TextDataViewType.Instance, rolWinNameValueGetter);

            var lagLeadAnnotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<ReadOnlyMemory<char>> lagLeadNameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "ColA_Vec_Lag1,ColA_Vec_Lead1".AsMemory();

            lagLeadAnnotations.Add<ReadOnlyMemory<char>>("ColumnNames=ColA_LL_Lag1,ColA_LL_Lead1", TextDataViewType.Instance, lagLeadNameValueGetter);


            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("ColA", NumberDataViewType.Double);
            schemaBuilder.AddColumn("ColA_RW", new VectorDataViewType(NumberDataViewType.Double, 1, 2), rolWinAnnotations.ToAnnotations());
            schemaBuilder.AddColumn("ColA_LL", new VectorDataViewType(NumberDataViewType.Double, 2, 2), lagLeadAnnotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            // Build the pipeline
            var pipeline = mlContext.Transforms.PivotForecastingData(new string[] { "ColA_RW", "ColA_LL" });
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var colA = debugView.ColumnView[0].Values;
            var horizonCol = debugView.ColumnView[3].Values;
            var rollingWindowCol = debugView.ColumnView[4].Values;
            var lagCol = debugView.ColumnView[5].Values;
            var leadCol = debugView.ColumnView[6].Values;

            // Correct output for:
            // ColA,    ColA_RW_Mean_MinWin1_MaxWin1,   ColA_LL_Lag_1,  ColA_LL_Lead_1, Horizon
            // 2.0,     1.0,                            1.0,            2.0,            1
            // 3.0,     1.0,                            2.0,            3.0,            2

            Assert.True(colA.Length == 2);

            // Make sure the values are correct.
            Assert.Equal(2.0, colA[0]);
            Assert.Equal(3.0, colA[1]);

            Assert.Equal(1.0, rollingWindowCol[0]);
            Assert.Equal(1.0, rollingWindowCol[1]);

            Assert.Equal(1.0, lagCol[0]);
            Assert.Equal(2.0, lagCol[1]);

            Assert.Equal(2.0, leadCol[0]);
            Assert.Equal(3.0, leadCol[1]);

            Assert.Equal((UInt32)1, horizonCol[0]);
            Assert.Equal((UInt32)2, horizonCol[1]);

            TestEstimatorCore(pipeline, data);
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
            Assert.Throws<ArgumentOutOfRangeException>(() => mlContext.Transforms.PivotForecastingData(new string[] { }));
            Assert.Throws<ArgumentNullException>(() => mlContext.Transforms.PivotForecastingData(null));

            Done();
        }

    }
}
