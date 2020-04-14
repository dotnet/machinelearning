// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Core.Tests.UnitTests;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Ensemble;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Transforms.TimeSeries;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;
using Xunit.Abstractions;

using Microsoft.ML.RunTests;

using Microsoft.ML.Tests.Transformers;
namespace Microsoft.ML.RunTests
{
    public class TestFeaturizersEntryPoints : CoreBaseTestClass
    {
        public TestFeaturizersEntryPoints(ITestOutputHelper output) : base(output)
        {
            Env.ComponentCatalog.RegisterAssembly(typeof(ForecastingPivotFeaturizerEstimator).Assembly);
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

        [Fact]
        public void ForecastingPivot_SimpleRWTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new SimpleRWTestData { ColA = 1.0, ColA_Vec = new [] { 1.0, 2.0 } }
            };

            var annotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<UInt32> minWindowSizeValueGetter = (ref UInt32 dst) => dst = 1;
            ValueGetter<UInt32> maxWindowSizeValueGetter = (ref UInt32 dst) => dst = 1;
            ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "RollingWindow".AsMemory();
            ValueGetter<ReadOnlyMemory<char>> calculationValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "Mean".AsMemory();

            annotations.Add<UInt32>("MinWindowSize=1", NumberDataViewType.UInt32, minWindowSizeValueGetter);
            annotations.Add<UInt32>("MaxWindowSize=1", NumberDataViewType.UInt32, maxWindowSizeValueGetter);
            annotations.Add<ReadOnlyMemory<char>>("FeaturizerName=RollingWindow", TextDataViewType.Instance, nameValueGetter);
            annotations.Add<ReadOnlyMemory<char>>("Calculation=Mean", TextDataViewType.Instance, calculationValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("ColA", NumberDataViewType.Double);
            schemaBuilder.AddColumn("ColA_Vec", new VectorDataViewType(NumberDataViewType.Double, 1, 2), annotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ForecastingPivot',
                    'Inputs': {
                            'ColumnsToPivot': ['ColA_Vec'],
                            'Data' : '$data'
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            var addedColumn = schema["ColA_Vec_Mean_MinWin1_MaxWin1"];
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

        [Fact]
        public void ForecastingPivot_SimpleLagLeadTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new SimpleLagLeadTestData { ColA = 1.0, ColA_Vec = new [] { double.NaN, double.NaN, double.NaN, double.NaN } },
                new SimpleLagLeadTestData { ColA = 2.0, ColA_Vec = new [] { double.NaN, 1.0, double.NaN, 1.0 } },
                new SimpleLagLeadTestData { ColA = 3.0, ColA_Vec = new [] { 1.0, 2.0, 1.0, 2.0 } }
            };
            var annotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<VBuffer<long>> offsetValueGetter = (ref VBuffer<long> dst) => dst = new VBuffer<long>(2, new long[] { -1, 1 });
            ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "LagLead".AsMemory();

            annotations.Add<VBuffer<long>>("Offsets=-1,1", new VectorDataViewType(NumberDataViewType.Int64, 2), offsetValueGetter);
            annotations.Add<ReadOnlyMemory<char>>("FeaturizerName=LagLead", TextDataViewType.Instance, nameValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("ColA", NumberDataViewType.Double);
            schemaBuilder.AddColumn("ColA_Vec", new VectorDataViewType(NumberDataViewType.Double, 2, 2), annotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ForecastingPivot',
                    'Inputs': {
                            'ColumnsToPivot': ['ColA_Vec'],
                            'Data' : '$data'
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
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

            Done();
        }

        [Fact]
        public void ForecastingPivot_Horizon2RWTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new SimpleRWTestData { ColA = 1.0, ColA_Vec = new [] { double.NaN, double.NaN } },
                new SimpleRWTestData { ColA = 2.0, ColA_Vec = new [] { double.NaN, 1.0 } },
                new SimpleRWTestData { ColA = 3.0, ColA_Vec = new [] { 1.0, 2.0 } }
            };
            var annotations = new DataViewSchema.Annotations.Builder();
            ValueGetter<UInt32> minWindowSizeValueGetter = (ref UInt32 dst) => dst = 1;
            ValueGetter<UInt32> maxWindowSizeValueGetter = (ref UInt32 dst) => dst = 1;
            ValueGetter<ReadOnlyMemory<char>> nameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "RollingWindow".AsMemory();
            ValueGetter<ReadOnlyMemory<char>> calculationValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "Mean".AsMemory();

            annotations.Add<UInt32>("MinWindowSize=1", NumberDataViewType.UInt32, minWindowSizeValueGetter);
            annotations.Add<UInt32>("MaxWindowSize=1", NumberDataViewType.UInt32, maxWindowSizeValueGetter);
            annotations.Add<ReadOnlyMemory<char>>("FeaturizerName=RollingWindow", TextDataViewType.Instance, nameValueGetter);
            annotations.Add<ReadOnlyMemory<char>>("Calculation=Mean", TextDataViewType.Instance, calculationValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("ColA", NumberDataViewType.Double);
            schemaBuilder.AddColumn("ColA_Vec", new VectorDataViewType(NumberDataViewType.Double, 1, 2), annotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ForecastingPivot',
                    'Inputs': {
                            'ColumnsToPivot': ['ColA_Vec'],
                            'Data' : '$data'
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var colA = debugView.ColumnView[0].Values;
            var horizonCol = debugView.ColumnView[2].Values;
            var pivotCol = debugView.ColumnView[3].Values;

            // Correct output for:
            // ColA,    ColA_RW_Mean_MinWin1_MaxWin1,  Horizon
            // 2.0,     1.0,                        1
            // 3.0,     1.0,                        2
            // 3.0,     2.0,                        1

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

            Done();
        }

        [Fact]
        public void ForecastingPivot_Horizon2LagLeadTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new Horizon2LagLeadRWTestData { ColA = 1.0, ColA_RW = new [] { double.NaN, double.NaN }, ColA_LL = new [] { double.NaN, double.NaN, double.NaN, double.NaN } },
                new Horizon2LagLeadRWTestData { ColA = 2.0, ColA_RW = new [] { double.NaN, 1.0 }, ColA_LL = new [] { double.NaN, 1.0, double.NaN, 2.0 } },
                new Horizon2LagLeadRWTestData { ColA = 3.0, ColA_RW = new [] { 1.0, 2.0 }, ColA_LL = new [] { 2.0, double.NaN, 3.0, double.NaN } }
            };
            // Set up the value getters for the annotations.
            ValueGetter<VBuffer<long>> offsetValueGetter = (ref VBuffer<long> dst) => dst = new VBuffer<long>(2, new long[] { -1, 1 });
            ValueGetter<ReadOnlyMemory<char>> lagNameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "LagLead".AsMemory();
            ValueGetter<UInt32> minWindowSizeValueGetter = (ref UInt32 dst) => dst = 1;
            ValueGetter<UInt32> maxWindowSizeValueGetter = (ref UInt32 dst) => dst = 1;
            ValueGetter<ReadOnlyMemory<char>> rollWinNameValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "RollingWindow".AsMemory();
            ValueGetter<ReadOnlyMemory<char>> calculationValueGetter = (ref ReadOnlyMemory<char> dst) => dst = "Mean".AsMemory();

            // Make our annotations for rolling window and lag lead.
            var rolWinAnnotations = new DataViewSchema.Annotations.Builder();
            rolWinAnnotations.Add<UInt32>("MinWindowSize=1", NumberDataViewType.UInt32, minWindowSizeValueGetter);
            rolWinAnnotations.Add<UInt32>("MaxWindowSize=1", NumberDataViewType.UInt32, maxWindowSizeValueGetter);
            rolWinAnnotations.Add<ReadOnlyMemory<char>>("FeaturizerName=RollingWindow", TextDataViewType.Instance, rollWinNameValueGetter);
            rolWinAnnotations.Add<ReadOnlyMemory<char>>("Calculation=Mean", TextDataViewType.Instance, calculationValueGetter);

            var lagLeadAnnotations = new DataViewSchema.Annotations.Builder();
            lagLeadAnnotations.Add<VBuffer<long>>("Offsets=-1,1", new VectorDataViewType(NumberDataViewType.Int64, 2), offsetValueGetter);
            lagLeadAnnotations.Add<ReadOnlyMemory<char>>("FeaturizerName=LagLead", TextDataViewType.Instance, lagNameValueGetter);

            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("ColA", NumberDataViewType.Double);
            schemaBuilder.AddColumn("ColA_RW", new VectorDataViewType(NumberDataViewType.Double, 1, 2), rolWinAnnotations.ToAnnotations());
            schemaBuilder.AddColumn("ColA_LL", new VectorDataViewType(NumberDataViewType.Double, 2, 2), lagLeadAnnotations.ToAnnotations());

            var data = mlContext.Data.LoadFromEnumerable(dataList, schemaBuilder.ToSchema());

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ForecastingPivot',
                    'Inputs': {
                            'ColumnsToPivot': ['ColA_RW', 'ColA_LL'],
                            'Data' : '$data'
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var colA = debugView.ColumnView[0].Values;
            var horizonCol = debugView.ColumnView[3].Values;
            var rollingWindowCol = debugView.ColumnView[4].Values;
            var lagCol = debugView.ColumnView[5].Values;
            var leadCol = debugView.ColumnView[6].Values;

            // Correct output for:
            // ColA,    ColA_RW_Mean_MinWin1_MaxWin1,  ColA_Lag_1, ColA_Lead_1,    Horizon
            // 2.0,     1.0,                        1.0,        2.0,            1
            // 3.0,     1.0,                        2.0,        3.0,            2

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

            Done();
        }
    }
}
