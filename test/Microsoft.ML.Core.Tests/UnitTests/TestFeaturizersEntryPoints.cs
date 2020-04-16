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
            Env.ComponentCatalog.RegisterAssembly(typeof(RollingWindowEstimator).Assembly);
        }
        [NotCentOS7Fact]
        public void RollingWindow_LargeNumberTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumn': ['Grain'],
                            'Column' : [{ 'Name' : 'Target', 'Source' : 'Target' }],
                            'Data' : '$data',
                            'Horizon': 2147483647,
                            'MaxWindowSize' : 4294967294,
                            'MinWindowSize' : 4294967294,
                            'WindowCalculation' : 'Min'
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

            var addedColumn = schema["Target"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 1);
            Assert.True(columnType.Dimensions[1] == 2147483647);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> columnName = default;

            var annotations = addedColumn.Annotations;
            var columnAnnotationName = annotations.Schema.Where(x => x.Name.StartsWith("ColumnNames")).First().Name;

            annotations.GetValue<ReadOnlyMemory<char>>(columnAnnotationName, ref columnName);

            Assert.Equal("Target_Min_MinWin4294967294_MaxWin4294967294", columnName.ToString());
        }

        [NotCentOS7Fact]
        public void RollingWindow_SimpleMeanTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumn': ['Grain'],
                            'Column' : [{ 'Name' : 'Target', 'Source' : 'Target' }],
                            'Data' : '$data',
                            'Horizon': 1,
                            'MaxWindowSize' : 1,
                            'MinWindowSize' : 1,
                            'WindowCalculation' : 'Mean'
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

            var addedColumn = schema["Target"];
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

            Assert.Equal("Target_Mean_MinWin1_MaxWin1", columnName.ToString());

            Done();
        }

        [NotCentOS7Fact]
        public void RollingWindow_SimpleMinTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumn': ['Grain'],
                            'Column' : [{ 'Name' : 'Target', 'Source' : 'Target' }],
                            'Data' : '$data',
                            'Horizon': 1,
                            'MaxWindowSize' : 1,
                            'MinWindowSize' : 1,
                            'WindowCalculation' : 'Min'
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

            var addedColumn = schema["Target"];
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

            Assert.Equal("Target_Min_MinWin1_MaxWin1", columnName.ToString());

            Done();
        }

        [NotCentOS7Fact]
        public void RollingWindow_SimpleMaxTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumn': ['Grain'],
                            'Column' : [{ 'Name' : 'Target', 'Source' : 'Target' }],
                            'Data' : '$data',
                            'Horizon': 1,
                            'MaxWindowSize' : 1,
                            'MinWindowSize' : 1,
                            'WindowCalculation' : 'Max'
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

            var addedColumn = schema["Target"];
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

            Assert.Equal("Target_Max_MinWin1_MaxWin1", columnName.ToString());

            Done();
        }

        [NotCentOS7Fact]
        public void RollingWindow_ComplexTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumn': ['Grain'],
                            'Column' : [{ 'Name' : 'NewCol', 'Source' : 'Target' }],
                            'Data' : '$data',
                            'Horizon': 4,
                            'MaxWindowSize' : 3,
                            'MinWindowSize' : 2,
                            'WindowCalculation' : 'Mean'
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

            var addedColumn = schema["NewCol"];
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

            Assert.Equal("NewCol_Mean_MinWin2_MaxWin3", columnName.ToString());

            Done();
        }

        [NotCentOS7Fact]
        public void ShortDrop_LargeNumberTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ShortDrop',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Data' : '$data',
                            'MinRows' : 4294967294
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

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 4294967294 and we only have 1, should have no rows back.
            Assert.True(rows.Length == 0);

            // Also make sure that the value column was dropped correctly.
            Assert.True(cols[0].Values.Length == 0);
            Assert.True(cols[1].Values.Length == 0);

            Done();
        }
        [NotCentOS7Fact]
        public void ShortDrop_EntryPointTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ShortDrop',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Data' : '$data',
                            'MinRows' : 2
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

            Done();
        }
        [NotCentOS7Fact]
        public void ShortDrop_Drop()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ShortDrop',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Data' : '$data',
                            'MinRows' : 2
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

            Done();
        }
        [NotCentOS7Fact]
        public void ShortDrop_Keep()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1 },
                new { Grain = "one", Target = 1 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ShortDrop',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Data' : '$data',
                            'MinRows' : 2
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

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 2 and we have 2, should have all rows back.
            Assert.True(rows.Length == 2);

            // Also make sure that the value column was dropped correctly.
            Assert.True(cols[0].Values.Length == 2);
            Assert.True(cols[1].Values.Length == 2);

            Done();
        }
    }
}
