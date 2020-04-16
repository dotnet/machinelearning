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
            Env.ComponentCatalog.RegisterAssembly(typeof(LagLeadOperatorEstimator).Assembly);
        }

        [NotCentOS7Fact]
        public void LagLead_SimpleSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.LagLeadOperator',
                    'Inputs': {
                            'GrainColumns': ['GrainA'],
                            'Column' : [{ 'Name' : 'ColA', 'Source' : 'ColA' }],
                            'Data' : '$data',
                            'Horizon': 1,
                            'Offsets' : [-1]
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
        public void LagLead_LargeNumberSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.LagLeadOperator',
                    'Inputs': {
                            'GrainColumns': ['GrainA'],
                            'Column' : [{ 'Name' : 'ColA', 'Source' : 'ColA' }],
                            'Data' : '$data',
                            'Horizon': 500000,
                            'Offsets' : [500000]
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

            Done();
        }
        [NotCentOS7Fact]
        public void LagLead_ComplexSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.LagLeadOperator',
                    'Inputs': {
                            'GrainColumns': ['GrainA'],
                            'Column' : [{ 'Name' : 'ColA_New', 'Source' : 'ColA' }],
                            'Data' : '$data',
                            'Horizon': 3,
                            'Offsets' : [-1, 1, -2, 2]
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
        public void LagLead_LagTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.LagLeadOperator',
                    'Inputs': {
                            'GrainColumns': ['GrainA'],
                            'Column' : [{ 'Name' : 'ColA', 'Source' : 'ColA' }],
                            'Data' : '$data',
                            'Horizon': 2,
                            'Offsets' : [-2, -1]
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

            Done();
        }

        [NotCentOS7Fact]
        public void LagLead_Lead1Test()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0, ColB = 1 },
                new { GrainA = "Grain", ColA = 2.0, ColB = 2 },
                new { GrainA = "Grain", ColA = 3.0, ColB = 3 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.LagLeadOperator',
                    'Inputs': {
                            'GrainColumns': ['GrainA'],
                            'Column' : [{ 'Name' : 'ColA', 'Source' : 'ColA' }],
                            'Data' : '$data',
                            'Horizon': 1,
                            'Offsets' : [1]
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

            Done();
        }

        [NotCentOS7Fact]
        public void LagLead_Lead2Test()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0, ColB = 1 },
                new { GrainA = "Grain", ColA = 2.0, ColB = 2 },
                new { GrainA = "Grain", ColA = 3.0, ColB = 3 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.LagLeadOperator',
                    'Inputs': {
                            'GrainColumns': ['GrainA'],
                            'Column' : [{ 'Name' : 'ColA', 'Source' : 'ColA' }],
                            'Data' : '$data',
                            'Horizon': 1,
                            'Offsets' : [2]
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

            Done();
        }

        [NotCentOS7Fact]
        public void LagLead_ComplexLagLeadTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0, ColB = 1 },
                new { GrainA = "Grain", ColA = 2.0, ColB = 2 },
                new { GrainA = "Grain", ColA = 3.0, ColB = 3 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            var offsets = new long[] { -2, -1, 1, 2 };

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.LagLeadOperator',
                    'Inputs': {
                            'GrainColumns': ['GrainA'],
                            'Column' : [{ 'Name' : 'ColA', 'Source' : 'ColA' }],
                            'Data' : '$data',
                            'Horizon': 2,
                            'Offsets' : [-2, -1, 1, 2]
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

            var expectedOutput = new[] {
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

            Done();
        }
    }
}
