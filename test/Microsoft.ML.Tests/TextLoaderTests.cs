// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Newtonsoft.Json.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.EntryPoints.Tests
{
    public sealed class TextLoaderTestPipe : TestDataPipeBase
    {
        public TextLoaderTestPipe(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void TestTextLoaderDataTypes()
        {
            string pathData = DeleteOutputPath("SavePipe", "TextInput.txt");
            File.WriteAllLines(pathData, new string[] {
                string.Format("{0},{1},{2},{3}", sbyte.MinValue, short.MinValue, int.MinValue, long.MinValue),
                string.Format("{0},{1},{2},{3}", sbyte.MaxValue, short.MaxValue, int.MaxValue, long.MaxValue),
                "\"\",\"\",\"\",\"\"",
            });

            var data = TestCore(pathData, true,
                new[] {
                "loader=Text{quote+ col=DvInt1:I1:0 col=DvInt2:I2:1 col=DvInt4:I4:2 col=DvInt8:I8:3 sep=comma}",
                }, logCurs: true);

            using (var cursor = data.GetRowCursorForAllColumns())
            {
                var col1 = cursor.GetGetter<sbyte>(cursor.Schema[0]);
                var col2 = cursor.GetGetter<short>(cursor.Schema[1]);
                var col3 = cursor.GetGetter<int>(cursor.Schema[2]);
                var col4 = cursor.GetGetter<long>(cursor.Schema[3]);

                Assert.True(cursor.MoveNext());

                sbyte[] sByteTargets = new sbyte[] { sbyte.MinValue, sbyte.MaxValue, default };
                short[] shortTargets = new short[] { short.MinValue, short.MaxValue, default };
                int[] intTargets = new int[] { int.MinValue, int.MaxValue, default };
                long[] longTargets = new long[] { long.MinValue, long.MaxValue, default };

                int i = 0;
                for (; i < sByteTargets.Length; i++)
                {
                    sbyte sbyteValue = -1;
                    col1(ref sbyteValue);
                    Assert.Equal(sByteTargets[i], sbyteValue);

                    short shortValue = -1;
                    col2(ref shortValue);
                    Assert.Equal(shortTargets[i], shortValue);

                    int intValue = -1;
                    col3(ref intValue);
                    Assert.Equal(intTargets[i], intValue);

                    long longValue = -1;
                    col4(ref longValue);
                    Assert.Equal(longTargets[i], longValue);

                    if (i < sByteTargets.Length - 1)
                        Assert.True(cursor.MoveNext());
                    else
                        Assert.False(cursor.MoveNext());
                }

                Assert.Equal(i, sByteTargets.Length);
            }
        }

        [Fact]
        public void TestTextLoaderInvalidLongMin()
        {
            string pathData = DeleteOutputPath("SavePipe", "TextInput.txt");
            File.WriteAllLines(pathData, new string[] {
                "-9223372036854775809"

            });

            try
            {
                var data = TestCore(pathData, true,
                    new[] {
                    "loader=Text{col=DvInt8:I8:0 sep=comma}",
                    }, logCurs: true);
            }
            catch (Exception ex)
            {
                Assert.Contains("Could not parse value -9223372036854775809 in line 1, column DvInt8", ex.Message);
                return;
            }

            Assert.True(false, "Test failed.");
        }

        [Fact]
        public void TestTextLoaderInvalidLongMax()
        {
            string pathData = DeleteOutputPath("SavePipe", "TextInput.txt");
            File.WriteAllLines(pathData, new string[] {
                "9223372036854775808"
            });

            try
            {
                var data = TestCore(pathData, true,
                    new[] {
                    "loader=Text{col=DvInt8:I8:0 sep=comma}",
                    }, logCurs: true);
            }
            catch (Exception ex)
            {
                Assert.Contains("Could not parse value 9223372036854775808 in line 1, column DvInt8", ex.Message);
                return;
            }

            Assert.True(false, "Test failed.");
        }
    }

    public class TextLoaderTests : BaseTestClass
    {
        ConsoleEnvironment env;
        public TextLoaderTests(ITestOutputHelper output)
            : base(output)
        {
            env = new ConsoleEnvironment(42).AddStandardComponents();
        }

        [Fact]
        public void ConstructorDoesntThrow()
        {
            var mlContext = new MLContext(seed: 1);

            Assert.NotNull(mlContext.Data.LoadFromTextFile<Input>("fakeFile.txt"));
            Assert.NotNull(mlContext.Data.LoadFromTextFile<Input>("fakeFile.txt", hasHeader: true));
            Assert.NotNull(mlContext.Data.LoadFromTextFile<Input>("fakeFile.txt", hasHeader: false));
            Assert.NotNull(mlContext.Data.LoadFromTextFile<Input>("fakeFile.txt", hasHeader: false, trimWhitespace: false, allowSparse: false));
            Assert.NotNull(mlContext.Data.LoadFromTextFile<Input>("fakeFile.txt", hasHeader: false, allowSparse: false));
            Assert.NotNull(mlContext.Data.LoadFromTextFile<Input>("fakeFile.txt", hasHeader: false, allowQuoting: false));
            Assert.NotNull(mlContext.Data.LoadFromTextFile<InputWithUnderscore>("fakeFile.txt"));
        }

        [Fact]
        public void CanSuccessfullyApplyATransform()
        {
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': true,
                                'AllowSparse': true,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': [{
                                        'Name': 'String1',
                                        'Type': 'TX',
                                        'Source': [{
                                                'Min': 0,
                                                'Max': 0,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'Number1',
                                        'Type': 'R4',
                                        'Source': [{
                                                'Min': 1,
                                                'Max': 1,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }
                                ],
                                'TrimWhitespace': false,
                                'HasHeader': false
                            }
                        },
                        'Outputs': {
                            'Data': '$data'
                        }
                    }
                ]
            }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(env, "fakeFile.txt", false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("data");
            Assert.NotNull(data);
        }

        [Fact]
        public void CanSuccessfullyRetrieveQuotedData()
        {
            string dataPath = GetDataPath("QuotingData.csv");
            string inputGraph = @"
            {  
               'Nodes':[  
                  {  
                     'Name':'Data.TextLoader',
                     'Inputs':{  
                        'InputFile':'$inputFile',
                        'Arguments':{  
                           'UseThreads':true,
                           'HeaderFile':null,
                           'MaxRows':null,
                           'AllowQuoting':true,
                           'AllowSparse':false,
                           'InputSize':null,
                           'Separator':[  
                              ','
                           ],
                           'Column':[  
                              {  
                                 'Name':'ID',
                                 'Type':'R4',
                                 'Source':[  
                                    {  
                                       'Min':0,
                                       'Max':0,
                                       'AutoEnd':false,
                                       'VariableEnd':false,
                                       'AllOther':false,
                                       'ForceVector':false
                                    }
                                 ],
                                 'KeyCount':null
                              },
                              {  
                                 'Name':'Text',
                                 'Type':'TX',
                                 'Source':[  
                                    {  
                                       'Min':1,
                                       'Max':1,
                                       'AutoEnd':false,
                                       'VariableEnd':false,
                                       'AllOther':false,
                                       'ForceVector':false
                                    }
                                 ],
                                 'KeyCount':null
                              }
                           ],
                           'TrimWhitespace':false,
                           'HasHeader':true
                        }
                     },
                     'Outputs':{  
                        'Data':'$data'
                     }
                  }
               ]
            }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("data"); Assert.NotNull(data);

            using (var cursor = data.GetRowCursorForAllColumns())
            {
                var IDGetter = cursor.GetGetter<float>(cursor.Schema[0]);
                var TextGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor.Schema[1]);

                Assert.True(cursor.MoveNext());

                float ID = 0;
                IDGetter(ref ID);
                Assert.Equal(1, ID);

                ReadOnlyMemory<char> Text = new ReadOnlyMemory<char>();
                TextGetter(ref Text);
                Assert.Equal("This text contains comma, within quotes.", Text.ToString());

                Assert.True(cursor.MoveNext());

                ID = 0;
                IDGetter(ref ID);
                Assert.Equal(2, ID);

                Text = new ReadOnlyMemory<char>();
                TextGetter(ref Text);
                Assert.Equal("This text contains extra punctuations and special characters.;*<>?!@#$%^&*()_+=-{}|[]:;'", Text.ToString());

                Assert.True(cursor.MoveNext());

                ID = 0;
                IDGetter(ref ID);
                Assert.Equal(3, ID);

                Text = new ReadOnlyMemory<char>();
                TextGetter(ref Text);
                Assert.Equal("This text has no quotes", Text.ToString());

                Assert.False(cursor.MoveNext());
            }
        }

        [Fact]
        public void CanSuccessfullyRetrieveSparseData()
        {
            string dataPath = GetDataPath("SparseData.txt");
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': false,
                                'AllowSparse': true,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': [{
                                        'Name': 'C1',
                                        'Type': 'R4',
                                        'Source': [{
                                                'Min': 0,
                                                'Max': 0,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'C2',
                                        'Type': 'R4',
                                        'Source': [{
                                                'Min': 1,
                                                'Max': 1,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'C3',
                                        'Type': 'R4',
                                        'Source': [{
                                                'Min': 2,
                                                'Max': 2,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'C4',
                                        'Type': 'R4',
                                        'Source': [{
                                                'Min': 3,
                                                'Max': 3,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'C5',
                                        'Type': 'R4',
                                        'Source': [{
                                                'Min': 4,
                                                'Max': 4,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }
                                ],
                                'TrimWhitespace': false,
                                'HasHeader': true
                            }
                        },
                        'Outputs': {
                            'Data': '$data'
                        }
                    }
                ]
            }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("data");
            Assert.NotNull(data);

            using (var cursor = data.GetRowCursorForAllColumns())
            {
                var getters = new ValueGetter<float>[]{
                        cursor.GetGetter<float>(cursor.Schema[0]),
                        cursor.GetGetter<float>(cursor.Schema[1]),
                        cursor.GetGetter<float>(cursor.Schema[2]),
                        cursor.GetGetter<float>(cursor.Schema[3]),
                        cursor.GetGetter<float>(cursor.Schema[4])
                    };


                Assert.True(cursor.MoveNext());

                float[] targets = new float[] { 1, 2, 3, 4, 5 };
                for (int i = 0; i < getters.Length; i++)
                {
                    float value = 0;
                    getters[i](ref value);
                    Assert.Equal(targets[i], value);
                }

                Assert.True(cursor.MoveNext());

                targets = new float[] { 0, 0, 0, 4, 5 };
                for (int i = 0; i < getters.Length; i++)
                {
                    float value = 0;
                    getters[i](ref value);
                    Assert.Equal(targets[i], value);
                }

                Assert.True(cursor.MoveNext());

                targets = new float[] { 0, 2, 0, 0, 0 };
                for (int i = 0; i < getters.Length; i++)
                {
                    float value = 0;
                    getters[i](ref value);
                    Assert.Equal(targets[i], value);
                }

                Assert.False(cursor.MoveNext());
            }

        }

        [Fact]
        public void CanSuccessfullyTrimSpaces()
        {
            string dataPath = GetDataPath("TrimData.csv");
            string inputGraph = @"{
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': false,
                                'AllowSparse': false,
                                'InputSize': null,
                                'Separator': [
                                    ','
                                ],
                                'Column': [{
                                        'Name': 'ID',
                                        'Type': 'R4',
                                        'Source': [{
                                                'Min': 0,
                                                'Max': 0,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'Text',
                                        'Type': 'TX',
                                        'Source': [{
                                                'Min': 1,
                                                'Max': 1,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }
                                ],
                                'TrimWhitespace': true,
                                'HasHeader': true
                            }
                        },
                        'Outputs': {
                            'Data': '$data'
                        }
                    }
                ]
            }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("data");
            Assert.NotNull(data);

            using (var cursor = data.GetRowCursorForAllColumns())
            {
                var IDGetter = cursor.GetGetter<float>(cursor.Schema[0]);
                var TextGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor.Schema[1]);

                Assert.True(cursor.MoveNext());

                float ID = 0;
                IDGetter(ref ID);
                Assert.Equal(1, ID);

                ReadOnlyMemory<char> Text = new ReadOnlyMemory<char>();
                TextGetter(ref Text);
                Assert.Equal("There is a space at the end", Text.ToString());

                Assert.True(cursor.MoveNext());

                ID = 0;
                IDGetter(ref ID);
                Assert.Equal(2, ID);

                Text = new ReadOnlyMemory<char>();
                TextGetter(ref Text);
                Assert.Equal("There is no space at the end", Text.ToString());

                Assert.False(cursor.MoveNext());
            }
        }

        [Fact]
        public void ThrowsExceptionWithPropertyName()
        {
            var mlContext = new MLContext(seed: 1);
            try
            {
                mlContext.Data.LoadFromTextFile<ModelWithoutColumnAttribute>("fakefile.txt");
            }
            // REVIEW: the issue of different exceptions being thrown is tracked under #2037.
            catch (Xunit.Sdk.TrueException) { }
            catch (NullReferenceException) { };
        }

        public class QuoteInput
        {
            [LoadColumn(0)]
            public float ID;

            [LoadColumn(1)]
            public string Text;
        }

        public class SparseInput
        {
            [LoadColumn(0)]
            public float C1;

            [LoadColumn(1)]
            public float C2;

            [LoadColumn(2)]
            public float C3;

            [LoadColumn(3)]
            public float C4;

            [LoadColumn(4)]
            public float C5;
        }

        public class Input
        {
            [LoadColumn(0)]
            public string String1;

            [LoadColumn(1)]
            public float Number1;
        }

        public class InputWithUnderscore
        {
            [LoadColumn(0)]
            public string String_1;

            [LoadColumn(1)]
            public float Number_1;
        }

        public class ModelWithoutColumnAttribute
        {
            public string String1;
        }

        public class ModelWithColumnNameAttribute
        {
            [LoadColumn(0), ColumnName("Col1")]
            public string String_1;

            [LoadColumn(1)]
            [ColumnName("Col2")]
            public string String_2;

            [LoadColumn(3)]
            public string String_3;
        }
    }

    public class TextLoaderFromModelTests : BaseTestClass
    {
        public TextLoaderFromModelTests(ITestOutputHelper output)
           : base(output)
        {

        }

        public class Iris
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string Type;
        }

        public class IrisStartEnd
        {
            [LoadColumn(start: 0, end: 3), ColumnName("Features")]
            public float Features;

            [LoadColumn(4), ColumnName("Label")]
            public string Type;
        }

        public class IrisColumnIndices
        {
            [LoadColumn(new[] { 0, 2 })]
            public float Features;

            [LoadColumn(4), ColumnName("Label")]
            public string Type;
        }

        [Fact]
        public void LoaderColumnsFromIrisData()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var mlContext = new MLContext();

            var irisFirstRow = new Dictionary<string, float>();
            irisFirstRow["SepalLength"] = 5.1f;
            irisFirstRow["SepalWidth"] = 3.5f;
            irisFirstRow["PetalLength"] = 1.4f;
            irisFirstRow["PetalWidth"] = 0.2f;

            var irisFirstRowValues = irisFirstRow.Values.GetEnumerator();

            // Simple load
            var dataIris = mlContext.Data.CreateTextLoader<Iris>(separatorChar: ',').Load(dataPath);
            var previewIris = dataIris.Preview(1);

            Assert.Equal(5, previewIris.ColumnView.Length);
            Assert.Equal("SepalLength", previewIris.Schema[0].Name);
            Assert.Equal(NumberDataViewType.Single, previewIris.Schema[0].Type);
            int index = 0;
            foreach (var entry in irisFirstRow)
            {
                Assert.Equal(entry.Key, previewIris.RowView[0].Values[index].Key);
                Assert.Equal(entry.Value, previewIris.RowView[0].Values[index++].Value);
            }
            Assert.Equal("Type", previewIris.RowView[0].Values[index].Key);
            Assert.Equal("Iris-setosa", previewIris.RowView[0].Values[index].Value.ToString());

            // Load with start and end indexes
            var dataIrisStartEnd = mlContext.Data.CreateTextLoader<IrisStartEnd>(separatorChar: ',').Load(dataPath);
            var previewIrisStartEnd = dataIrisStartEnd.Preview(1);

            Assert.Equal(2, previewIrisStartEnd.ColumnView.Length);
            Assert.Equal("Features", previewIrisStartEnd.RowView[0].Values[0].Key);
            var featureValue = (VBuffer<float>)previewIrisStartEnd.RowView[0].Values[0].Value;
            Assert.True(featureValue.IsDense);
            Assert.Equal(4, featureValue.Length);

            irisFirstRowValues = irisFirstRow.Values.GetEnumerator();
            foreach (var val in featureValue.GetValues())
            {
                irisFirstRowValues.MoveNext();
                Assert.Equal(irisFirstRowValues.Current, val);
            }

            // load setting the distinct columns. Loading column 0 and 2
            var dataIrisColumnIndices = mlContext.Data.CreateTextLoader<IrisColumnIndices>(separatorChar: ',').Load(dataPath);
            var previewIrisColumnIndices = dataIrisColumnIndices.Preview(1);

            Assert.Equal(2, previewIrisColumnIndices.ColumnView.Length);
            featureValue = (VBuffer<float>)previewIrisColumnIndices.RowView[0].Values[0].Value;
            Assert.True(featureValue.IsDense);
            Assert.Equal(2, featureValue.Length);
            var vals4 = featureValue.GetValues();

            irisFirstRowValues = irisFirstRow.Values.GetEnumerator();
            irisFirstRowValues.MoveNext();
            Assert.Equal(vals4[0], irisFirstRowValues.Current);
            irisFirstRowValues.MoveNext(); irisFirstRowValues.MoveNext(); // skip col 1
            Assert.Equal(vals4[1], irisFirstRowValues.Current);
        }

        [Fact]
        public void TestTextLoaderKeyTypeBackCompat()
        {
            // Model generated with the following command on a version of the code previous to the KeyType change that removed Min and Contiguous:
            // Train data=...\breast-cancer.txt loader =TextLoader{col=Label:R4:0 col=Features:R4:1-9 col=key:U4[0-*]:3} tr=LogisticRegression {} out=model.zip
            var mlContext = new MLContext();
            string textLoaderModelPath = GetDataPath("backcompat/textloader-with-key-model.zip");
            string breastCancerPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            using (FileStream modelfs = File.OpenRead(textLoaderModelPath))
            using (var rep = RepositoryReader.Open(modelfs, mlContext))
            {
                var result = ModelFileUtils.LoadLoader(mlContext, rep, new MultiFileSource(breastCancerPath), false);
                Assert.True(result.Schema.TryGetColumnIndex("key", out int featureIdx));
                Assert.True(result.Schema[featureIdx].Type is KeyType keyType && keyType.Count == typeof(uint).ToMaxInt());
            }
        }
    }
}
