﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon;
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

            Done();
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
        ConsoleEnvironment _env;
        public TextLoaderTests(ITestOutputHelper output)
            : base(output)
        {
            _env = new ConsoleEnvironment(42).AddStandardComponents();
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
            var runner = new GraphRunner(_env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(_env, "fakeFile.txt", false, false);
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
            var runner = new GraphRunner(_env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(_env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("data"); Assert.NotNull(data);

            using (var cursor = data.GetRowCursorForAllColumns())
            {
                var idGetter = cursor.GetGetter<float>(cursor.Schema[0]);
                var textGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor.Schema[1]);

                Assert.True(cursor.MoveNext());

                float id = 0;
                idGetter(ref id);
                Assert.Equal(1, id);

                ReadOnlyMemory<char> text = new ReadOnlyMemory<char>();
                textGetter(ref text);
                Assert.Equal("This text contains comma, within quotes.", text.ToString());

                Assert.True(cursor.MoveNext());

                id = 0;
                idGetter(ref id);
                Assert.Equal(2, id);

                text = new ReadOnlyMemory<char>();
                textGetter(ref text);
                Assert.Equal("This text contains extra punctuations and special characters.;*<>?!@#$%^&*()_+=-{}|[]:;'", text.ToString());

                Assert.True(cursor.MoveNext());

                id = 0;
                idGetter(ref id);
                Assert.Equal(3, id);

                text = new ReadOnlyMemory<char>();
                textGetter(ref text);
                Assert.Equal("This text has no quotes", text.ToString());

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
            var runner = new GraphRunner(_env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(_env, dataPath, false, false);
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
            var runner = new GraphRunner(_env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(_env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("data");
            Assert.NotNull(data);

            using (var cursor = data.GetRowCursorForAllColumns())
            {
                var idGetter = cursor.GetGetter<float>(cursor.Schema[0]);
                var textGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor.Schema[1]);

                Assert.True(cursor.MoveNext());

                float id = 0;
                idGetter(ref id);
                Assert.Equal(1, id);

                ReadOnlyMemory<char> text = new ReadOnlyMemory<char>();
                textGetter(ref text);
                Assert.Equal("There is a space at the end", text.ToString());

                Assert.True(cursor.MoveNext());

                id = 0;
                idGetter(ref id);
                Assert.Equal(2, id);

                text = new ReadOnlyMemory<char>();
                textGetter(ref text);
                Assert.Equal("There is no space at the end", text.ToString());

                Assert.False(cursor.MoveNext());
            }
        }

        [Fact]
        public void ThrowsExceptionWithMissingFile()
        {
            var mlContext = new MLContext(seed: 1);
            var ex = Assert.Throws<ArgumentOutOfRangeException>(() => mlContext.Data.LoadFromTextFile<ModelWithoutColumnAttribute>("fakefile.txt"));
            Assert.StartsWith("File does not exist at path: fakefile.txt", ex.Message);
        }

        [Fact]
        public void ParseSchemaFromTextFile()
        {
            var mlContext = new MLContext(seed: 1);
            var fileName = GetDataPath(TestDatasets.adult.trainFilename);
            var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options(), new MultiFileSource(fileName));
            var data = loader.Load(new MultiFileSource(fileName));
            Assert.NotNull(data.Schema.GetColumnOrNull("Label"));
            Assert.NotNull(data.Schema.GetColumnOrNull("Workclass"));
            Assert.NotNull(data.Schema.GetColumnOrNull("Categories"));
            Assert.NotNull(data.Schema.GetColumnOrNull("NumericFeatures"));
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

    public class TextLoaderFromModelTests : BaseTestBaseline
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

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void LoaderColumnsFromIrisData(bool useOptionsObject)
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var mlContext = new MLContext(1);

            var irisFirstRow = new Dictionary<string, float>();
            irisFirstRow["SepalLength"] = 5.1f;
            irisFirstRow["SepalWidth"] = 3.5f;
            irisFirstRow["PetalLength"] = 1.4f;
            irisFirstRow["PetalWidth"] = 0.2f;

            var irisFirstRowValues = irisFirstRow.Values.GetEnumerator();

            // Simple load
            IDataView dataIris;
            if (useOptionsObject)
                dataIris = mlContext.Data.CreateTextLoader<Iris>(new TextLoader.Options() { Separator = ",", AllowQuoting = false }).Load(dataPath);
            else
                dataIris = mlContext.Data.CreateTextLoader<Iris>(separatorChar: ',').Load(dataPath);

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
            IDataView dataIrisStartEnd;
            if (useOptionsObject)
                dataIrisStartEnd = mlContext.Data.CreateTextLoader<IrisStartEnd>(new TextLoader.Options() { Separator = ",", AllowQuoting = false }).Load(dataPath);
            else
                dataIrisStartEnd = mlContext.Data.CreateTextLoader<IrisStartEnd>(separatorChar: ',').Load(dataPath);

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
            IDataView dataIrisColumnIndices;
            if (useOptionsObject)
                dataIrisColumnIndices = mlContext.Data.CreateTextLoader<IrisColumnIndices>(new TextLoader.Options() { Separator = ",", AllowQuoting = false }).Load(dataPath);
            else
                dataIrisColumnIndices = mlContext.Data.CreateTextLoader<IrisColumnIndices>(separatorChar: ',').Load(dataPath);

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
            var mlContext = new MLContext(1);
            string textLoaderModelPath = GetDataPath("backcompat/textloader-with-key-model.zip");
            string breastCancerPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            using (FileStream modelfs = File.OpenRead(textLoaderModelPath))
            using (var rep = RepositoryReader.Open(modelfs, mlContext))
            {
                var result = ModelFileUtils.LoadLoader(mlContext, rep, new MultiFileSource(breastCancerPath), false);
                Assert.True(result.Schema.TryGetColumnIndex("key", out int featureIdx));
                Assert.True(result.Schema[featureIdx].Type is KeyDataViewType keyType && keyType.Count == typeof(uint).ToMaxInt());
            }
        }

        [Fact]
        public void TestTextLoaderBackCompat_VerWritt_0x0001000C()
        {
            // Checks backward compatibility with a text loader created with "verWrittenCur: 0x0001000C"
            // Model generated with:
            // loader=text{header+ col=SepalLength:Num:0 col=SepalWidth:Num:1 col=PetalLength:Num:2 col=PetalWidth:Num:2 col=Cat:TX:1-8 col=Num:9-14 col=Type:TX:4}
            var mlContext = new MLContext(1);
            string textLoaderModelPath = GetDataPath("backcompat/textloader_VerWritt_0x0001000C.zip");
            string irisPath = GetDataPath(TestDatasets.irisData.trainFilename);

            IDataView iris;
            using (FileStream modelfs = File.OpenRead(textLoaderModelPath))
            using (var rep = RepositoryReader.Open(modelfs, mlContext))
            {
                iris = ModelFileUtils.LoadLoader(mlContext, rep, new MultiFileSource(irisPath), false);
            }

            var previewIris = iris.Preview(1);
            var irisFirstRow = new Dictionary<string, float>();
            irisFirstRow["SepalLength"] = 5.1f;
            irisFirstRow["SepalWidth"] = 3.5f;
            irisFirstRow["PetalLength"] = 1.4f;
            irisFirstRow["PetalWidth"] = 0.2f;

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
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestCommaAsDecimalMarker(bool useCsvVersion)
        {
            // When userCsvVersion == false:
            // Datasets iris.txt and iris-decimal-marker-as-comma.txt are the exact same, except for their
            // decimal markers. Decimal marker in iris.txt is '.', and ',' in iris-decimal-marker-as-comma.txt.

            // When userCsvVersion == true:
            // Check to confirm TextLoader can read data from a CSV file where the separator is ',', decimals
            // are enclosed with quotes, and with the decimal marker being ','.

            // Do these checks with both float and double as types of features being read, to test decimal marker
            // recognition with both doubles and floats.
            TestCommaAsDecimalMarkerHelper<float>(useCsvVersion);
            TestCommaAsDecimalMarkerHelper<double>(useCsvVersion);
        }
        
        private void TestCommaAsDecimalMarkerHelper<T>(bool useCsvVersion)
        {
            // Datasets iris.txt and iris-decimal-marker-as-comma.txt are the exact same, except for their
            // decimal markers. Decimal marker in iris.txt is '.', and ',' in iris-decimal-marker-as-comma.txt.
            // Datasets iris.txt and iris-decimal-marker-as-comma.csv have the exact same data, however the .csv
            // version has ',' as decimal marker and separator, and feature values are enclosed with quotes.
            // T varies as either float or double, so that decimal markers can be tested for both floating
            // point value types.
            var mlContext = new MLContext(seed: 1);

            // Read dataset with period as decimal marker.
            string dataPathDecimalMarkerPeriod = GetDataPath("iris.txt");
            var readerDecimalMarkerPeriod = new TextLoader(mlContext, new TextLoader.Options()
            {
                Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.UInt32, 0),
                            new TextLoader.Column("Features", typeof(T) == typeof(double) ? DataKind.Double : DataKind.Single, new [] { new TextLoader.Range(1, 4) }),
                        },
                DecimalMarker = '.'
            });
            var textDataDecimalMarkerPeriod = readerDecimalMarkerPeriod.Load(GetDataPath(dataPathDecimalMarkerPeriod));

            // Load values from iris.txt
            DataViewSchema columnsPeriod = textDataDecimalMarkerPeriod.Schema;
            using DataViewRowCursor cursorPeriod = textDataDecimalMarkerPeriod.GetRowCursor(columnsPeriod);
            UInt32 labelPeriod = default;
            ValueGetter<UInt32> labelDelegatePeriod = cursorPeriod.GetGetter<UInt32>(columnsPeriod[0]);
            VBuffer<T> featuresPeriod = default;
            ValueGetter<VBuffer<T>> featuresDelegatePeriod = cursorPeriod.GetGetter<VBuffer<T>>(columnsPeriod[1]);

            // Iterate over each row and save labels and features to array for future comparison
            int count = 0;
            UInt32[] labels = new uint[150];
            T[][] features = new T[150][];
            while (cursorPeriod.MoveNext())
            {
                //Get values from respective columns
                labelDelegatePeriod(ref labelPeriod);
                featuresDelegatePeriod(ref featuresPeriod);
                labels[count] = labelPeriod;
                features[count] = featuresPeriod.GetValues().ToArray();
                count++;
            }

            // Read dataset with comma as decimal marker.
            // Dataset is either the .csv version or the .txt version.
            string dataPathDecimalMarkerComma;
            TextLoader.Options options = new TextLoader.Options()
            {
                Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.UInt32, 0),
                            new TextLoader.Column("Features", typeof(T) == typeof(double) ? DataKind.Double : DataKind.Single, new [] { new TextLoader.Range(1, 4) })
                        },
            };
            // Set TextLoader.Options for the .csv or .txt cases.
            if (useCsvVersion)
            {
                dataPathDecimalMarkerComma = GetDataPath("iris-decimal-marker-as-comma.csv");
                options.DecimalMarker = ',';
                options.Separator = ",";
                options.AllowQuoting = true;
                options.HasHeader = true;
            }
            else
            {
                dataPathDecimalMarkerComma = GetDataPath("iris-decimal-marker-as-comma.txt");
                options.DecimalMarker = ',';
            }
            var readerDecimalMarkerComma = new TextLoader(mlContext, options);
            var textDataDecimalMarkerComma = readerDecimalMarkerComma.Load(GetDataPath(dataPathDecimalMarkerComma));

            // Load values from dataset with comma as decimal marker
            DataViewSchema columnsComma = textDataDecimalMarkerComma.Schema;
            using DataViewRowCursor cursorComma = textDataDecimalMarkerComma.GetRowCursor(columnsComma);
            UInt32 labelComma = default;
            ValueGetter<UInt32> labelDelegateComma = cursorComma.GetGetter<UInt32>(columnsComma[0]);
            VBuffer<T> featuresComma = default;
            ValueGetter<VBuffer<T>> featuresDelegateComma = cursorComma.GetGetter<VBuffer<T>>(columnsComma[1]);

            // Check values from dataset with comma as decimal marker match those in iris.txt (period decimal marker)
            count = 0;
            while (cursorComma.MoveNext())
            {
                //Get values from respective columns
                labelDelegateComma(ref labelComma);
                featuresDelegateComma(ref featuresComma);
                Assert.Equal(labels[count], labelComma);
                Assert.Equal(features[count], featuresComma.GetValues().ToArray());
                count++;
            }
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestWrongDecimalMarkerInputs(bool useCommaAsDecimalMarker)
        {
            // When DecimalMarker does not match the actual decimal marker used in the dataset,
            // we obtain values of NaN. Check that the values are indeed NaN in this case.
            // Do this check for both cases where decimal markers in the dataset are '.' and ','.
            var mlContext = new MLContext(seed: 1);

            // Try reading a dataset where '.' is the actual decimal marker, but DecimalMarker = ',',
            // and vice versa.
            string dataPath;
            TextLoader.Options options = new TextLoader.Options()
            {
                Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.UInt32, 0),
                            new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 4) })
                        },
            };
            if (useCommaAsDecimalMarker)
            {
                dataPath = GetDataPath("iris.txt"); //  Has '.' as decimal marker inside dataset
                options.DecimalMarker = ','; // Choose wrong decimal marker on purpose
            }
            else
            {
                dataPath = GetDataPath("iris-decimal-marker-as-comma.txt"); // Has ',' as decimal marker inside dataset
                options.DecimalMarker = '.'; // Choose wrong decimal marker on purpose
            }
            var reader = new TextLoader(mlContext, options);
            var textData = reader.Load(GetDataPath(dataPath));

            // Check that the features being loaded are NaN.
            DataViewSchema columns = textData.Schema;
            using DataViewRowCursor cursor = textData.GetRowCursor(columns);
            VBuffer<Single> featuresPeriod = default;
            ValueGetter<VBuffer<Single>> featuresDelegatePeriod = cursor.GetGetter<VBuffer<Single>>(columns[1]);
            
            // Iterate over each row and check that feature values are NaN.
            while (cursor.MoveNext())
            {
                featuresDelegatePeriod.Invoke(ref featuresPeriod);
                foreach(float feature in featuresPeriod.GetValues().ToArray())
                    Assert.Equal(feature, Single.NaN);
            }
        }

        private class IrisNoFields
        {
        }

        private class IrisPrivateFields
        {
            [LoadColumn(0)]
            private float _sepalLength;

            [LoadColumn(1)]
            private float SepalWidth { get; }

            public float GetSepalLenght()
                => _sepalLength;

            public void SetSepalLength(float sepalLength)
            {
                _sepalLength = sepalLength;
            }
        }
        private class IrisPublicGetProperties
        {
            [LoadColumn(0)]
            public float SepalLength { get; }

            [LoadColumn(1)]
            public float SepalWidth { get; }
        }

        private class IrisPublicFields
        {
            public IrisPublicFields(float sepalLength, float sepalWidth)
            {
                SepalLength = sepalLength;
                SepalWidth = sepalWidth;
            }

            [LoadColumn(0)]
            public readonly float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;
        }

        private class IrisPublicProperties
        {
            [LoadColumn(0)]
            public float SepalLength { get; set; }

            [LoadColumn(1)]
            public float SepalWidth { get; set; }
        }

        [Fact]
        public void TestTextLoaderNoFields()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var mlContext = new MLContext(1);

            // Class with get property only.
            var dataIris = mlContext.Data.CreateTextLoader<IrisPublicGetProperties>(separatorChar: ',').Load(dataPath);
            var oneIrisData = mlContext.Data.CreateEnumerable<IrisPublicProperties>(dataIris, false).First();
            Assert.True(oneIrisData.SepalLength != 0 && oneIrisData.SepalWidth != 0);

            // Class with read only fields.
            dataIris = mlContext.Data.CreateTextLoader<IrisPublicFields>(separatorChar: ',').Load(dataPath);
            oneIrisData = mlContext.Data.CreateEnumerable<IrisPublicProperties>(dataIris, false).First();
            Assert.True(oneIrisData.SepalLength != 0 && oneIrisData.SepalWidth != 0);

            // Class with no fields.
            try
            {
                dataIris = mlContext.Data.CreateTextLoader<IrisNoFields>(separatorChar: ',').Load(dataPath);
                Assert.False(true);
            }
            catch (Exception ex)
            {
                Assert.StartsWith("Should define at least one public, readable field or property in TInput.", ex.Message);
            }

            // Class with no public readable fields.
            try
            {
                dataIris = mlContext.Data.CreateTextLoader<IrisPrivateFields>(separatorChar: ',').Load(dataPath);
                Assert.False(true);
            }
            catch (Exception ex)
            {
                Assert.StartsWith("Should define at least one public, readable field or property in TInput.", ex.Message);
            }
        }

        public class BreastCancerInputModelWithKeyType
        {
            [LoadColumn(0)]
            public bool IsMalignant { get; set; }

            [LoadColumn(1), KeyType(10)]
            public uint Thickness { get; set; }
        }

        public class BreastCancerInputModelWithoutKeyType
        {
            [LoadColumn(0)]
            public bool IsMalignant { get; set; }

            [LoadColumn(1)]
            public uint Thickness { get; set; }
        }

        [Fact]
        public void TestLoadTextWithKeyTypeAttribute()
        {
            ulong expectedCount = 10;

            var mlContext = new MLContext(seed: 1);
            string breastCancerPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            var data = mlContext.Data.CreateTextLoader<BreastCancerInputModelWithKeyType>(separatorChar: ',').Load(breastCancerPath);

            Assert.Equal(expectedCount, data.Schema[1].Type.GetKeyCount());
        }

        [Fact]
        public void TestLoadTextWithoutKeyTypeAttribute()
        {
            ulong expectedCount = 0;
            var mlContext = new MLContext(seed: 1);
            string breastCancerPath = GetDataPath(TestDatasets.breastCancer.trainFilename);

            var data = mlContext.Data.CreateTextLoader<BreastCancerInputModelWithoutKeyType>(separatorChar: ',').Load(breastCancerPath);

            Assert.Equal(expectedCount, data.Schema[1].Type.GetKeyCount());
        }

        [Theory]
        [InlineData(true, false)]
        [InlineData(false, false)]
        [InlineData(true, true)]
        [InlineData(false, true)]
        public void TestLoadTextWithEscapedNewLinesAndEscapeChar(bool useSaved, bool useCustomEscapeChar)
        {
            var mlContext = new MLContext(seed: 1);
            string dataPath;

            if (!useCustomEscapeChar)
                dataPath = GetDataPath("multiline.csv");
            else
                dataPath = GetDataPath("multiline-escapechar.csv");

            var baselinePath = GetBaselinePath("TextLoader", "multiline.csv");
            var options = new TextLoader.Options()
            {
                HasHeader = true,
                Separator = ",",
                AllowQuoting = true,
                ReadMultilines = true,
                EscapeChar = useCustomEscapeChar ? '\\' : TextLoader.Defaults.EscapeChar,
                Columns = new[]
                {
                    new TextLoader.Column("id", DataKind.Int32, 0),
                    new TextLoader.Column("description", DataKind.String, 1),
                    new TextLoader.Column("animal", DataKind.String, 2),
                },
            };

            var data = mlContext.Data.LoadFromTextFile(dataPath, options);
            if (useSaved)
            {
                // Check that loading the data view from a text file,
                // and then saving that data view to another text file, then loading it again
                // also matches the baseline.

                string savedPath;

                if (!useCustomEscapeChar)
                    savedPath = DeleteOutputPath("multiline-saved.tsv");
                else
                    savedPath = DeleteOutputPath("multiline-escapechar-saved.tsv");

                using (var fs = File.Create(savedPath))
                    mlContext.Data.SaveAsText(data, fs, separatorChar: '\t');

                options.Separator = "\t";
                options.EscapeChar = '"'; // TextSaver always uses " as escape char
                data = mlContext.Data.LoadFromTextFile(savedPath, options);
            }

            // Get values from loaded dataview
            var ids = new List<string>();
            var descriptions = new List<string>();
            var animals = new List<string>();
            using(var curs = data.GetRowCursorForAllColumns())
            {
                var idGetter = curs.GetGetter<int>(data.Schema["id"]);
                var descriptionGetter = curs.GetGetter<ReadOnlyMemory<char>>(data.Schema["description"]);
                var animalGetter = curs.GetGetter<ReadOnlyMemory<char>>(data.Schema["animal"]);

                int id = default;
                ReadOnlyMemory<char> description = default;
                ReadOnlyMemory<char> animal = default;

                while(curs.MoveNext())
                {
                    idGetter(ref id);
                    descriptionGetter(ref description);
                    animalGetter(ref animal);

                    ids.Add(id.ToString());
                    descriptions.Add(description.ToString());
                    animals.Add(animal.ToString());
                }
            }

            const int numRows = 13;
            Assert.Equal(numRows, ids.Count());
            Assert.Equal(numRows, descriptions.Count());
            Assert.Equal(numRows, animals.Count());

            // Compare values with baseline file
            string line;
            using (var file = new StreamReader(baselinePath))
            {
                for(int i = 0; i < numRows; i++)
                {
                    line = file.ReadLine();
                    Assert.Equal(ids[i], line);
                }

                for (int i = 0; i < numRows; i++)
                {
                    line = file.ReadLine();
                    line = line.Replace("\\n", "\n");
                    Assert.Equal(descriptions[i], line);
                }

                for (int i = 0; i < numRows; i++)
                {
                    line = file.ReadLine();
                    Assert.Equal(animals[i], line);
                }
            }
        }

        [Fact]
        public void TestInvalidMultilineCSVQuote()
        {
            var mlContext = new MLContext(seed: 1);

            string badInputCsv =
                "id,description,animal\n" +
                "9,\"this is a quoted field correctly formatted\",cat\n" +
                "10,\"this is a quoted field\nwithout closing quote,cat\n" +
                "11,this field isn't quoted,dog\n" +
                "12,this will reach the end of the file without finding a closing quote so it will throw,frog\n"
                ;

            var filePath = GetOutputPath("multiline-invalid.csv");
            File.WriteAllText(filePath, badInputCsv);

            bool threwException = false;
            try
            {
                var options = new TextLoader.Options()
                {
                    HasHeader = true,
                    Separator = ",",
                    AllowQuoting = true,
                    ReadMultilines = true,
                    Columns = new[]
                    {
                    new TextLoader.Column("id", DataKind.Int32, 0),
                    new TextLoader.Column("description", DataKind.String, 1),
                    new TextLoader.Column("animal", DataKind.String, 2),
                },
                };

                var data = mlContext.Data.LoadFromTextFile(filePath, options);

                data.Preview();
            }
            catch(EndOfStreamException)
            {
                threwException = true;
            }
            catch(FormatException)
            {
                threwException = true;
            }

            Assert.True(threwException, "Invalid file should have thrown an exception");
        }
    }
}
