// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.EntryPoints.Tests
{
#pragma warning disable 612, 618
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
                "loader=Text{col=DvInt1:I1:0 col=DvInt2:I2:1 col=DvInt4:I4:2 col=DvInt8:I8:3 sep=comma}",
                }, logCurs: true);

            using (var cursor = data.GetRowCursor((a => true)))
            {
                var col1 = cursor.GetGetter<sbyte>(0);
                var col2 = cursor.GetGetter<short>(1);
                var col3 = cursor.GetGetter<int>(2);
                var col4 = cursor.GetGetter<long>(3);

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
                Assert.Equal("Could not parse value -9223372036854775809 in line 1, column DvInt8", ex.Message);
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
                Assert.Equal("Could not parse value 9223372036854775808 in line 1, column DvInt8", ex.Message);
                return;
            }

            Assert.True(false, "Test failed.");
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
            [LoadColumn(start:0, end:3), ColumnName("Features")]
            public float Features;

            [LoadColumn(4), ColumnName("Label")]
            public string Type;
        }

        public class IrisColumnIndices
        {
            [LoadColumn(columnIndexes: new[] { 0, 2 })]
            public float Features;

            [LoadColumn(4), ColumnName("Label")]
            public string Type;
        }

        [Fact]
        public void LoaderColumnsFromIrisData()
        {
            var dataPath = GetDataPath(TestDatasets.irisData.trainFilename);
            var ml = new MLContext();

            var irisFirstRow = new Dictionary<string, float>();
            irisFirstRow["SepalLength"] = 5.1f;
            irisFirstRow["SepalWidth"] = 3.5f;
            irisFirstRow["PetalLength"] = 1.4f;
            irisFirstRow["PetalWidth"] = 0.2f;

            var irisFirstRowValues = irisFirstRow.Values.GetEnumerator();

            // Simple load
            var dataIris = ml.Data.CreateTextReader<Iris>(separatorChar: ',').Read(dataPath);
            var previewIris = dataIris.Preview(1);

            Assert.Equal(5, previewIris.ColumnView.Length);
            Assert.Equal("SepalLength", previewIris.Schema[0].Name);
            Assert.Equal(NumberType.R4, previewIris.Schema[0].Type);
            int index = 0;
            foreach (var entry in irisFirstRow)
            {
                Assert.Equal(entry.Key, previewIris.RowView[0].Values[index].Key);
                Assert.Equal(entry.Value, previewIris.RowView[0].Values[index++].Value);
            }
            Assert.Equal("Type", previewIris.RowView[0].Values[index].Key);
            Assert.Equal("Iris-setosa", previewIris.RowView[0].Values[index].Value.ToString());

            // Load with start and end indexes
            var dataIrisStartEnd = ml.Data.CreateTextReader<IrisStartEnd>(separatorChar: ',').Read(dataPath);
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
            var dataIrisColumnIndices = ml.Data.CreateTextReader<IrisColumnIndices>(separatorChar: ',').Read(dataPath);
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
    }
#pragma warning restore 612, 618
}
