// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.Utilities;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.CodeGenerator.Tests
{
    class TestClass
    {
        [LoadColumn(0)]
        public string Label { get; set; }

        [LoadColumn(1)]
        public string STR { get; set; }

        [LoadColumn(2)]
        public string PATH { get; set; }

        [LoadColumn(3)]
        public int INT { get; set; }

        [LoadColumn(4)]
        public Double DOUBLE { get; set; }

        [LoadColumn(5)]
        public float FLOAT { get; set; }

        [LoadColumn(6)]
        public string TrickySTR { get; set; }

        [LoadColumn(7)]
        public float SingleNan { get; set; }

        [LoadColumn(8)]
        public float SinglePositiveInfinity { get; set; }

        [LoadColumn(9)]
        public float SingleNegativeInfinity { get; set; }

        [LoadColumn(10)]
        public string EmptyString { get; set; }

        [LoadColumn(11)]
        public bool One { get; set; }

        [LoadColumn(12)]
        public bool T { get; set; }
    }

    class TestClassContainsDuplicates
    {
        [LoadColumn(0)]
        public string Label_col_0 { get; set; }

        [LoadColumn(1)]
        public string STR_col_1 { get; set; }

        [LoadColumn(2)]
        public string STR_col_2 { get; set; }

        [LoadColumn(3)]
        public string PATH_col_3 { get; set; }

        [LoadColumn(4)]
        public int INT_col_4 { get; set; }

        [LoadColumn(5)]
        public Double DOUBLE_col_5 { get; set; }

        [LoadColumn(6)]
        public float FLOAT_col_6 { get; set; }

        [LoadColumn(7)]
        public float FLOAT_col_7 { get; set; }

        [LoadColumn(8)]
        public string TrickySTR_col_8 { get; set; }

        [LoadColumn(9)]
        public float SingleNan_col_9 { get; set; }

        [LoadColumn(10)]
        public float SinglePositiveInfinity_col_10 { get; set; }

        [LoadColumn(11)]
        public float SingleNegativeInfinity_col_11 { get; set; }

        [LoadColumn(12)]
        public float SingleNegativeInfinity_col_12 { get; set; }

        [LoadColumn(13)]
        public string EmptyString_col_13 { get; set; }

        [LoadColumn(14)]
        public bool One_col_14 { get; set; }

        [LoadColumn(15)]
        public bool T_col_15 { get; set; }
    }

    public class UtilTest : BaseTestClass
    {
        public UtilTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public async Task TestGenerateSampleDataAsync()
        {
            var filePath = "sample.txt";
            using (var file = new StreamWriter(filePath))
            {
                await file.WriteLineAsync("Label,STR,PATH,INT,DOUBLE,FLOAT,TrickySTR,SingleNan,SinglePositiveInfinity,SingleNegativeInfinity,EmptyString,One,T");
                await file.WriteLineAsync("label1,feature1,/path/to/file,2,1.2,1.223E+10,ab\"\';@#$%^&-++==,NaN,Infinity,-Infinity,,1,T");
                await file.FlushAsync();
                file.Close();
                var context = new MLContext();
                var dataView = context.Data.LoadFromTextFile<TestClass>(filePath, separatorChar: ',', hasHeader: true);
                var columnInference = new ColumnInferenceResults()
                {
                    ColumnInformation = new ColumnInformation()
                    {
                        LabelColumnName = "Label"
                    }
                };
                var sampleData = Utils.GenerateSampleData(dataView, columnInference);
                Assert.Equal("@\"feature1\"", sampleData["STR"]);
                Assert.Equal("@\"/path/to/file\"", sampleData["PATH"]);
                Assert.Equal("2", sampleData["INT"]);
                Assert.Equal("1.2", sampleData["DOUBLE"]);
                Assert.Equal("1.223E+10F", sampleData["FLOAT"]);
                Assert.Equal("@\"ab\\\"\';@#$%^&-++==\"", sampleData["TrickySTR"]);
                Assert.Equal($"Single.NaN", sampleData["SingleNan"]);
                Assert.Equal($"Single.PositiveInfinity", sampleData["SinglePositiveInfinity"]);
                Assert.Equal($"Single.NegativeInfinity", sampleData["SingleNegativeInfinity"]);
                Assert.Equal("@\"\"", sampleData["EmptyString"]);
                Assert.Equal($"true", sampleData["One"]);
                Assert.Equal($"true", sampleData["T"]);
            }
        }

        [Fact]
        public async Task TestGenerateSampleDataAsyncDuplicateColumnNames()
        {
            var filePath = "sample2.txt";
            using (var file = new StreamWriter(filePath))
            {
                await file.WriteLineAsync("Label,STR,STR,PATH,INT,DOUBLE,FLOAT,FLOAT,TrickySTR,SingleNan,SinglePositiveInfinity,SingleNegativeInfinity,SingleNegativeInfinity,EmptyString,One,T");
                await file.WriteLineAsync("label1,feature1,feature2,/path/to/file,2,1.2,1.223E+10,1.223E+11,ab\"\';@#$%^&-++==,NaN,Infinity,-Infinity,-Infinity,,1,T");
                await file.FlushAsync();
                file.Close();
                var context = new MLContext();
                var dataView = context.Data.LoadFromTextFile<TestClassContainsDuplicates>(filePath, separatorChar: ',', hasHeader: true);
                var columnInference = new ColumnInferenceResults()
                {
                    ColumnInformation = new ColumnInformation()
                    {
                        LabelColumnName = "Label_col_0"
                    }
                };
                var sampleData = Utils.GenerateSampleData(dataView, columnInference);
                Assert.Equal("@\"feature1\"", sampleData["STR_col_1"]);
                Assert.Equal("@\"feature2\"", sampleData["STR_col_2"]);
                Assert.Equal("@\"/path/to/file\"", sampleData["PATH_col_3"]);
                Assert.Equal("2", sampleData["INT_col_4"]);
                Assert.Equal("1.2", sampleData["DOUBLE_col_5"]);
                Assert.Equal("1.223E+10F", sampleData["FLOAT_col_6"]);
                Assert.Equal("1.223E+11F", sampleData["FLOAT_col_7"]);
                Assert.Equal("@\"ab\\\"\';@#$%^&-++==\"", sampleData["TrickySTR_col_8"]);
                Assert.Equal($"Single.NaN", sampleData["SingleNan_col_9"]);
                Assert.Equal($"Single.PositiveInfinity", sampleData["SinglePositiveInfinity_col_10"]);
                Assert.Equal($"Single.NegativeInfinity", sampleData["SingleNegativeInfinity_col_11"]);
                Assert.Equal($"Single.NegativeInfinity", sampleData["SingleNegativeInfinity_col_12"]);
                Assert.Equal("@\"\"", sampleData["EmptyString_col_13"]);
                Assert.Equal($"true", sampleData["One_col_14"]);
                Assert.Equal($"true", sampleData["T_col_15"]);
            }
        }

        [Fact]
        public void NormalizeTest()
        {
            var testStrArray = new string[] { "Abc Abc", "abc ABC", "12", "12.3", "1AB .C" };
            var expectedStrArray = new string[] { "Abc_Abc", "Abc_ABC", "_12", "_12_3", "_1AB__C" };
            for (int i = 0; i != expectedStrArray.Count(); ++i)
            {
                var actualStr = Microsoft.ML.CodeGenerator.Utilities.Utils.Normalize(testStrArray[i]);
                Assert.Equal(expectedStrArray[i], actualStr);
            }
        }
    }
}
