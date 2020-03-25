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
                var dataView = context.Data.LoadFromTextFile<TestClass>(filePath,separatorChar:',', hasHeader: true);
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
        public void NormalizeTest()
        {
            var testStrArray = new string[] { "Abc Abc", "abc ABC", "12", "12.3", "1AB .C"};
            var expectedStrArray = new string[] { "Abc_Abc", "Abc_ABC", "_12", "_12_3", "_1AB__C" };
            for (int i = 0; i != expectedStrArray.Count(); ++i)
            {
                var actualStr = Microsoft.ML.CodeGenerator.Utilities.Utils.Normalize(testStrArray[i]);
                Assert.Equal(expectedStrArray[i], actualStr);
            }
        }
    }
}
