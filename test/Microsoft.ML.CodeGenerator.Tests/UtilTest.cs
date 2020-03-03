using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.Utilities;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.CodeGenerator.Tests
{
    public class UtilTest : BaseTestClass
    {
        public UtilTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestGenerateSampleData()
        {
            var data = new[]
            {
                new
                {
                    Label = "label1",
                    STR = "feature1",
                    INT = 2,
                    DOUBLE = 1.2,
                    FLOAT = 1.223E+10F,
                    TrickySTR = "ab\"\';@#$%^&-++==",
                }
            };

            var context = new MLContext();
            var dataView = context.Data.LoadFromEnumerable(data);
            var columnInference = new ColumnInferenceResults()
            {
                ColumnInformation = new ColumnInformation()
                {
                    LabelColumnName = "Label"
                }
            };

            var sampleData = Utils.GenerateSampleData(dataView, columnInference);
            Assert.Equal("\"feature1\"", sampleData["STR"]);
            Assert.Equal("2", sampleData["INT"]);
            Assert.Equal("1.2", sampleData["DOUBLE"]);
            Assert.Equal("1.223E+10F", sampleData["FLOAT"]);
            Assert.Equal("\"ab\\\"\';@#$%^&-++==\"", sampleData["TrickySTR"]);
        }
    }
}
