// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{

    public class ColumnInformationUtilTests : BaseTestClass
    {
        public ColumnInformationUtilTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void GetColumnPurpose()
        {
            var columnInfo = new ColumnInformation()
            {
                LabelColumnName = "Label",
                ExampleWeightColumnName = "Weight",
                SamplingKeyColumnName = "SamplingKey",
                UserIdColumnName = "UserId",
                ItemIdColumnName = "MovieId",
                GroupIdColumnName = "GroupId"
            };

            columnInfo.CategoricalColumnNames.Add("Cat");
            columnInfo.NumericColumnNames.Add("Num");
            columnInfo.TextColumnNames.Add("Text");
            columnInfo.IgnoredColumnNames.Add("Ignored");

            Assert.Equal(ColumnPurpose.Label, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Label"));
            Assert.Equal(ColumnPurpose.Weight, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Weight"));
            Assert.Equal(ColumnPurpose.SamplingKey, ColumnInformationUtil.GetColumnPurpose(columnInfo, "SamplingKey"));
            Assert.Equal(ColumnPurpose.UserId, ColumnInformationUtil.GetColumnPurpose(columnInfo, "UserId"));
            Assert.Equal(ColumnPurpose.GroupId, ColumnInformationUtil.GetColumnPurpose(columnInfo, "GroupId"));
            Assert.Equal(ColumnPurpose.ItemId, ColumnInformationUtil.GetColumnPurpose(columnInfo, "MovieId"));
            Assert.Equal(ColumnPurpose.CategoricalFeature, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Cat"));
            Assert.Equal(ColumnPurpose.NumericFeature, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Num"));
            Assert.Equal(ColumnPurpose.TextFeature, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Text"));
            Assert.Equal(ColumnPurpose.Ignore, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Ignored"));
            Assert.Null(ColumnInformationUtil.GetColumnPurpose(columnInfo, "NonExistent"));
        }

        [Fact]
        public void GetColumnNamesTest()
        {
            var columnInfo = new ColumnInformation()
            {
                LabelColumnName = "Label",
                SamplingKeyColumnName = "SamplingKey",
                UserIdColumnName = "UserId",
                ItemIdColumnName = "MovieId",
                GroupIdColumnName = "GroupId"
            };
            columnInfo.CategoricalColumnNames.Add("Cat1");
            columnInfo.CategoricalColumnNames.Add("Cat2");
            columnInfo.NumericColumnNames.Add("Num");
            var columnNames = ColumnInformationUtil.GetColumnNames(columnInfo);
            Assert.Equal(8, columnNames.Count());
            Assert.Contains("Label", columnNames);
            Assert.Contains("SamplingKey", columnNames);
            Assert.Contains("UserId", columnNames);
            Assert.Contains("MovieId", columnNames);
            Assert.Contains("GroupId", columnNames);
            Assert.Contains("Cat1", columnNames);
            Assert.Contains("Cat2", columnNames);
            Assert.Contains("Num", columnNames);
        }
    }
}
