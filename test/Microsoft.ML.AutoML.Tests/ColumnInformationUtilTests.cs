// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class ColumnInformationUtilTests
    {
        [TestMethod]
        public void GetColumnPurpose()
        {
            var columnInfo = new ColumnInformation()
            {
                LabelColumnName = "Label",
                ExampleWeightColumnName = "Weight",
                SamplingKeyColumnName = "SamplingKey",
            };
            columnInfo.CategoricalColumnNames.Add("Cat");
            columnInfo.NumericColumnNames.Add("Num");
            columnInfo.TextColumnNames.Add("Text");
            columnInfo.IgnoredColumnNames.Add("Ignored");

            Assert.AreEqual(ColumnPurpose.Label, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Label"));
            Assert.AreEqual(ColumnPurpose.Weight, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Weight"));
            Assert.AreEqual(ColumnPurpose.SamplingKey, ColumnInformationUtil.GetColumnPurpose(columnInfo, "SamplingKey"));
            Assert.AreEqual(ColumnPurpose.CategoricalFeature, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Cat"));
            Assert.AreEqual(ColumnPurpose.NumericFeature, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Num"));
            Assert.AreEqual(ColumnPurpose.TextFeature, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Text"));
            Assert.AreEqual(ColumnPurpose.Ignore, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Ignored"));
            Assert.AreEqual(null, ColumnInformationUtil.GetColumnPurpose(columnInfo, "NonExistent"));
        }
    }
}
