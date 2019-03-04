using System;
using System.Collections.Generic;
using System.Text;
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
                LabelColumn = "Label",
                WeightColumn = "Weight",
            };
            columnInfo.CategoricalColumns.Add("Cat");
            columnInfo.NumericColumns.Add("Num");
            columnInfo.TextColumns.Add("Text");
            columnInfo.IgnoredColumns.Add("Ignored");

            Assert.AreEqual(ColumnPurpose.Label, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Label"));
            Assert.AreEqual(ColumnPurpose.Weight, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Weight"));
            Assert.AreEqual(ColumnPurpose.CategoricalFeature, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Cat"));
            Assert.AreEqual(ColumnPurpose.NumericFeature, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Num"));
            Assert.AreEqual(ColumnPurpose.TextFeature, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Text"));
            Assert.AreEqual(ColumnPurpose.Ignore, ColumnInformationUtil.GetColumnPurpose(columnInfo, "Ignored"));
            Assert.AreEqual(null, ColumnInformationUtil.GetColumnPurpose(columnInfo, "NonExistent"));
        }
    }
}
