// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Xunit;
using Microsoft.DotNet.Interactive.Formatting;

namespace Microsoft.Data.Analysis.Interactive.Tests
{
    public class DataFrameInteractiveTests
    {
        private const string ButtonHtmlPart = "button onclick";
        private const string TableHtmlPart = "<table";

        public static DataFrame MakeDataFrameWithTwoColumns(int length, bool withNulls = true)
        {
            DataFrameColumn dataFrameColumn1 = new Int32DataFrameColumn("Int1", Enumerable.Range(0, length).Select(x => x));
            DataFrameColumn dataFrameColumn2 = new Int32DataFrameColumn("Int2", Enumerable.Range(10, length).Select(x => x));
            if (withNulls)
            {
                dataFrameColumn1[length / 2] = null;
                dataFrameColumn2[length / 2] = null;
            }
            DataFrame dataFrame = new DataFrame();
            dataFrame.Columns.Insert(0, dataFrameColumn1);
            dataFrame.Columns.Insert(1, dataFrameColumn2);
            return dataFrame;
        }

        [Fact]
        public void LessThanOnePageDataFrameTest()
        {
            DataFrame dataFrame = MakeDataFrameWithTwoColumns(length: 5);
            DataFrameKernelExtension.RegisterDataFrame();
            var html = dataFrame.ToDisplayString("text/html");

            Assert.Contains(TableHtmlPart, html);
            Assert.DoesNotContain(ButtonHtmlPart, html);
        }

        [Fact]
        public void MoreThanOnePageDataFrameTest()
        {
            DataFrame dataFrame = MakeDataFrameWithTwoColumns(length: 26);
            DataFrameKernelExtension.RegisterDataFrame();
            var html = dataFrame.ToDisplayString("text/html");

            Assert.Contains(TableHtmlPart, html);
            Assert.Contains(ButtonHtmlPart, html);
        }

        [Fact]
        public void DataFrameInfoTest()
        {
            DataFrame dataFrame = MakeDataFrameWithTwoColumns(length: 5);
            DataFrameKernelExtension.RegisterDataFrame();
            var html = dataFrame.Info().ToDisplayString("text/html");

            Assert.Contains(TableHtmlPart, html);
            Assert.DoesNotContain(ButtonHtmlPart, html);
        }
    }
}
