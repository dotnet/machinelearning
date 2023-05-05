// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.DotNet.Interactive.Formatting;
using Microsoft.DotNet.Interactive.Formatting.TabularData;
using Xunit;

namespace Microsoft.Data.Analysis.Interactive.Tests
{
    public class DataFrameInteractiveTests
    {
        private Regex _buttonHtmlPart = new Regex(@"<\s*button.*onclick=.*>");
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
            Assert.DoesNotMatch(_buttonHtmlPart, html);
        }

        [Fact]
        public void MoreThanOnePageDataFrameTest()
        {
            DataFrame dataFrame = MakeDataFrameWithTwoColumns(length: 26);
            DataFrameKernelExtension.RegisterDataFrame();
            var html = dataFrame.ToDisplayString("text/html");

            Assert.Contains(TableHtmlPart, html);
            Assert.Matches(_buttonHtmlPart, html);
        }

        [Fact]
        public void DataFrameInfoTest()
        {
            DataFrame dataFrame = MakeDataFrameWithTwoColumns(length: 5);
            DataFrameKernelExtension.RegisterDataFrame();
            var html = dataFrame.Info().ToDisplayString("text/html");

            Assert.Contains(TableHtmlPart, html);
            Assert.DoesNotMatch(_buttonHtmlPart, html);
        }

        [Fact]
        public void LoadFromTabularDataResource()
        {
            var schema = new TableSchema();
            schema.Fields.Add(new TableSchemaFieldDescriptor("TrueOrFalse", TableSchemaFieldType.Boolean));
            schema.Fields.Add(new TableSchemaFieldDescriptor("Integer", TableSchemaFieldType.Integer));
            schema.Fields.Add(new TableSchemaFieldDescriptor("Double", TableSchemaFieldType.Number));
            schema.Fields.Add(new TableSchemaFieldDescriptor("Text", TableSchemaFieldType.String));
            var data = new List<IDictionary<string, object>>();

            for (var i = 0; i < 5; i++)
            {
                data.Add(new Dictionary<string, object>
                {
                    ["TrueOrFalse"] = ((i % 2) == 0),
                    ["Integer"] = i,
                    ["Double"] = i / 0.5,
                    ["Text"] = $"hello {i}!"
                });
            }

            var tableData = new TabularDataResource(schema, data);

            var dataFrame = tableData.ToDataFrame();
            Assert.Equal(dataFrame.Columns.Select(c => c.Name).ToArray(), new[] { "TrueOrFalse", "Integer", "Double", "Text" });
        }
    }
}
