// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.Data.Analysis.Tests
{
    public class DataFrameGroupByTests : BaseTestClass
    {
        public DataFrameGroupByTests(ITestOutputHelper output) : base(output, true)
        {
        }

        [Fact]
        public void TestGroupingWithTKeyTypeofString()
        {
            const int length = 11;

            //Create test dataframe (numbers starting from 0 up to length)
            DataFrame df = MakeTestDataFrameWithParityAndTensColumns(length);

            var grouping = df.GroupBy<string>("Parity").Groupings;

            //Check groups count
            Assert.Equal(2, grouping.Count());

            //Check number of elements in each group
            var oddGroup = grouping.Where(gr => gr.Key == "odd").FirstOrDefault();
            Assert.NotNull(oddGroup);
            Assert.Equal(length / 2, oddGroup.Count());

            var evenGroup = grouping.Where(gr => gr.Key == "even").FirstOrDefault();
            Assert.NotNull(evenGroup);
            Assert.Equal(length / 2 + length % 2, evenGroup.Count());


        }

        [Fact]
        public void TestGroupingWithTKey_CornerCases()
        {
            //Check corner cases
            var df = MakeTestDataFrameWithParityAndTensColumns(0);
            var grouping = df.GroupBy<string>("Parity").Groupings;
            Assert.Empty(grouping);


            df = MakeTestDataFrameWithParityAndTensColumns(1);
            grouping = df.GroupBy<string>("Parity").Groupings;
            Assert.Single(grouping);
            Assert.Equal("even", grouping.First().Key);
        }


        [Fact]
        public void TestGroupingWithTKeyPrimitiveType()
        {
            const int length = 55;

            //Create test dataframe (numbers starting from 0 up to length)
            DataFrame df = MakeTestDataFrameWithParityAndTensColumns(length);

            //Group elements by int column, that contain the amount of full tens in each int
            var groupings = df.GroupBy<int>("Tens").Groupings.ToDictionary(g => g.Key, g => g.ToList());

            //Get the amount of all number based columns
            int numberColumnsCount = df.Columns.Count - 2; //except "Parity" and "Tens" columns

            //Check each group
            for (int i = 0; i < length / 10; i++)
            {
                Assert.Equal(10, groupings[i].Count());

                var rows = groupings[i];
                for (int colIndex = 0; colIndex < numberColumnsCount; colIndex++)
                {
                    var values = rows.Select(row => Convert.ToInt32(row[colIndex]));

                    for (int j = 0; j < 10; j++)
                    {
                        Assert.Contains(i * 10 + j, values);
                    }
                }
            }

            //Last group should contain smaller amount of items
            Assert.Equal(length % 10, groupings[length / 10].Count());
        }

        [Fact]
        public void TestGroupingWithTKeyOfWrongType()
        {
            var message = string.Empty;

            //Create test dataframe (numbers starting from 0 up to length)
            DataFrame df = MakeTestDataFrameWithParityAndTensColumns(1);

            //Use wrong type for grouping
            Assert.Throws<InvalidCastException>(() => df.GroupBy<double>("Tens"));
        }


        [Fact]
        public void TestGroupByMinMaxProductSum()
        {
            var key = new Int32DataFrameColumn("Key", Enumerable.Range(0, 30).Select(x => x / 10));
            var positive = new Int32DataFrameColumn("Positive", Enumerable.Range(0, 30).Select(x => x % 10 + 1));
            var negative = new Int32DataFrameColumn("Negative", Enumerable.Range(0, 30).Select(x => -(x % 10) - 1));
            var small = new Int32DataFrameColumn("Small", Enumerable.Range(0, 30).Select(x => x % 3 + 1));
            var dates = new PrimitiveDataFrameColumn<DateTime>("Date", Enumerable.Range(0, 30).Select(x => new DateTime(2020, 1, 1).AddDays(x)));

            // Nulls must be ignored by the aggregations
            positive[0] = null;
            negative[0] = null;
            small[0] = null;
            dates[0] = null;

            // Group 2 contains only nulls
            for (int i = 20; i < 30; i++)
            {
                positive[i] = null;
                negative[i] = null;
                small[i] = null;
                dates[i] = null;
            }

            var df = new DataFrame(key, positive, negative, small, dates);

            DataFrame min = df.GroupBy("Key").Min("Positive", "Negative", "Date");
            Assert.Equal(2, min.Columns["Positive"][0]);
            Assert.Equal(1, min.Columns["Positive"][1]);
            Assert.Null(min.Columns["Positive"][2]);
            Assert.Equal(-10, min.Columns["Negative"][0]);
            Assert.Null(min.Columns["Negative"][2]);
            Assert.Equal(new DateTime(2020, 1, 2), min.Columns["Date"][0]);
            Assert.Equal(new DateTime(2020, 1, 11), min.Columns["Date"][1]);
            Assert.Null(min.Columns["Date"][2]);

            DataFrame max = df.GroupBy("Key").Max("Positive", "Negative", "Date");
            Assert.Equal(10, max.Columns["Positive"][0]);
            Assert.Null(max.Columns["Positive"][2]);
            Assert.Equal(-2, max.Columns["Negative"][0]);
            Assert.Equal(-1, max.Columns["Negative"][1]);
            Assert.Null(max.Columns["Negative"][2]);
            Assert.Equal(new DateTime(2020, 1, 10), max.Columns["Date"][0]);
            Assert.Equal(new DateTime(2020, 1, 20), max.Columns["Date"][1]);
            Assert.Null(max.Columns["Date"][2]);

            // Group 0: 2,3,1,2,3,1,2,3,1 (first value is null); group 1: 2,3,1,2,3,1,2,3,1,2
            DataFrame product = df.GroupBy("Key").Product("Small");
            Assert.Equal(216, product.Columns["Small"][0]);
            Assert.Equal(432, product.Columns["Small"][1]);

            DataFrame sum = df.GroupBy("Key").Sum("Positive");
            Assert.Equal(54, sum.Columns["Positive"][0]);
            Assert.Equal(55, sum.Columns["Positive"][1]);
            Assert.Null(sum.Columns["Positive"][2]);
        }


        private DataFrame MakeTestDataFrameWithParityAndTensColumns(int length)
        {
            DataFrame df = DataFrameTests.MakeDataFrameWithNumericColumns(length, false);
            DataFrameColumn parityColumn = new StringDataFrameColumn("Parity", Enumerable.Range(0, length).Select(x => x % 2 == 0 ? "even" : "odd"));
            DataFrameColumn tensColumn = new Int32DataFrameColumn("Tens", Enumerable.Range(0, length).Select(x => x / 10));
            df.Columns.Insert(df.Columns.Count, parityColumn);
            df.Columns.Insert(df.Columns.Count, tensColumn);

            return df;
        }
    }
}
