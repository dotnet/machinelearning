// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Featurizers;
using System;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.TestFramework.Attributes;

namespace Microsoft.ML.Tests.Transformers
{
    public class DateTimeTransformerTests : TestDataPipeBase
    {
        public DateTimeTransformerTests(ITestOutputHelper output) : base(output)
        {
        }

        private class DateTimeInput
        {
            public long date;
        }

        [NotCentOS7FactAttribute]
        public void CorrectNumberOfColumnsAndSchema()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new DateTimeInput() { date = 0 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var columnPrefix = "DTC_";
            var pipeline = mlContext.Transforms.FeaturizeDateTime("date", columnPrefix);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            // Check the schema has 22 columns
            Assert.Equal(22, schema.Count);

            // Make sure names with prefix and order are correct
            Assert.Equal($"{columnPrefix}Year", schema[1].Name);
            Assert.Equal($"{columnPrefix}Month", schema[2].Name);
            Assert.Equal($"{columnPrefix}Day", schema[3].Name);
            Assert.Equal($"{columnPrefix}Hour", schema[4].Name);
            Assert.Equal($"{columnPrefix}Minute", schema[5].Name);
            Assert.Equal($"{columnPrefix}Second", schema[6].Name);
            Assert.Equal($"{columnPrefix}AmPm", schema[7].Name);
            Assert.Equal($"{columnPrefix}Hour12", schema[8].Name);
            Assert.Equal($"{columnPrefix}DayOfWeek", schema[9].Name);
            Assert.Equal($"{columnPrefix}DayOfQuarter", schema[10].Name);
            Assert.Equal($"{columnPrefix}DayOfYear", schema[11].Name);
            Assert.Equal($"{columnPrefix}WeekOfMonth", schema[12].Name);
            Assert.Equal($"{columnPrefix}QuarterOfYear", schema[13].Name);
            Assert.Equal($"{columnPrefix}HalfOfYear", schema[14].Name);
            Assert.Equal($"{columnPrefix}WeekIso", schema[15].Name);
            Assert.Equal($"{columnPrefix}YearIso", schema[16].Name);
            Assert.Equal($"{columnPrefix}MonthLabel", schema[17].Name);
            Assert.Equal($"{columnPrefix}AmPmLabel", schema[18].Name);
            Assert.Equal($"{columnPrefix}DayOfWeekLabel", schema[19].Name);
            Assert.Equal($"{columnPrefix}HolidayName", schema[20].Name);
            Assert.Equal($"{columnPrefix}IsPaidTimeOff", schema[21].Name);

            // Make sure types are correct
            Assert.Equal(typeof(int), schema[1].Type.RawType);
            Assert.Equal(typeof(byte), schema[2].Type.RawType);
            Assert.Equal(typeof(byte), schema[3].Type.RawType);
            Assert.Equal(typeof(byte), schema[4].Type.RawType);
            Assert.Equal(typeof(byte), schema[5].Type.RawType);
            Assert.Equal(typeof(byte), schema[6].Type.RawType);
            Assert.Equal(typeof(byte), schema[7].Type.RawType);
            Assert.Equal(typeof(byte), schema[8].Type.RawType);
            Assert.Equal(typeof(byte), schema[9].Type.RawType);
            Assert.Equal(typeof(byte), schema[10].Type.RawType);
            Assert.Equal(typeof(ushort), schema[11].Type.RawType);
            Assert.Equal(typeof(ushort), schema[12].Type.RawType);
            Assert.Equal(typeof(byte), schema[13].Type.RawType);
            Assert.Equal(typeof(byte), schema[14].Type.RawType);
            Assert.Equal(typeof(byte), schema[15].Type.RawType);
            Assert.Equal(typeof(int), schema[16].Type.RawType);
            Assert.Equal(typeof(ReadOnlyMemory<char>), schema[17].Type.RawType);
            Assert.Equal(typeof(ReadOnlyMemory<char>), schema[18].Type.RawType);
            Assert.Equal(typeof(ReadOnlyMemory<char>), schema[19].Type.RawType);
            Assert.Equal(typeof(ReadOnlyMemory<char>), schema[20].Type.RawType);
            Assert.Equal(typeof(byte), schema[21].Type.RawType);

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7FactAttribute]
        public void CanUseDateFromColumn()
        {
            // Future Date - 2025 June 30
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new DateTimeInput() { date = 1751241600 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.FeaturizeDateTime("date", "DTC");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);

            // Get the data from the first row and make sure it matches expected
            var row = output.Preview(1).RowView[0].Values;

            // Assert the data from the first row is what we expect
            Assert.Equal(2025, row[1].Value);                           // Year
            Assert.Equal((byte)6, row[2].Value);                        // Month
            Assert.Equal((byte)30, row[3].Value);                       // Day
            Assert.Equal((byte)0, row[4].Value);                        // Hour
            Assert.Equal((byte)0, row[5].Value);                        // Minute
            Assert.Equal((byte)0, row[6].Value);                        // Second
            Assert.Equal((byte)0, row[7].Value);                        // AmPm
            Assert.Equal((byte)0, row[8].Value);                        // Hour12
            Assert.Equal((byte)1, row[9].Value);                        // DayOfWeek
            Assert.Equal((byte)91, row[10].Value);                      // DayOfQuarter
            Assert.Equal((ushort)180, row[11].Value);                   // DayOfYear
            Assert.Equal((ushort)4, row[12].Value);                     // WeekOfMonth
            Assert.Equal((byte)2, row[13].Value);                       // QuarterOfYear
            Assert.Equal((byte)1, row[14].Value);                       // HalfOfYear
            Assert.Equal((byte)27, row[15].Value);                      // WeekIso
            Assert.Equal(2025, row[16].Value);                          // YearIso
            Assert.Equal("June", row[17].Value.ToString());             // MonthLabel
            Assert.Equal("am", row[18].Value.ToString());               // AmPmLabel
            Assert.Equal("Monday", row[19].Value.ToString());           // DayOfWeekLabel
            Assert.Equal("", row[20].Value.ToString());  // HolidayName
            Assert.Equal((byte)0, row[21].Value);                       // IsPaidTimeOff

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7FactAttribute]
        public void HolidayTest()
        {
            // Future Date - 2025 June 30
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new DateTimeInput() { date = 157161600 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.FeaturizeDateTime("date", "DTC", country: DateTimeEstimator.HolidayList.Canada);
            var model = pipeline.Fit(data);
            var output = model.Transform(data);

            // Get the data from the first row and make sure it matches expected
            var row = output.Preview(1).RowView[0].Values;

            // Assert the data from the first row for holidays is what we expect
            Assert.Equal("Christmas Day", row[20].Value.ToString());  // HolidayName
            Assert.Equal((byte)0, row[21].Value);                     // IsPaidTimeOff

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7FactAttribute]
        public void ManyRowsTest()
        {
            // Future Date - 2025 June 30
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new DateTimeInput() { date = 1751241600 }, new DateTimeInput() { date = 1751241600 }, new DateTimeInput() { date = 12341 },
                new DateTimeInput() { date = 134 }, new DateTimeInput() { date = 134 }, new DateTimeInput() { date = 1234 }, new DateTimeInput() { date = 1751241600 },
                new DateTimeInput() { date = 1751241600 }, new DateTimeInput() { date = 12341 },
                new DateTimeInput() { date = 134 }, new DateTimeInput() { date = 134 }, new DateTimeInput() { date = 1234 }};

            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var pipeline = mlContext.Transforms.FeaturizeDateTime("date", "DTC");
            var model = pipeline.Fit(data);
            var output = model.Transform(data);

            // Get the data from the first row and make sure it matches expected
            var row = output.Preview().RowView[0].Values;

            // Assert the data from the first row is what we expect
            Assert.Equal(2025, row[1].Value);                           // Year
            Assert.Equal((byte)6, row[2].Value);                        // Month
            Assert.Equal((byte)30, row[3].Value);                       // Day
            Assert.Equal((byte)0, row[4].Value);                        // Hour
            Assert.Equal((byte)0, row[5].Value);                        // Minute
            Assert.Equal((byte)0, row[6].Value);                        // Second
            Assert.Equal((byte)0, row[7].Value);                        // AmPm
            Assert.Equal((byte)0, row[8].Value);                        // Hour12
            Assert.Equal((byte)1, row[9].Value);                        // DayOfWeek
            Assert.Equal((byte)91, row[10].Value);                      // DayOfQuarter
            Assert.Equal((ushort)180, row[11].Value);                   // DayOfYear
            Assert.Equal((ushort)4, row[12].Value);                     // WeekOfMonth
            Assert.Equal((byte)2, row[13].Value);                       // QuarterOfYear
            Assert.Equal((byte)1, row[14].Value);                       // HalfOfYear
            Assert.Equal((byte)27, row[15].Value);                      // WeekIso
            Assert.Equal(2025, row[16].Value);                          // YearIso
            Assert.Equal("June", row[17].Value.ToString());             // MonthLabel
            Assert.Equal("am", row[18].Value.ToString());               // AmPmLabel
            Assert.Equal("Monday", row[19].Value.ToString());           // DayOfWeekLabel
            Assert.Equal("", row[20].Value.ToString());  // HolidayName
            Assert.Equal((byte)0, row[21].Value);                       // IsPaidTimeOff

            TestEstimatorCore(pipeline, data);
            Done();
        }

        [NotCentOS7FactAttribute]
        public void EntryPointTest()
        {
            // Future Date - 2025 June 30
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new DateTimeInput() { date = 1751241600 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, fit, and transform it.
            var options = new DateTimeEstimator.Options
            {
                Source = "date",
                Prefix = "pref_",
                Data = data
            };

            var entryOutput = DateTimeTransformerEntrypoint.DateTimeSplit(mlContext.Transforms.GetEnvironment(), options);
            var output = entryOutput.OutputData;

            // Get the data from the first row and make sure it matches expected
            var row = output.Preview(1).RowView[0].Values;

            // Assert the data from the first row is what we expect
            Assert.Equal(2025, row[1].Value);                           // Year
            Assert.Equal((byte)6, row[2].Value);                        // Month
            Assert.Equal((byte)30, row[3].Value);                       // Day
            Assert.Equal((byte)0, row[4].Value);                        // Hour
            Assert.Equal((byte)0, row[5].Value);                        // Minute
            Assert.Equal((byte)0, row[6].Value);                        // Second
            Assert.Equal((byte)0, row[7].Value);                        // AmPm
            Assert.Equal((byte)0, row[8].Value);                        // Hour12
            Assert.Equal((byte)1, row[9].Value);                        // DayOfWeek
            Assert.Equal((byte)91, row[10].Value);                      // DayOfQuarter
            Assert.Equal((ushort)180, row[11].Value);                   // DayOfYear
            Assert.Equal((ushort)4, row[12].Value);                     // WeekOfMonth
            Assert.Equal((byte)2, row[13].Value);                       // QuarterOfYear
            Assert.Equal((byte)1, row[14].Value);                       // HalfOfYear
            Assert.Equal((byte)27, row[15].Value);                      // WeekIso
            Assert.Equal(2025, row[16].Value);                          // YearIso
            Assert.Equal("June", row[17].Value.ToString());             // MonthLabel
            Assert.Equal("am", row[18].Value.ToString());               // AmPmLabel
            Assert.Equal("Monday", row[19].Value.ToString());           // DayOfWeekLabel
            Assert.Equal("", row[20].Value.ToString());  // HolidayName
            Assert.Equal((byte)0, row[21].Value);                       // IsPaidTimeOff

            Done();
        }
    }
}
