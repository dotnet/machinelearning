using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;

namespace Samples.Dynamic
{
    public static class DateTimeTransformer
    {
        private class DateTimeInput
        {
            public long Date;
        }

        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            // Future Date - 2025 June 30
            var samples = new[] { new DateTimeInput() { Date = 1751241600 } };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for splitting the time features into individual columns
            var pipeline = mlContext.Transforms.FeaturizeDateTime("Date", "DTC");

            // The transformed data.
            var transformedData = pipeline.Fit(dataview).Transform(dataview);

            // Now let's take a look at what this did. We should have created 21 more columns with all the
            // DateTime information split into its own columns
            var featuresColumn = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            // And we can write out a few rows
            Console.WriteLine($"Features column obtained post-transformation.");
            foreach (var featureRow in featuresColumn)
                Console.WriteLine(featureRow.Date + ", " + featureRow.DTCYear + ", " + featureRow.DTCMonth + ", " +
                    featureRow.DTCDay + ", " + featureRow.DTCHour + ", " + featureRow.DTCMinute + ", " +
                    featureRow.DTCSecond + ", " + featureRow.DTCAmPm + ", " + featureRow.DTCHour12 + ", " +
                    featureRow.DTCDayOfWeek + ", " + featureRow.DTCDayOfQuarter + ", " + featureRow.DTCDayOfYear +
                    ", " + featureRow.DTCWeekOfMonth + ", " + featureRow.DTCQuarterOfYear + ", " + featureRow.DTCHalfOfYear +
                    ", " + featureRow.DTCWeekIso + ", " + featureRow.DTCYearIso + ", " + featureRow.DTCMonthLabel + ", " +
                    featureRow.DTCAmPmLabel + ", " + featureRow.DTCDayOfWeekLabel + ", " + featureRow.DTCHolidayName + ", " +
                    featureRow.DTCIsPaidTimeOff);

            // Expected output:
            //  Features columns obtained post-transformation.
            //  1751241600, 2025, 6, 30, 0, 0, 0, 0, 0, 1, 91, 180, 4, 2, 1, 27, 2025, June, am, Monday, , 0
        }

        // These columns start with DTC because that is the prefix we picked
        private sealed class TransformedData
        {
            public long Date { get; set; }
            public int DTCYear { get; set; }
            public byte DTCMonth { get; set; }
            public byte DTCDay { get; set; }
            public byte DTCHour { get; set; }
            public byte DTCMinute { get; set; }
            public byte DTCSecond { get; set; }
            public byte DTCAmPm { get; set; }
            public byte DTCHour12 { get; set; }
            public byte DTCDayOfWeek { get; set; }
            public byte DTCDayOfQuarter { get; set; }
            public ushort DTCDayOfYear { get; set; }
            public ushort DTCWeekOfMonth { get; set; }
            public byte DTCQuarterOfYear { get; set; }
            public byte DTCHalfOfYear { get; set; }
            public byte DTCWeekIso { get; set; }
            public int DTCYearIso { get; set; }
            public string DTCMonthLabel { get; set; }
            public string DTCAmPmLabel { get; set; }
            public string DTCDayOfWeekLabel { get; set; }
            public string DTCHolidayName { get; set; }
            public byte DTCIsPaidTimeOff { get; set; }
        }
    }
}
