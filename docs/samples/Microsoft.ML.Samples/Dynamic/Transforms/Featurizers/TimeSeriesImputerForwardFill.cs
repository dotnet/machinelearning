using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;

namespace Samples.Dynamic
{
    public static class TimeSeriesImputerForwardFill
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new[] { new InputData() { Date = 0, GrainA = "A", DataA = 2.0f },
                new InputData() { Date = 1, GrainA = "A", DataA = float.NaN },
                new InputData() { Date = 3, GrainA = "A", DataA = 5.0f },
                new InputData() { Date = 5, GrainA = "A", DataA = float.NaN },
                new InputData() { Date = 7, GrainA = "A", DataA = float.NaN }};

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for imputing the missing rows and values in the columns using the default "ForwardFill" strategy.
            var pipeline = mlContext.Transforms.TimeSeriesImputer("Date", new string[] { "GrainA" });

            // The transformed data.
            var transformedData = pipeline.Fit(dataview).Transform(dataview);

            // Now let's take a look at what this did. The NaN values should be filled in with last value that was not NaN,
            // and rows should be created to fill in the missing gaps in the time column.
            // We can extract the newly created columns as an IEnumerable of TransformedData.
            var featuresColumn = mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false);

            // And we can write out a few rows
            Console.WriteLine($"Features column obtained post-transformation.");
            foreach (var featureRow in featuresColumn)
                Console.WriteLine(featureRow.Date + ", " + featureRow.GrainA + ", " + featureRow.DataA + ", " + featureRow.IsRowImputed);

            // Expected output:
            //  Features column obtained post-transformation.
            //  0, A, 2.0, false
            //  1, A, 2.0, false
            //  2, A, 2.0, true
            //  3, A, 5.0, false
            //  4, A, 5.0, true
            //  5, A, 5.0, false
            //  6, A, 5.0, true
            //  7, A, 5.0, false
        }

        private class InputData
        {
            public long Date;
            public string GrainA;
            public float DataA;
        }

        private sealed class TransformedData
        {
            public long Date { get; set; }
            public string GrainA { get; set; }
            public float DataA { get; set; }
            public bool IsRowImputed { get; set; }
        }
    }
}
