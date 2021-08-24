using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.TimeSeries;

namespace Samples.Dynamic
{
    public static class DetectSeasonality
    {
        public static void Example()
        {
            /* Create a new ML context, for ML.NET operations. It can be used for
             exception tracking and logging, as well as the source of randomness.*/
            var mlContext = new MLContext();

            // Create a seasonal data as input: y = sin(2 * Pi + x)
            var seasonalData = Enumerable.Range(0, 100).Select(x => new TimeSeriesData(Math.Sin(2 * Math.PI + x)));

            // Load the input data as a DataView.
            var dataView = mlContext.Data.LoadFromEnumerable(seasonalData);

            /* Two option parameters:
             * seasonalityWindowSize: Default value is -1. When set to -1, use the whole input to fit model; 
             * when set to a positive integer, only the first windowSize number of values will be considered.
             * randomnessThreshold: Randomness threshold that specifies how confidence the input values follows 
             * a predictable pattern recurring as seasonal data. By default, it is set as 0.99. 
             * The higher the threshold is set, the more strict recurring pattern the 
             * input values should follow to be determined as seasonal data.
             */
            int period = mlContext.AnomalyDetection.DetectSeasonality(
                dataView,
                nameof(TimeSeriesData.Value),
                seasonalityWindowSize: 40);

            // Print the Seasonality Period result.
            Console.WriteLine($"Seasonality Period: #{period}");
        }

        private class TimeSeriesData
        {
            public double Value;

            public TimeSeriesData(double value)
            {
                Value = value;
            }
        }

    }
}
