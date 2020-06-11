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
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            var dataView = mlContext.Data.LoadFromEnumerable(GetPointsWithSeasonality());
            // Create a seasonal data as input for DetectSeasonality.
            var input = GetPointsWithSeasonality();
            SeasonalityDetector seasonalityDetector = new SeasonalityDetector();
            int period = mlContext.AnomalyDetection.DetectSeasonality(dataView, "Input");

            // Print the Seasonality Period result.
            Console.WriteLine($"Seasonality Period: #{period}");
        }

        private static IEnumerable<TimeSeriesData> GetPointsWithSeasonality()
        {
            return new List<double>() {18.004, 87.401, 87.411, 18.088, 18.017, 87.759, 33.996, 18.043, 87.853, 18.364, 18.004, 86.992, 87.555,
                18.088, 18.029, 87.906, 87.471, 18.039, 18.099, 87.403, 18.030, 72.991, 87.804, 18.381, 18.016, 87.145, 87.771,
                18.029, 18.084, 87.976, 34.913, 18.064, 18.302, 87.723, 18.001, 86.401, 87.344, 18.295, 18.002, 87.793,
                87.531, 18.055, 18.005, 87.947, 18.003, 72.743, 87.722, 18.142 }.Select(t => new TimeSeriesData(t));
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