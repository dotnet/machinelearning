using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class ForecastingWithConfidenceInternal
    {
        // This example creates a time series (list of Data with the i-th element
        // corresponding to the i-th time slot) and then does forecasting.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a recurring pattern.
            var data = new List<TimeSeriesData>()
            {
                new TimeSeriesData(0),
                new TimeSeriesData(1),
                new TimeSeriesData(2),
                new TimeSeriesData(3),
                new TimeSeriesData(4),

                new TimeSeriesData(0),
                new TimeSeriesData(1),
                new TimeSeriesData(2),
                new TimeSeriesData(3),
                new TimeSeriesData(4),

                new TimeSeriesData(0),
                new TimeSeriesData(1),
                new TimeSeriesData(2),
                new TimeSeriesData(3),
                new TimeSeriesData(4),
            };

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup arguments.
            var inputColumnName = nameof(TimeSeriesData.Value);
            var outputColumnName = nameof(ForecastResult.Forecast);

            // Instantiate the forecasting model.
            var model = ml.Forecasting.ForecastBySsa(outputColumnName,
                inputColumnName, 5, 11, data.Count, 5,
                confidenceLevel: 0.95f,
                confidenceLowerBoundColumn: "ConfidenceLowerBound",
                confidenceUpperBoundColumn: "ConfidenceUpperBound");

            // Train.
            var transformer = model.Fit(dataView);

            // Forecast next five values.
            var forecastEngine = transformer.CreateTimeSeriesEngine<TimeSeriesData,
                ForecastResult>(ml);

            var forecast = forecastEngine.Predict();

            PrintForecastValuesAndIntervals(forecast.Forecast, forecast
                .ConfidenceLowerBound, forecast.ConfidenceUpperBound);
            // Forecasted values:
            // [1.977226, 1.020494, 1.760543, 3.437509, 4.266461]
            // Confidence intervals:
            // [0.3451088 - 3.609343] [-0.7967533 - 2.83774] [-0.058467 - 3.579552] [1.61505 - 5.259968] [2.349299 - 6.183623]

            // Update with new observations.
            forecastEngine.Predict(new TimeSeriesData(0));
            forecastEngine.Predict(new TimeSeriesData(0));
            forecastEngine.Predict(new TimeSeriesData(0));
            forecastEngine.Predict(new TimeSeriesData(0));

            // Checkpoint.
            forecastEngine.CheckPoint(ml, "model.zip");

            // Load the checkpointed model from disk.
            // Load the model.
            ITransformer modelCopy;
            using (var file = File.OpenRead("model.zip"))
                modelCopy = ml.Model.Load(file, out DataViewSchema schema);

            // We must create a new prediction engine from the persisted model.
            var forecastEngineCopy = modelCopy.CreateTimeSeriesEngine<
                TimeSeriesData, ForecastResult>(ml);

            // Forecast with the checkpointed model loaded from disk.
            forecast = forecastEngineCopy.Predict();
            PrintForecastValuesAndIntervals(forecast.Forecast, forecast
                .ConfidenceLowerBound, forecast.ConfidenceUpperBound);

            // [1.791331, 1.255525, 0.3060154, -0.200446, 0.5657795]
            // Confidence intervals:
            // [0.1592142 - 3.423448] [-0.5617217 - 3.072772] [-1.512994 - 2.125025] [-2.022905 - 1.622013] [-1.351382 - 2.482941]

            // Forecast with the original model(that was checkpointed to disk).
            forecast = forecastEngine.Predict();
            PrintForecastValuesAndIntervals(forecast.Forecast,
                forecast.ConfidenceLowerBound, forecast.ConfidenceUpperBound);

            // [1.791331, 1.255525, 0.3060154, -0.200446, 0.5657795]
            // Confidence intervals:
            // [0.1592142 - 3.423448] [-0.5617217 - 3.072772] [-1.512994 - 2.125025] [-2.022905 - 1.622013] [-1.351382 - 2.482941]
        }

        static void PrintForecastValuesAndIntervals(float[] forecast, float[]
            confidenceIntervalLowerBounds, float[] confidenceIntervalUpperBounds)
        {
            Console.WriteLine($"Forecasted values:");
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            Console.WriteLine($"Confidence intervals:");
            for (int index = 0; index < forecast.Length; index++)
                Console.Write($"[{confidenceIntervalLowerBounds[index]} -" +
                    $" {confidenceIntervalUpperBounds[index]}] ");
            Console.WriteLine();
        }

        class ForecastResult
        {
            public float[] Forecast { get; set; }
            public float[] ConfidenceLowerBound { get; set; }
            public float[] ConfidenceUpperBound { get; set; }
        }

        class TimeSeriesData
        {
            public float Value;

            public TimeSeriesData(float value)
            {
                Value = value;
            }
        }
    }
}
