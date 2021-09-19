using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class Forecasting
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
                inputColumnName, 5, 11, data.Count, 5);

            // Train.
            var transformer = model.Fit(dataView);

            // Forecast next five values.
            var forecastEngine = transformer.CreateTimeSeriesEngine<TimeSeriesData,
                ForecastResult>(ml);

            var forecast = forecastEngine.Predict();

            Console.WriteLine($"Forecasted values:");
            Console.WriteLine("[{0}]", string.Join(", ", forecast.Forecast));
            // Forecasted values:
            // [1.977226, 1.020494, 1.760543, 3.437509, 4.266461]

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
            Console.WriteLine("[{0}]", string.Join(", ", forecast.Forecast));
            // [1.791331, 1.255525, 0.3060154, -0.200446, 0.5657795]

            // Forecast with the original model(that was checkpointed to disk).
            forecast = forecastEngine.Predict();
            Console.WriteLine("[{0}]", string.Join(", ", forecast.Forecast));
            // [1.791331, 1.255525, 0.3060154, -0.200446, 0.5657795]

        }

        class ForecastResult
        {
            public float[] Forecast { get; set; }
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
