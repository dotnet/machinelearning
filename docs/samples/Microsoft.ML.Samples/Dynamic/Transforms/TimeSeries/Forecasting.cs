using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class Forecasting
    {
        // This example creates a time series (list of Data with the i-th element corresponding to the i-th time slot) and then
        // does forecasting.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
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

            // Instantiate the forecasting model.
            var model = ml.Forecasting.ForecastBySsa(inputColumnName, inputColumnName, 5, 11, data.Count, 5);

            // Train.
            var transformer = model.Fit(dataView);

            // Forecast next five values.
            var forecastEngine = transformer.CreateForecastingEngine<TimeSeriesData>(ml);
            var forecast = ml.Data.CreateEnumerable<ForecastResult>(forecastEngine.Forecast(5), false);
            Console.WriteLine($"Forecasted values:");
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            // Forecasted values:
            // [2.452744, 2.589339, 2.729183, 2.873005, 3.028931]

            // Update with new observations.
            forecastEngine.Update(new TimeSeriesData(0));
            forecastEngine.Update(new TimeSeriesData(0));
            forecastEngine.Update(new TimeSeriesData(0));
            forecastEngine.Update(new TimeSeriesData(0));

            // Checkpoint.
            forecastEngine.CheckPoint(ml, "model.zip");

            // Load the checkpointed model from disk.
            // Load the model.
            ITransformer modelCopy;
            using (var file = File.OpenRead("model.zip"))
                modelCopy = ml.Model.Load(file, out DataViewSchema schema);

            // We must create a new prediction engine from the persisted model.
            var forecastEngineCopy = modelCopy.CreateForecastingEngine<TimeSeriesData>(ml);

            // Forecast with the checkpointed model loaded from disk.
            forecast = ml.Data.CreateEnumerable<ForecastResult>(forecastEngineCopy.Forecast(5), false);
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            // [0.8681176, 0.8185108, 0.8069275, 0.84405, 0.9455081]

            // Forecast with the original model(that was checkpointed to disk).
            forecast = ml.Data.CreateEnumerable<ForecastResult>(forecastEngine.Forecast(5), false);
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            // [0.8681176, 0.8185108, 0.8069275, 0.84405, 0.9455081]

        }

        class ForecastResult
        {
#pragma warning disable CS0649
            public float Forecast;
#pragma warning restore CS0649
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
