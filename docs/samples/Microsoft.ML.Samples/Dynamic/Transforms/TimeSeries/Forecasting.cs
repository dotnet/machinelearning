using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using Microsoft.ML.TimeSeries;

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

            // Generate sample series data with a recurring pattern
            const int SeasonalitySize = 5;
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
            var model = ml.Forecasting.AdaptiveSingularSpectrumSequenceModeler(inputColumnName, data.Count, SeasonalitySize + 1, SeasonalitySize,
                1, AdaptiveSingularSpectrumSequenceModeler.RankSelectionMethod.Exact, null, SeasonalitySize / 2, false, false);

            // Train.
            model.Train(dataView);

            // Forecast next five values.
            var forecast = model.Forecast(5);
            Console.WriteLine($"Forecasted values:");
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            // Forecasted values:
            // [2.452744, 2.589339, 2.729183, 2.873005, 3.028931]

            // Update with new observations.
            dataView = ml.Data.LoadFromEnumerable(new List<TimeSeriesData>() { new TimeSeriesData(0), new TimeSeriesData(0), new TimeSeriesData(0), new TimeSeriesData(0) });
            model.Update(dataView);

            // Checkpoint.
            ml.Model.SaveForecastingModel(model, "model.zip");

            // Load the checkpointed model from disk.
            var modelCopy = ml.Model.LoadForecastingModel<float>("model.zip");

            // Forecast with the checkpointed model loaded from disk.
            forecast = modelCopy.Forecast(5);
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            // [0.8681176, 0.8185108, 0.8069275, 0.84405, 0.9455081]

            // Forecast with the original model(that was checkpointed to disk).
            forecast = model.Forecast(5);
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            // [0.8681176, 0.8185108, 0.8069275, 0.84405, 0.9455081]

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
