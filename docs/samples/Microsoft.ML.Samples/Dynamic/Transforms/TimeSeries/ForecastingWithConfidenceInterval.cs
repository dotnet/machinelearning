using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using Microsoft.ML.TimeSeries;

namespace Samples.Dynamic
{
    public static class ForecastingWithConfidenceInternal
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

            // Train the forecasting model.
            var model = ml.Forecasting.AdaptiveSingularSpectrumSequenceModeler(inputColumnName, data.Count, SeasonalitySize + 1, SeasonalitySize,
                1, AdaptiveSingularSpectrumSequenceModeler.RankSelectionMethod.Exact, null, SeasonalitySize / 2, shouldComputeForecastIntervals: true, false);

            // Train.
            model.Train(dataView);

            // Forecast next five values with confidence internal.
            float[] forecast;
            float[] confidenceIntervalLowerBounds;
            float[] confidenceIntervalUpperBounds;
            model.ForecastWithConfidenceIntervals(5, out forecast, out confidenceIntervalLowerBounds, out confidenceIntervalUpperBounds);
            Console.WriteLine($"Forecasted values:");
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            Console.WriteLine($"Confidence intervals:");
            for(int index = 0; index < 5; index++)
                Console.Write($"[{0} - {1}] ", confidenceIntervalLowerBounds[index], confidenceIntervalUpperBounds[index]);

            // Forecasted values:
            // [2.452744, 2.589339, 2.729183, 2.873005, 3.028931]
            // Confidence intervals:
            // [0 - 1] [0 - 1] [0 - 1] [0 - 1] [0 - 1]

            // Update with new observations.
            dataView = ml.Data.LoadFromEnumerable(new List<TimeSeriesData>() { new TimeSeriesData(0), new TimeSeriesData(0), new TimeSeriesData(0), new TimeSeriesData(0) });
            model.Update(dataView);

            // Checkpoint.
            ml.Model.SaveForecastingModel(model, "model.zip");

            // Load the checkpointed model from disk.
            var modelCopy = ml.Model.LoadForecastingModel<float>("model.zip");

            // Forecast with the checkpointed model loaded from disk.
            float[] forecastCopy;
            float[] confidenceIntervalLowerBoundsCopy;
            float[] confidenceIntervalUpperBoundsCopy;
            modelCopy.ForecastWithConfidenceIntervals(5, out forecastCopy, out confidenceIntervalLowerBoundsCopy, out confidenceIntervalUpperBoundsCopy);
            Console.WriteLine($"Forecasted values:");
            Console.WriteLine("[{0}]", string.Join(", ", forecastCopy));
            Console.WriteLine($"Confidence intervals:");
            for (int index = 0; index < 5; index++)
                Console.Write($"[{0} - {1}] ", confidenceIntervalLowerBoundsCopy[index], confidenceIntervalUpperBoundsCopy[index]);

            // Forecasted values:
            // [0.8681176, 0.8185108, 0.8069275, 0.84405, 0.9455081]
            // Confidence intervals:
            // [0 - 1] [0 - 1] [0 - 1] [0 - 1] [0 - 1]

            // Forecast with the original model(that was checkpointed to disk).
            model.ForecastWithConfidenceIntervals(5, out forecast, out confidenceIntervalLowerBounds, out confidenceIntervalUpperBounds);
            Console.WriteLine($"Forecasted values:");
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            Console.WriteLine($"Confidence intervals:");
            for (int index = 0; index < 5; index++)
                Console.Write($"[{0} - {1}] ", confidenceIntervalLowerBounds[index], confidenceIntervalUpperBounds[index]);

            // Forecasted values:
            // [0.8681176, 0.8185108, 0.8069275, 0.84405, 0.9455081]
            // Confidence intervals:
            // [0 - 1] [0 - 1] [0 - 1] [0 - 1] [0 - 1]
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
