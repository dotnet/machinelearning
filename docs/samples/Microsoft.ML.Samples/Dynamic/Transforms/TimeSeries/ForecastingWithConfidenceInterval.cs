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

            // Instantiate forecasting model.
            var model = ml.Forecasting.AdaptiveSingularSpectrumSequenceModeler(inputColumnName, data.Count, SeasonalitySize + 1, SeasonalitySize,
                1, AdaptiveSingularSpectrumSequenceModeler.RankSelectionMethod.Exact, null, SeasonalitySize / 2, shouldComputeForecastIntervals: true, false);

            // Train.
            model.Train(dataView);

            // Forecast next five values with confidence internal.
            float[] forecast;
            float[] confidenceIntervalLowerBounds;
            float[] confidenceIntervalUpperBounds;
            model.ForecastWithConfidenceIntervals(5, out forecast, out confidenceIntervalLowerBounds, out confidenceIntervalUpperBounds);
            PrintForecastValuesAndIntervals(forecast, confidenceIntervalLowerBounds, confidenceIntervalUpperBounds);
            // Forecasted values:
            // [2.452744, 2.589339, 2.729183, 2.873005, 3.028931]
            // Confidence intervals:
            // [-0.2235315 - 5.12902] [-0.08777174 - 5.266451] [0.05076938 - 5.407597] [0.1925406 - 5.553469] [0.3469928 - 5.71087]

            // Update with new observations.
            dataView = ml.Data.LoadFromEnumerable(new List<TimeSeriesData>() { new TimeSeriesData(0), new TimeSeriesData(0), new TimeSeriesData(0), new TimeSeriesData(0) });
            model.Update(dataView);

            // Checkpoint.
            ml.Model.SaveForecastingModel(model, "model.zip");

            // Load the checkpointed model from disk.
            var modelCopy = ml.Model.LoadForecastingModel<float>("model.zip");

            // Forecast with the checkpointed model loaded from disk.
            modelCopy.ForecastWithConfidenceIntervals(5, out forecast, out confidenceIntervalLowerBounds, out confidenceIntervalUpperBounds);
            PrintForecastValuesAndIntervals(forecast, confidenceIntervalLowerBounds, confidenceIntervalUpperBounds);
            // Forecasted values:
            // [0.8681176, 0.8185108, 0.8069275, 0.84405, 0.9455081]
            // Confidence intervals:
            // [-1.808158 - 3.544394] [-1.8586 - 3.495622] [-1.871486 - 3.485341] [-1.836414 - 3.524514] [-1.736431 - 3.627447]

            // Forecast with the original model(that was checkpointed to disk).
            model.ForecastWithConfidenceIntervals(5, out forecast, out confidenceIntervalLowerBounds, out confidenceIntervalUpperBounds);
            PrintForecastValuesAndIntervals(forecast, confidenceIntervalLowerBounds, confidenceIntervalUpperBounds);
            // Forecasted values:
            // [0.8681176, 0.8185108, 0.8069275, 0.84405, 0.9455081]
            // Confidence intervals:
            // [-1.808158 - 3.544394] [-1.8586 - 3.495622] [-1.871486 - 3.485341] [-1.836414 - 3.524514] [-1.736431 - 3.627447]
        }

        static void PrintForecastValuesAndIntervals(float[] forecast, float[] confidenceIntervalLowerBounds, float[] confidenceIntervalUpperBounds)
        {
            Console.WriteLine($"Forecasted values:");
            Console.WriteLine("[{0}]", string.Join(", ", forecast));
            Console.WriteLine($"Confidence intervals:");
            for (int index = 0; index < forecast.Length; index++)
                Console.Write($"[{confidenceIntervalLowerBounds[index]} - {confidenceIntervalUpperBounds[index]}] ");
            Console.WriteLine();
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
