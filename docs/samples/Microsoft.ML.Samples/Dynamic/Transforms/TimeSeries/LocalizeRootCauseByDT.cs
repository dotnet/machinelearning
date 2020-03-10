using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class LocalizeRootCause
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create an empty list as the dataset. The 'NormalizeText' API does not
            // require training data as the estimator ('TextNormalizingEstimator')
            // created by 'NormalizeText' API is not a trainable estimator. The
            // empty list is only needed to pass input schema to the pipeline.
            var emptySamples = new List<RootCauseLocalizationData>();

            // Convert sample list to an empty IDataView.
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

            // A pipeline for localizeing root cause.
            var localizePipeline = mlContext.Transforms.LocalizeRootCauseByDT(nameof(RootCauseLocalizationTransformedData.RootCause), nameof(RootCauseLocalizationData.Input));

            // Fit to data.
            var localizeTransformer = localizePipeline.Fit(emptyDataView);

            // Create the prediction engine to get the root cause result from the
            // input data.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<RootCauseLocalizationData,
                RootCauseLocalizationTransformedData>(localizeTransformer);

            // Call the prediction API.
            var data = new RootCauseLocalizationData(new DateTime(), new Dictionary<String, String>(), new List<MetricSlice>() { new MetricSlice(new DateTime(), new List<Microsoft.ML.Transforms.TimeSeries.Point>()) }, "SUM", "SUM");

            var prediction = predictionEngine.Predict(data);

            // Print the localization result.
            Console.WriteLine($"Localized result: {prediction.RootCause}");
        }

      
        private class RootCauseLocalizationData
        {
            [RootCauseLocalizationInputType]
            public RootCauseLocalizationInput Input { get; set; }

            public RootCauseLocalizationData()
            {
                Input = null;
            }

            public RootCauseLocalizationData(DateTime anomalyTimestamp, Dictionary<string, string> anomalyDimensions, List<MetricSlice> slices,String aggregateType, string aggregateSymbol)
            {
                Input = new RootCauseLocalizationInput(anomalyTimestamp, anomalyDimensions, slices, DTRootCauseLocalizationEstimator.AggregateType.Sum, aggregateSymbol);
            }
        }

        private class RootCauseLocalizationTransformedData
        {
            [RootCauseType()]
            public RootCause RootCause { get; set; }

            public RootCauseLocalizationTransformedData()
            {
                RootCause = null;
            }
        }
    }
}
