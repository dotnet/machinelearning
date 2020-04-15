using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.TimeSeries;

namespace Samples.Dynamic
{
    public static class LocalizeRootCauseByDT
    {
        private static string AGG_SYMBOL = "##SUM##";
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create an empty list as the dataset. The 'DTRootCauseLocalization' API does not
            // require training data as the estimator ('DTRootCauseLocalizationEstimator')
            // created by 'DTRootCauseLocalization' API is not a trainable estimator. The
            // empty list is only needed to pass input schema to the pipeline.
            var emptySamples = new List<RootCauseLocalizationData>();

            // Convert sample list to an empty IDataView.
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

            // A pipeline for localizeing root cause.
            var localizePipeline = mlContext.Transforms.LocalizeRootCauseByDT(nameof(RootCauseLocalizationTransformedData.RootCause), nameof(RootCauseLocalizationData.Input));

            // Fit to data.
            var localizeTransformer = localizePipeline.Fit(emptyDataView);

            // Create the prediction engine to get the root cause result from the input data.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<RootCauseLocalizationData,
                RootCauseLocalizationTransformedData>(localizeTransformer);

            // Call the prediction API.
            DateTime timestamp = GetTimestamp();
            var data = new RootCauseLocalizationData(timestamp, GetAnomalyDimension(), new List<MetricSlice>() { new MetricSlice(timestamp, GetPoints()) }, AggregateType.Sum, AGG_SYMBOL);

            var prediction = predictionEngine.Predict(data);

            // Print the localization result.
            int count = 0;
            foreach (RootCauseItem item in prediction.RootCause.Items)
            {
                count++;
                Console.WriteLine($"Root cause item #{count} ...");
                Console.WriteLine($"Score: {item.Score}, Path: {String.Join(" ",item.Path)}, Direction: {item.Direction}, Dimension:{String.Join(" ", item.Dimension)}");
            }

            //Item #1 ...
            //Score: 0.26670448876705927, Path: DataCenter, Direction: Up, Dimension:[Country, UK] [DeviceType, ##SUM##] [DataCenter, DC1]
        }

        private static List<Point> GetPoints()
        {
            List<Point> points = new List<Point>();

            Dictionary<string, Object> dic1 = new Dictionary<string, Object>();
            dic1.Add("Country", "UK");
            dic1.Add("DeviceType", "Laptop");
            dic1.Add("DataCenter", "DC1");
            points.Add(new Point(200, 100, true, dic1));

            Dictionary<string, Object> dic2 = new Dictionary<string, Object>();
            dic2.Add("Country", "UK");
            dic2.Add("DeviceType", "Mobile");
            dic2.Add("DataCenter", "DC1");
            points.Add(new Point(1000, 100, true, dic2));

            Dictionary<string, Object> dic3 = new Dictionary<string, Object>();
            dic3.Add("Country", "UK");
            dic3.Add("DeviceType", AGG_SYMBOL);
            dic3.Add("DataCenter", "DC1");
            points.Add(new Point(1200, 200, true, dic3));

            Dictionary<string, Object> dic4 = new Dictionary<string, Object>();
            dic4.Add("Country", "UK");
            dic4.Add("DeviceType", "Laptop");
            dic4.Add("DataCenter", "DC2");
            points.Add(new Point(100, 100, false, dic4));

            Dictionary<string, Object> dic5 = new Dictionary<string, Object>();
            dic5.Add("Country", "UK");
            dic5.Add("DeviceType", "Mobile");
            dic5.Add("DataCenter", "DC2");
            points.Add(new Point(200, 200, false, dic5));

            Dictionary<string, Object> dic6 = new Dictionary<string, Object>();
            dic6.Add("Country", "UK");
            dic6.Add("DeviceType", AGG_SYMBOL);
            dic6.Add("DataCenter", "DC2");
            points.Add(new Point(300, 300, false, dic6));

            Dictionary<string, Object> dic7 = new Dictionary<string, Object>();
            dic7.Add("Country", "UK");
            dic7.Add("DeviceType", AGG_SYMBOL);
            dic7.Add("DataCenter", AGG_SYMBOL);
            points.Add(new Point(1500, 500, true, dic7));

            Dictionary<string, Object> dic8 = new Dictionary<string, Object>();
            dic8.Add("Country", "UK");
            dic8.Add("DeviceType", "Laptop");
            dic8.Add("DataCenter", AGG_SYMBOL);
            points.Add(new Point(300, 200, true, dic8));

            Dictionary<string, Object> dic9 = new Dictionary<string, Object>();
            dic9.Add("Country", "UK");
            dic9.Add("DeviceType", "Mobile");
            dic9.Add("DataCenter", AGG_SYMBOL);
            points.Add(new Point(1200, 300, true, dic9));

            return points;
        }

        private static Dictionary<string, Object> GetAnomalyDimension()
        {
            Dictionary<string, Object> dim = new Dictionary<string, Object>();
            dim.Add("Country", "UK");
            dim.Add("DeviceType", AGG_SYMBOL);
            dim.Add("DataCenter", AGG_SYMBOL);

            return dim;
        }

        private static DateTime GetTimestamp()
        {
            return new DateTime(2020, 3, 23, 0, 0, 0);
        }

        private class RootCauseLocalizationData
        {
            [RootCauseLocalizationInputType]
            public RootCauseLocalizationInput Input { get; set; }

            public RootCauseLocalizationData()
            {
                Input = null;
            }

            public RootCauseLocalizationData(DateTime anomalyTimestamp, Dictionary<string, Object> anomalyDimensions, List<MetricSlice> slices, AggregateType aggregateType, string aggregateSymbol)
            {
                Input = new RootCauseLocalizationInput(anomalyTimestamp, anomalyDimensions, slices, aggregateType,
                     aggregateSymbol);
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