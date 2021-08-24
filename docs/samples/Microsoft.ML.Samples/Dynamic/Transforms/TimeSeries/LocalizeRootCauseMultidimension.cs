using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.TimeSeries;

namespace Samples.Dynamic
{
    public static class LocalizeRootCauseMultipleDimensions
    {
        private static string AGG_SYMBOL = "##SUM##";

        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create an root cause localization input instance.
            DateTime timestamp = GetTimestamp();
            var data = new RootCauseLocalizationInput(timestamp, GetAnomalyDimension(), new List<MetricSlice>() { new MetricSlice(timestamp, GetTimeSeriesPoints()) }, AggregateType.Sum, AGG_SYMBOL);

            // Get the root cause localization result.
            List<RootCause> prediction = mlContext.AnomalyDetection.LocalizeRootCauses(data);

            // Print the localization results.
            int count = 0;
            foreach (RootCause cause in prediction)
            {
                count++;
                foreach (RootCauseItem item in cause.Items)
                {
                    Console.WriteLine($"Prepared cause #{count} ...");
                    Console.WriteLine($"Score: {item.Score}, Path: {String.Join(" ", item.Path)}, Direction: {item.Direction}, Dimension:{String.Join(" ", item.Dimension)}");
                }
            }

            //Prepared cause #1 ...
            //Score: 0.26670448876705927, Path: DataCenter, Direction: Up, Dimension:[Country, UK] [DeviceType, ##SUM##] [DataCenter, DC1]
            //Prepared cause #2 ...
            //Score: 0.254746585094852, Path: DeviceType, Direction: Up, Dimension:[Country, UK] [DeviceType, Laptop] [DataCenter, ##SUM##]        
        }

        private static List<TimeSeriesPoint> GetTimeSeriesPoints()
        {
            List<TimeSeriesPoint> TimeSeriesPoints = new List<TimeSeriesPoint>();

            Dictionary<string, Object> dic1 = new Dictionary<string, Object>
            {
                { "Country", "UK" },
                { "DeviceType", "Laptop" },
                { "DataCenter", "DC1" }
            };
            TimeSeriesPoints.Add(new TimeSeriesPoint(200, 100, true, dic1));

            Dictionary<string, Object> dic2 = new Dictionary<string, Object>();
            dic2.Add("Country", "UK");
            dic2.Add("DeviceType", "Mobile");
            dic2.Add("DataCenter", "DC1");
            TimeSeriesPoints.Add(new TimeSeriesPoint(1000, 100, true, dic2));

            Dictionary<string, Object> dic3 = new Dictionary<string, Object>();
            dic3.Add("Country", "UK");
            dic3.Add("DeviceType", AGG_SYMBOL);
            dic3.Add("DataCenter", "DC1");
            TimeSeriesPoints.Add(new TimeSeriesPoint(1200, 200, true, dic3));

            Dictionary<string, Object> dic4 = new Dictionary<string, Object>();
            dic4.Add("Country", "UK");
            dic4.Add("DeviceType", "Laptop");
            dic4.Add("DataCenter", "DC2");
            TimeSeriesPoints.Add(new TimeSeriesPoint(100, 100, false, dic4));

            Dictionary<string, Object> dic5 = new Dictionary<string, Object>();
            dic5.Add("Country", "UK");
            dic5.Add("DeviceType", "Mobile");
            dic5.Add("DataCenter", "DC2");
            TimeSeriesPoints.Add(new TimeSeriesPoint(200, 200, false, dic5));

            Dictionary<string, Object> dic6 = new Dictionary<string, Object>();
            dic6.Add("Country", "UK");
            dic6.Add("DeviceType", AGG_SYMBOL);
            dic6.Add("DataCenter", "DC2");
            TimeSeriesPoints.Add(new TimeSeriesPoint(300, 300, false, dic6));

            Dictionary<string, Object> dic7 = new Dictionary<string, Object>();
            dic7.Add("Country", "UK");
            dic7.Add("DeviceType", AGG_SYMBOL);
            dic7.Add("DataCenter", AGG_SYMBOL);
            TimeSeriesPoints.Add(new TimeSeriesPoint(1800, 750, true, dic7));

            Dictionary<string, Object> dic8 = new Dictionary<string, Object>();
            dic8.Add("Country", "UK");
            dic8.Add("DeviceType", "Laptop");
            dic8.Add("DataCenter", AGG_SYMBOL);
            TimeSeriesPoints.Add(new TimeSeriesPoint(1500, 450, true, dic8));

            Dictionary<string, Object> dic9 = new Dictionary<string, Object>();
            dic9.Add("Country", "UK");
            dic9.Add("DeviceType", "Mobile");
            dic9.Add("DataCenter", AGG_SYMBOL);
            TimeSeriesPoints.Add(new TimeSeriesPoint(600, 550, false, dic9));

            Dictionary<string, Object> dic10 = new Dictionary<string, Object>();
            dic10.Add("Country", "UK");
            dic10.Add("DeviceType", "Mobile");
            dic10.Add("DataCenter", "DC3");
            TimeSeriesPoints.Add(new TimeSeriesPoint(100, 100, false, dic10));

            Dictionary<string, Object> dic11 = new Dictionary<string, Object>();
            dic11.Add("Country", "UK");
            dic11.Add("DeviceType", "Laptop");
            dic11.Add("DataCenter", "DC3");
            TimeSeriesPoints.Add(new TimeSeriesPoint(200, 250, false, dic11));

            Dictionary<string, Object> dic12 = new Dictionary<string, Object>();
            dic12.Add("Country", "UK");
            dic12.Add("DeviceType", AGG_SYMBOL);
            dic12.Add("DataCenter", "DC3");
            TimeSeriesPoints.Add(new TimeSeriesPoint(300, 350, false, dic12));

            return TimeSeriesPoints;
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
    }
}
