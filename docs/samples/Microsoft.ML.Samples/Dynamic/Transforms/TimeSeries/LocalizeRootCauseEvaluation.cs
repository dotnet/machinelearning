using System;
using System.Collections.Generic;
using System.Data;
using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using Microsoft.ML.Transforms.TimeSeries;

using Microsoft.VisualBasic.FileIO;

namespace Samples.Dynamic.Transforms.TimeSeries
{
    public static class LocalizeRootCauseEvaluation
    {
        public static void Example()
        {
            Dictionary<DateTime, Dictionary<string, string>> rootNodeMap = GetAnomalyRootMap();
            Dictionary<DateTime, List<Dictionary<string, string>>> labeledRootCauseMap = GetLabeledRootCauseMap();

            string aggSymbol = "##EMPTY##awqegp##";

            int totalTp = 0;
            int totalFp = 0;
            int totalFn = 0;
            int totalCount = 0;

            bool exactly = false;

            foreach (KeyValuePair<DateTime, Dictionary<string, string>> item in rootNodeMap)
            {
                DateTime timeStamp = item.Key;

                DateTime filterTime = DateTime.ParseExact("2019-11-13 13:00:00,000", "yyyy-MM-dd HH:mm:ss,fff",
                                       System.Globalization.CultureInfo.InvariantCulture);
                if (timeStamp.CompareTo(filterTime).Equals(0))
                {
                    int seconds = Convert.ToInt32(timeStamp.Subtract(new DateTime(1970, 1, 1, 0, 0, 0, 0)).TotalSeconds);
                    string path = String.Format("D:/rootcause/Dataset_yaniv/raw_data_201908_202002/{0}.csv", seconds);
                    List<Point> points = GetPoints(path);
                    List<MetricSlice> slices = new List<MetricSlice>();
                    slices.Add(new MetricSlice(timeStamp, points));

                    PredictionEngine<RootCauseLocalizationData, RootCauseLocalizationTransformedData> engine = GetRootCausePredictionEngine();

                    var newRootCauseInput = new RootCauseLocalizationData(timeStamp, rootNodeMap[timeStamp], new List<MetricSlice>() { new MetricSlice(timeStamp, points) }, DTRootCauseLocalizationEstimator.AggregateType.Sum, aggSymbol);

                    List<RootCauseItem> list = new List<RootCauseItem>();
                    GetRootCause(list, newRootCauseInput, engine);

                    List<Dictionary<string, string>> labeledRootCause = labeledRootCauseMap[timeStamp];
                    List<Dictionary<string, string>> detectedRootCause = ConvertRootCauseItemToDic(list);
                    RemoveAggSymbol(detectedRootCause, aggSymbol);

                    Tuple<int, int, int> evaluation = ScoreRootCause(detectedRootCause, labeledRootCause, exactly, timeStamp);
                    totalTp += evaluation.Item1;
                    totalFp += evaluation.Item2;
                    totalFn += evaluation.Item3;
                    totalCount++;
                }
            }

            double precision = (double)totalTp / (totalTp + totalFp);
            double recall = (double)totalTp / (totalTp + totalFn);
            double f1 = 2 * precision * recall / (precision + recall);
            Console.WriteLine(String.Format("Total Count : {0}, TP: {1}, FP: {2}, FN: {3}", totalCount, totalTp, totalFp, totalFn));
            Console.WriteLine(String.Format("Precision : {0}, Recall: {1}, F1: {2}", precision, recall, f1));
        }

        private static Tuple<int, int, int> ScoreRootCause(List<Dictionary<string, string>> detectedRootCause, List<Dictionary<string, string>> labeledRootCause, bool exactly, DateTime timeStamp)
        {
            int tp = 0;
            int fp = 0;
            int fn; 
            List<string> labelSet = new List<string>();
            foreach (Dictionary<string, string> cause in detectedRootCause)
            {
                string tpCause = FindTruePositive(cause, labeledRootCause, exactly);
                if (tpCause == null)
                {
                    fp++;
                    Console.WriteLine(String.Format("FP : timestamp - {0}, detected root cause ", timeStamp));
                    Console.WriteLine(string.Join(Environment.NewLine, cause));
                    Console.WriteLine(" ");
                }
                else
                {
                    tp++;
                    labelSet.Add(tpCause);
                }
            }

            fn = labeledRootCause.Count - labelSet.Count;
            if (fn != 0)
            {
                List<Dictionary<string, string>> nCause = GetFNegtiveCause(labeledRootCause, labelSet);
                if (nCause.Count > 0)
                {
                    Console.WriteLine(String.Format("FN : timestamp - {0}, labeled root cause", timeStamp));
                    foreach (Dictionary<string, string> cause in nCause)
                    {
                        Console.WriteLine(string.Join(Environment.NewLine, cause));
                        Console.WriteLine("---------------------");
                    }

                }
            }

            return new Tuple<int, int, int>(tp, fp, fn);
        }

        private static List<Dictionary<string, string>> GetFNegtiveCause(List<Dictionary<string, string>> labelCauses, List<string> labelSet)
        {
            List<Dictionary<string, string>> causeList = new List<Dictionary<string, string>>();
            foreach (Dictionary<string, string> cause in labelCauses)
            {
                if (!labelSet.Contains(GetDicHashCode(cause)))
                {
                    causeList.Add(cause);
                }
            }
            return causeList;
        }

        private static string FindTruePositive(Dictionary<string, string> cause, List<Dictionary<string, string>> labelCauses, bool exactly)
        {
            foreach (Dictionary<string, string> label in labelCauses)
            {
                string id = GetDicHashCode(label);
                int compare = CompareCause(cause, label);
                if (compare == 0)
                {
                    return id;
                }
                else if (!exactly && (compare == 1 || compare == 2))
                {
                    return id;
                }
            }
            return null;
        }


        private static string GetDicHashCode(Dictionary<string, string> dic)
        {
            return dic.GetHashCode().ToString();
        }

        private static int CompareCause(Dictionary<string, string> detect, Dictionary<string, string> label)
        {
            if (detect.Equals(label))
            {
                return 0;
            }
            else if (DTRootCauseLocalizationUtils.ContainsAll(detect, label))
            {
                return 1;
            }
            else if (DTRootCauseLocalizationUtils.ContainsAll(label, detect))
            {
                return 2;
            }
            return 3;
        }
        private static List<Dictionary<string, string>> ConvertRootCauseItemToDic(List<RootCauseItem> items)
        {
            List<Dictionary<string, string>> list = new List<Dictionary<string, string>>();
            foreach (RootCauseItem item in items)
            {
                list.Add(item.RootCause);
            }
            return list;
        }

        private static void RemoveAggSymbol(List<Dictionary<string, string>> dimensions, string aggSymbol)
        {
            foreach (Dictionary<string, string> dim in dimensions)
            {
                foreach (string key in dim.Keys)
                {
                    if (dim[key].Equals(aggSymbol))
                    {
                        dim.Remove(key);
                    }
                }
            }
        }

        private static PredictionEngine<RootCauseLocalizationData, RootCauseLocalizationTransformedData> GetRootCausePredictionEngine()
        {
            //// Create an root cause localizatiom input list from csv.
            var rootCauseLocalizationData = new List<RootCauseLocalizationData>() { new RootCauseLocalizationData(new DateTime(), new Dictionary<String, String>(), new List<MetricSlice>() { new MetricSlice(new DateTime(), new List<Microsoft.ML.TimeSeries.Point>()) }, DTRootCauseLocalizationEstimator.AggregateType.Sum, "SUM") };


            var ml = new MLContext(1);
            // Convert the list of root cause data to an IDataView object, which is consumable by ML.NET API.
            var data = ml.Data.LoadFromEnumerable(rootCauseLocalizationData);

            // Create pipeline to localize root cause by decision tree.
            var pipeline = ml.Transforms.LocalizeRootCauseByDT(nameof(RootCauseLocalizationTransformedData.RootCause), nameof(RootCauseLocalizationData.Input));

            // Fit the model.
            var model = pipeline.Fit(data);

            // Test path:  input list -> IDataView -> Enumerable of RootCauseLocalizationInputs.
            var transformedData = model.Transform(data);

            // Load input list in DataView back to Enumerable.
            var transformedDataPoints = ml.Data.CreateEnumerable<RootCauseLocalizationTransformedData>(transformedData, false);

            var engine = ml.Model.CreatePredictionEngine<RootCauseLocalizationData, RootCauseLocalizationTransformedData>(model);
            return engine;
        }

        private static string _ocsDataCenter = "OCSDatacenter";
        private static string _appType = "AppType";
        private static string _releaseAudienceGroup = "Release_AudienceGroup";
        private static string _wacDatacenter = "WACDatacenter";
        private static string _requestType = "RequestType";
        private static string _statusCode = "StatusCode";

        private static List<string> _dimensionKeys = new List<string>() { _ocsDataCenter, _appType, _releaseAudienceGroup, _wacDatacenter, _statusCode, _requestType };

        private static Dictionary<DateTime, Dictionary<string, string>> GetAnomalyRootMap()
        {
            var anomalyRootData = GetDataTabletFromCSVFile("D:/rootcause/Dataset_yaniv/root_cause_201908_202002/anomaly_root.csv");

            Dictionary<DateTime, Dictionary<string, string>> rootNodeMap = new Dictionary<DateTime, Dictionary<string, string>>();
            foreach (DataRow row in anomalyRootData.Rows)
            {
                // load the data, build the RootCauseInput, take care of empty value
                long seconds = long.Parse(row["TimeStamp"].ToString());
                DateTime t = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddSeconds(seconds);

                Dictionary<string, string> dimension = new Dictionary<string, string>();
                foreach (string key in _dimensionKeys)
                {
                    if (!row[key].ToString().Equals(""))
                    {
                        dimension.Add(key, row[key].ToString());
                    }
                }

                rootNodeMap.Add(t, dimension);
            }
            return rootNodeMap;
        }

        private static Dictionary<DateTime, List<Dictionary<string, string>>> GetLabeledRootCauseMap()
        {
            var labeldRootCause = GetDataTabletFromCSVFile("D:/rootcause/Dataset_yaniv/root_cause_201908_202002/labeled_root_cause.csv");

            Dictionary<DateTime, List<Dictionary<string, string>>> map = new Dictionary<DateTime, List<Dictionary<string, string>>>();
            foreach (DataRow row in labeldRootCause.Rows)
            {
                // load the data, build the labled result, take care of empty value
                long seconds = long.Parse(row["TimeStamp"].ToString());
                DateTime t = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddSeconds(seconds);

                Dictionary<string, string> dimension = new Dictionary<string, string>();
                foreach (string key in _dimensionKeys)
                {
                    if (!row[key].ToString().Equals(""))
                    {
                        dimension.Add(key, row[key].ToString());
                    }
                }

                if (map.ContainsKey(t))
                {
                    map[t].Add(dimension);
                }
                else
                {
                    map.Add(t, new List<Dictionary<string, string>>() { dimension });
                }
            }
            return map;
        }

        private static List<Point> GetPoints(string path)
        {


            var inputData = GetDataTabletFromCSVFile(path);

            DateTime timeStamp = new DateTime();

            List<Microsoft.ML.TimeSeries.Point> points = new List<Microsoft.ML.TimeSeries.Point>();
            foreach (DataRow row in inputData.Rows)
            {
                // load the data, build the RootCauseInput, take care of empty value
                long seconds = long.Parse(row["TimeStamp"].ToString());
                timeStamp = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddSeconds(seconds);
                double value = Double.Parse(row["Value"].ToString());
                double expectedValue = 0;
                if (!row["ExpectedValue"].ToString().Equals(""))
                {
                    expectedValue = Double.Parse(row["ExpectedValue"].ToString());
                }
                bool isAnomaly = Boolean.Parse(row["IsAnomaly"].ToString());
                Dictionary<string, string> dimension = new Dictionary<string, string>();
                foreach (string key in _dimensionKeys)
                {
                    if (!row[key].ToString().Equals(""))
                    {
                        dimension.Add(key, row[key].ToString());
                    }
                }

                points.Add(new Microsoft.ML.TimeSeries.Point(value, expectedValue, isAnomaly, dimension)); ;
            }

            return points;
        }

        private static void GetRootCause(List<RootCauseItem> rootCauseList, RootCauseLocalizationData inputData, PredictionEngine<RootCauseLocalizationData, RootCauseLocalizationTransformedData> engine)
        {

            RootCauseLocalizationTransformedData incrementalResult = engine.Predict(inputData);

            if (incrementalResult.RootCause.Items.Count == 0 || (
                incrementalResult.RootCause.Items.Count == 1 && incrementalResult.RootCause.Items[0].RootCause.Equals(inputData.Input.AnomalyDimensions)
                ))
            {
                if (!rootCauseList.Contains(new RootCauseItem(inputData.Input.AnomalyDimensions)))
                {
                    rootCauseList.Add(new RootCauseItem(inputData.Input.AnomalyDimensions));

                }
                return;
            }
            else
            {
                foreach (RootCauseItem item in incrementalResult.RootCause.Items)
                {
                    RootCauseLocalizationData newData = new RootCauseLocalizationData(inputData.Input.AnomalyTimestamp,
                       item.RootCause, inputData.Input.Slices, inputData.Input.AggType, inputData.Input.AggSymbol);
                    GetRootCause(rootCauseList, newData, engine);
                }
            }
        }

        private static DataTable GetDataTabletFromCSVFile(string filePath)
        {
            DataTable csvData = new DataTable();


            using (TextFieldParser csvReader = new TextFieldParser(filePath))
            {
                csvReader.SetDelimiters(new string[] { "," });
                csvReader.HasFieldsEnclosedInQuotes = true;
                string[] colFields = csvReader.ReadFields();
                foreach (string column in colFields)
                {
                    DataColumn datecolumn = new DataColumn(column);
                    datecolumn.AllowDBNull = true;
                    csvData.Columns.Add(datecolumn);
                }

                while (!csvReader.EndOfData)
                {
                    string[] fieldData = csvReader.ReadFields();
                    //Making empty value as null
                    for (int i = 0; i < fieldData.Length; i++)
                    {
                        if (fieldData[i] == "")
                        {
                            fieldData[i] = null;
                        }
                    }
                    csvData.Rows.Add(fieldData);
                }
            }

            return csvData;
        }

        private class RootCauseLocalizationData
        {
            [RootCauseLocalizationInputType]
            public RootCauseLocalizationInput Input { get; set; }

            public RootCauseLocalizationData()
            {
                Input = null;
            }

            public RootCauseLocalizationData(DateTime anomalyTimestamp, Dictionary<string, string> anomalyDimensions, List<MetricSlice> slices, DTRootCauseLocalizationEstimator.AggregateType aggregateteType, string aggregateSymbol)
            {
                Input = new RootCauseLocalizationInput(anomalyTimestamp, anomalyDimensions, slices, aggregateteType, aggregateSymbol);
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
