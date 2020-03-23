// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.TimeSeries
{
    public class DTRootCauseAnalyzer
    {
        private RootCauseLocalizationInput _src;
        private double _beta;
        public DTRootCauseAnalyzer(RootCauseLocalizationInput src, double beta)
        {
            _src = src;
            _beta = beta;
        }

        public RootCause Analyze()
        {
            RootCause dst = new RootCause();
            DimensionInfo dimensionInfo = SeperateDimension(_src.AnomalyDimensions, _src.AggSymbol);
            if (dimensionInfo.AggDim.Count == 0)
            {
                dst.Items.Add(new RootCauseItem(_src.AnomalyDimensions));
            }
            Dictionary<string, string> subDim = GetSubDim(_src.AnomalyDimensions, dimensionInfo.DetailDim);
            List<Point> totalPoints = GetTotalPointsForAnomalyTimestamp(_src, subDim);

            GetRootCauseList(_src, ref dst, dimensionInfo, totalPoints, subDim);
            UpdateRootCauseDirection(totalPoints, ref dst);
            return dst;
        }

        public List<Point> GetTotalPointsForAnomalyTimestamp(RootCauseLocalizationInput src, Dictionary<string, string> subDim)
        {
            List<Point> points = new List<Point>();
            foreach (MetricSlice slice in src.Slices)
            {
                if (slice.TimeStamp.Equals(src.AnomalyTimestamp))
                {
                    points = slice.Points;
                }
            }

            List<Point> totalPoints = SelectPoints(points, subDim);

            return totalPoints;
        }

        public void GetRootCauseList(RootCauseLocalizationInput src, ref RootCause dst, DimensionInfo dimensionInfo, List<Point> totalPoints, Dictionary<string, string> subDim)
        {
            PointTree pointTree = BuildPointTree(totalPoints, dimensionInfo.AggDim, subDim, src.AggSymbol, src.AggType);
            PointTree anomalyTree = BuildPointTree(totalPoints, dimensionInfo.AggDim, subDim, src.AggSymbol, src.AggType, true);

            //which means there is no aggregation in the input anomaly dimension
            if (anomalyTree.ParentNode == null)
            {
                return;
            }

            List<RootCauseItem> rootCauses = new List<RootCauseItem>();
            // no point under anomaly dimension
            if (totalPoints.Count == 0)
            {
                if (anomalyTree.Leaves.Count != 0)
                {
                    throw new Exception("point leaves not match with anomaly leaves");
                }

                rootCauses.Add(new RootCauseItem(src.AnomalyDimensions));
            }
            else
            {
                double totalEntropy = 1;
                if (anomalyTree.Leaves.Count > 0)
                {
                    totalEntropy = GetEntropy(pointTree.Leaves.Count, anomalyTree.Leaves.Count);
                }

                if (totalEntropy > 0.9)
                {
                    rootCauses.AddRange(LocalizeRootCauseByDimension(pointTree.Leaves, anomalyTree, pointTree, totalEntropy, src.AnomalyDimensions));
                }
                else
                {
                    rootCauses.AddRange(LocalizeRootCauseByDimension(pointTree.Leaves, anomalyTree, pointTree, totalEntropy, src.AnomalyDimensions));
                }

                dst.Items = rootCauses;
            }
        }

        public DimensionInfo SeperateDimension(Dictionary<string, string> dimensions, string aggSymbol)
        {
            DimensionInfo info = DimensionInfo.CreateDefaultInstance();
            foreach (KeyValuePair<string, string> entry in dimensions)
            {
                string key = entry.Key;
                if (aggSymbol.Equals(entry.Value))
                {
                    info.AggDim.Add(key);
                }
                else
                {
                    info.DetailDim.Add(key);
                }
            }

            return info;
        }

        protected PointTree BuildPointTree(List<Point> pointList, List<string> aggDims, Dictionary<string, string> subDim, string aggSymbol, AggregateType aggType, bool filterByAnomaly = false)
        {
            PointTree tree = PointTree.CreateDefaultInstance();

            foreach (Point point in pointList)
            {
                bool isValidPoint = true;
                if (filterByAnomaly)
                {
                    isValidPoint = point.IsAnomaly == true;
                }
                if (ContainsAll(point.Dimensions, subDim) && isValidPoint)
                {
                    if (aggDims.Count == 0)
                    {
                        tree.ParentNode = point;
                        tree.Leaves.Add(point);
                    }
                    else
                    {
                        int aggNum = 0;
                        string nextDim = null;

                        foreach (string dim in aggDims)
                        {
                            if (IsAggregationDimension(point.Dimensions[dim], aggSymbol))
                            {
                                aggNum++;
                            }
                            else
                            {
                                nextDim = dim;
                            }
                        }

                        if (aggNum == aggDims.Count)
                        {
                            tree.ParentNode = point;
                        }
                        else if (aggNum == aggDims.Count - 1)
                        {
                            if (!tree.ChildrenNodes.ContainsKey(nextDim))
                            {
                                tree.ChildrenNodes.Add(nextDim, new List<Point>());
                            }
                            tree.ChildrenNodes[nextDim].Add(point);
                        }

                        if (aggNum == 0)
                        {
                            tree.Leaves.Add(point);
                        }
                    }
                }
            }

            return tree;
        }
        private PointTree CompleteTreeBottomUp(PointTree tree, AggregateType aggType, string aggSymbol, List<string> aggDims)
        {

            if (tree.Leaves.Count == 0) return tree;

            Dictionary<string, HashSet<string>> map = new Dictionary<string, HashSet<string>>();
            foreach (Point p in tree.Leaves)
            {
                foreach (KeyValuePair<string, string> keyValuePair in p.Dimensions)
                {
                    if (aggDims.Contains(keyValuePair.Key))
                    {
                        if (map.ContainsKey(keyValuePair.Key))
                        {
                            map[keyValuePair.Key].Add(keyValuePair.Value);
                        }
                        else
                        {
                            map.Add(keyValuePair.Key, new HashSet<string>() { keyValuePair.Value });
                        }
                    }
                }
            }

            foreach (KeyValuePair<string, HashSet<string>> pair in map)
            {
                if (tree.ChildrenNodes.ContainsKey(pair.Key))
                {
                    if (tree.ChildrenNodes[pair.Key].Count < pair.Value.Count)
                    {
                        foreach (string value in pair.Value)
                        {
                            if (!IsAggDimensionExisted(pair.Key, value, tree.ChildrenNodes[pair.Key]))
                            {
                                Point p = SimulateBottomUpValue(tree.Leaves, pair.Key, value, aggType, aggSymbol);
                                tree.ChildrenNodes[pair.Key].Add(p);
                            }
                        }
                    }
                }
                else
                {
                    List<Point> childPoints = new List<Point>();
                    foreach (string value in pair.Value)
                    {
                        //simulate the aggregation value
                        Point p = SimulateBottomUpValue(tree.Leaves, pair.Key, value, aggType, aggSymbol);
                        childPoints.Add(p);
                    }

                    tree.ChildrenNodes.Add(pair.Key, childPoints);
                }
            }

            return tree;
        }

        private bool IsAggDimensionExisted(string key, string value, List<Point> points)
        {
            foreach (Point p in points)
            {
                if (p.Dimensions[key].Equals(value))
                {
                    return true;
                }
            }
            return false;
        }

        private Point SimulateBottomUpValue(List<Point> leaves, string key, string keyValue, AggregateType type, string aggSymbol)
        {
            Point p = null;

            Dictionary<string, string> dimension = new Dictionary<string, string>();

            dimension.Add(key, keyValue);

            foreach (KeyValuePair<string, string> pair in leaves[0].Dimensions)
            {
                if (!pair.Key.Equals(key))
                {
                    dimension.Add(pair.Key, aggSymbol);
                }
            }

            if (type.Equals(AggregateType.Sum))
            {

                bool isAnomaly = false;
                double value = 0;
                double expectedValue = 0;
                foreach (Point leave in leaves)
                {

                    if (leave.Dimensions.ContainsKey(key) && leave.Dimensions[key].Equals(keyValue))
                    {
                        value += leave.Value;
                        expectedValue = leave.ExpectedValue;
                        isAnomaly = isAnomaly || leave.IsAnomaly;
                    }
                }

                p = new Point(value, expectedValue, isAnomaly, dimension);
            }

            return p;
        }

        public Dictionary<string, string> GetSubDim(Dictionary<string, string> dimension, List<string> keyList)
        {
            Dictionary<string, string> subDim = new Dictionary<string, string>();

            foreach (String dim in keyList)
            {
                subDim.Add(dim, dimension[dim]);
            }
            return subDim;
        }

        protected List<Point> SelectPoints(List<Point> points, Dictionary<string, string> subDim)
        {
            List<Point> list = new List<Point>();

            foreach (Point point in points)
            {
                if (ContainsAll(point.Dimensions, subDim))
                {
                    //remove duplicated points
                    if (!list.Contains(point))
                    {
                        list.Add(point);
                    }
                }
            }

            return list;
        }

        protected List<RootCauseItem> LocalizeRootCauseByDimension(List<Point> totalPoints, PointTree anomalyTree, PointTree pointTree, double totoalEntropy, Dictionary<string, string> anomalyDimension)
        {
            var set = anomalyTree.ChildrenNodes.Keys;

            BestDimension best = null;
            if (anomalyTree.Leaves.Count > 0)
            {
                best = SelectBestDimension(totalPoints, anomalyTree.Leaves, set.ToList(), totoalEntropy);
            }
            else
            {
                //has no leaves information, should calculate the entropy information according to the children nodes
                best = SelectBestDimension(pointTree.ChildrenNodes, anomalyTree.ChildrenNodes, set.ToList(), totoalEntropy);
            }

            if (best == null)
            {
                return new List<RootCauseItem>() { new RootCauseItem(anomalyDimension) };
            }

            List<Point> children = GetTopAnomaly(anomalyTree.ChildrenNodes[best.DimensionKey], anomalyTree.ParentNode, totalPoints, best.DimensionKey);
            if (children == null)
            {
                //As the cause couldn't be found, the root cause should be itself
                return new List<RootCauseItem>() { new RootCauseItem(anomalyDimension, best.DimensionKey) };
            }
            else
            {
                List<RootCauseItem> causes = new List<RootCauseItem>();
                // For the found causes, we return the result
                foreach (Point anomaly in children)
                {
                    causes.Add(new RootCauseItem(UpdateDimensionValue(anomalyDimension, best.DimensionKey, anomaly.Dimensions[best.DimensionKey]), best.DimensionKey));
                }
                return causes;
            }
        }

        protected double GetEntropy(int totalNum, int anomalyNum)
        {
            double ratio = (double)anomalyNum / totalNum;
            if (ratio == 0 || ratio == 1)
            {
                return 0;
            }

            return -(ratio * Log2(ratio) + (1 - ratio) * Log2(1 - ratio));
        }

        protected Dictionary<string, double> GetEntropyList(BestDimension best, List<Point> points)
        {
            Dictionary<string, double> list = new Dictionary<string, double>();
            // need to update, change to children if necessary
            foreach (Point point in points)
            {
                string dimVal = point.Dimensions[best.DimensionKey];
                int pointSize = GetPointSize(best, dimVal);
                int anomalySize = GetAnomalyPointSize(best, dimVal);

                double dimEntropy = GetEntropy(pointSize, anomalySize);
                list.Add(dimVal, dimEntropy);
            }

            return list;
        }

        protected List<Point> GetTopAnomaly(List<Point> anomalyPoints, Point root, List<Point> totalPoints, string dimKey)
        {
            Dictionary<string, int> pointDistribution = new Dictionary<string, int>();
            UpdateDistribution(pointDistribution, totalPoints, dimKey);

            anomalyPoints.OrderBy(x => x.Delta);

            if (root.Delta > 0)
            {
                anomalyPoints.Reverse();
            }

            if (anomalyPoints.Count == 1)
            {
                return anomalyPoints;
            }

            double delta = 0;
            double preDelta = 0;
            List<Point> causeList = new List<Point>();
            foreach (Point anomaly in anomalyPoints)
            {
                if (StopAnomalyComparison(delta, root.Delta, anomaly.Delta, preDelta))
                {
                    break;
                }

                delta += anomaly.Delta;
                causeList.Add(anomaly);
                preDelta = anomaly.Delta;
            }

            int pointSize = GetTotalNumber(pointDistribution);
            if (ShouldSeperateAnomaly(delta, root.Delta, pointSize, causeList.Count))
            {
                return causeList;
            }

            return null;
        }

        protected BestDimension SelectBestDimension(List<Point> totalPoints, List<Point> anomalyPoints, List<string> aggDim, double totalEntropy)
        {
            Dictionary<BestDimension, double> entroyGainMap = new Dictionary<BestDimension, double>();
            Dictionary<BestDimension, double> entroyGainRatioMap = new Dictionary<BestDimension, double>();
            double sumGain = 0;

            foreach (string dimKey in aggDim)
            {
                BestDimension dimension = BestDimension.CreateDefaultInstance();
                dimension.DimensionKey = dimKey;

                UpdateDistribution(dimension.PointDis, totalPoints, dimKey);
                UpdateDistribution(dimension.AnomalyDis, anomalyPoints, dimKey);

                double gain = GetDimensionEntropyGain(dimension.PointDis, dimension.AnomalyDis, totalEntropy);
                dimension.Entropy = totalEntropy - gain;
                entroyGainMap.Add(dimension, gain);

                double gainRatio = gain / GetDimensionInstrinsicValue(dimension.PointDis, dimension.AnomalyDis, totalEntropy);
                entroyGainRatioMap.Add(dimension, gainRatio);

                sumGain += gain;
            }

            double meanGain = sumGain / aggDim.Count();

            BestDimension best = FindBestDimension(entroyGainMap, entroyGainRatioMap, meanGain);
            return best;
        }

        public BestDimension SelectBestDimension(Dictionary<string, List<Point>> pointChildren, Dictionary<string, List<Point>> anomalyChildren, List<String> aggDim, double totalEntropy)
        {
            Dictionary<BestDimension, double> entroyGainMap = new Dictionary<BestDimension, double>();
            Dictionary<BestDimension, double> entroyGainRatioMap = new Dictionary<BestDimension, double>();
            double sumGain = 0;

            foreach (string dimKey in aggDim)
            {
                BestDimension dimension = BestDimension.CreateDefaultInstance();
                dimension.DimensionKey = dimKey;

                UpdateDistribution(dimension.PointDis, pointChildren[dimKey], dimKey);
                UpdateDistribution(dimension.AnomalyDis, anomalyChildren[dimKey], dimKey);

                double gain = GetDimensionEntropyGain(dimension.PointDis, dimension.AnomalyDis, totalEntropy);
                dimension.Entropy = totalEntropy - gain;
                entroyGainMap.Add(dimension, gain);

                double gainRatio = gain / GetDimensionInstrinsicValue(dimension.PointDis, dimension.AnomalyDis, totalEntropy);
                entroyGainRatioMap.Add(dimension, gainRatio);

                sumGain += gain;
            }

            double meanGain = sumGain / aggDim.Count();

            BestDimension best = FindBestDimension(entroyGainMap, entroyGainRatioMap, meanGain);

            return best;
        }

        private BestDimension FindBestDimension(Dictionary<BestDimension, double> entropyGainMap, Dictionary<BestDimension, double> entropyGainRationMap, double meanGain)
        {
            BestDimension best = null;
            foreach (KeyValuePair<BestDimension, double> dimension in entropyGainMap)
            {
                if (dimension.Key.AnomalyDis.Count == 1 || dimension.Value >= meanGain)
                {
                    if (dimension.Key.AnomalyDis.Count > 1)
                    {
                        if (best == null || (best.AnomalyDis.Count != 1 && entropyGainRationMap[best].CompareTo(dimension.Value) < 0))
                        {
                            best = dimension.Key;
                        }
                    }
                    else
                    {
                        if (best == null || best.AnomalyDis.Count > 1)
                        {
                            best = dimension.Key;
                        }
                        else
                        {
                            if (entropyGainRationMap[best].CompareTo(dimension.Value) < 0)
                            {
                                best = dimension.Key;
                            }
                        }
                    }
                }
            }
            return best;
        }

        public Point FindPointByDimension(Dictionary<string, string> dim, List<Point> points)
        {
            foreach (Point p in points)
            {
                bool isEqual = true;
                foreach (KeyValuePair<string, string> item in p.Dimensions)
                {
                    if (!dim[item.Key].Equals(item.Value))
                    {
                        isEqual = false;
                    }
                }

                if (isEqual)
                {
                    return p;
                }
            }

            return null;
        }

        public void UpdateRootCauseDirection(List<Point> points, ref RootCause dst)
        {
            foreach (RootCauseItem item in dst.Items)
            {
                Point rootCausePoint = FindPointByDimension(item.Dimension, points);
                if (rootCausePoint != null)
                {
                    if (rootCausePoint.ExpectedValue < rootCausePoint.Value)
                    {
                        item.Direction = AnomalyDirection.Up;
                    }
                    else
                    {
                        item.Direction = AnomalyDirection.Down;
                    }
                }

            }
        }

        public void GetRootCauseScore(List<Point> points, Dictionary<string, string> anomalyRoot, ref RootCause dst, double beta)
        {
            if (dst.Items.Count > 1)
            {
                //get surprise value and explanary power value
                Point anomalyPoint = FindPointByDimension(anomalyRoot, points);

                double sumSurprise = 0;
                double sumEp = 0;
                List<RootCauseScore> scoreList = new List<RootCauseScore>();

                foreach (RootCauseItem item in dst.Items)
                {
                    Point rootCausePoint = FindPointByDimension(item.Dimension, points);
                    if (anomalyPoint != null && rootCausePoint != null)
                    {
                        Tuple<double, double> scores = GetSupriseAndExplainaryScore(rootCausePoint, anomalyPoint);
                        scoreList.Add(new RootCauseScore(scores.Item1, scores.Item2));
                        sumSurprise += scores.Item1;
                        sumEp += Math.Abs(scores.Item2);
                    }
                }

                //normalize and get final score
                for (int i = 0; i < scoreList.Count; i++)
                {
                    dst.Items[i].Score = GetFinalScore(scoreList[i].Surprise / sumSurprise, Math.Abs(scoreList[i].ExplainaryScore) / sumEp, beta);

                }
            }
            else if (dst.Items.Count == 1)
            {
                Point rootCausePoint = FindPointByDimension(dst.Items[0].Dimension, points);

                Point anomalyPoint = FindPointByDimension(anomalyRoot, points);
                if (anomalyPoint != null && rootCausePoint != null)
                {
                    Tuple<double, double> scores = GetSupriseAndExplainaryScore(rootCausePoint, anomalyPoint);
                    dst.Items[0].Score = GetFinalScore(scores.Item1, scores.Item2, beta);
                }
            }
        }

        private double GetSurpriseScore(Point rootCausePoint, Point anomalyPoint)
        {
            double p = rootCausePoint.ExpectedValue / anomalyPoint.ExpectedValue;
            double q = rootCausePoint.Value / anomalyPoint.Value;
            double surprise = 0.5 * (p * Log2(2 * p / (p + q)) + q * Log2(2 * q / (p + q)));

            return surprise;
        }

        private double GetFinalScore(double surprise, double ep, double beta)
        {
            return Math.Max(1, beta * surprise + (1 - beta) * ep);
        }

        private Tuple<double, double> GetSupriseAndExplainaryScore(Point rootCausePoint, Point anomalyPoint)
        {
            double surprise = GetSurpriseScore(rootCausePoint, anomalyPoint);
            double ep = (rootCausePoint.Value - rootCausePoint.ExpectedValue) / (anomalyPoint.Value - anomalyPoint.ExpectedValue);

            return new Tuple<double, double>(surprise, ep);
        }
        private static Dictionary<string, string> UpdateDimensionValue(Dictionary<string, string> dimension, string key, string value)
        {
            Dictionary<string, string> newDim = new Dictionary<string, string>(dimension);
            newDim[key] = value;
            return newDim;
        }

        private bool StopAnomalyComparison(double preTotal, double parent, double current, double pre)
        {
            if (Math.Abs(preTotal) < Math.Abs(parent) * 0.95)
            {
                return false;
            }

            return Math.Abs(pre) / Math.Abs(current) > 2;
        }

        private bool ShouldSeperateAnomaly(double total, double parent, int totalSize, int size)
        {
            if (Math.Abs(total) < Math.Abs(parent) * 0.95)
            {
                return false;
            }

            if (size == totalSize && size == 1)
            {
                return true;
            }

            return size <= totalSize * 0.5;
        }

        private double GetDimensionEntropyGain(Dictionary<string, int> pointDis, Dictionary<string, int> anomalyDis, double totalEntropy)
        {
            int total = GetTotalNumber(pointDis);
            double entropy = 0;
            foreach (string key in anomalyDis.Keys)
            {
                double dimEntropy = GetEntropy(pointDis[key], anomalyDis[key]);
                entropy += dimEntropy * pointDis[key] / total;
            }
            return totalEntropy - entropy;
        }

        private double GetDimensionInstrinsicValue(Dictionary<string, int> pointDis, Dictionary<string, int> anomalyDis, double totalEntropy)
        {
            double instrinsicValue = 0;

            foreach (string key in anomalyDis.Keys)
            {
                instrinsicValue -= Log2((double)anomalyDis[key] / pointDis[key]) * anomalyDis[key] / pointDis[key];
            }

            return instrinsicValue;
        }

        private int GetTotalNumber(Dictionary<string, int> distribution)
        {
            int total = 0;
            foreach (int num in distribution.Values)
            {
                total += num;
            }
            return total;
        }

        private void UpdateDistribution(Dictionary<string, int> distribution, List<Point> points, string dimKey)
        {
            foreach (Point point in points)
            {
                string dimVal = point.Dimensions[dimKey];
                if (!distribution.ContainsKey(dimVal))
                {
                    distribution.Add(dimVal, 0);
                }
                distribution[dimVal] = distribution[dimVal] + 1;
            }
        }

        public double Log2(double val)
        {
            return Math.Log(val) / Math.Log(2);
        }

        public bool ContainsAll(Dictionary<string, string> bigDic, Dictionary<string, string> smallDic)
        {
            foreach (var item in smallDic)
            {
                if (!bigDic.ContainsKey(item.Key))
                {
                    return false;
                }

                if (bigDic.ContainsKey(item.Key) && !bigDic[item.Key].Equals(smallDic[item.Key]))
                {
                    return false;
                }
            }
            return true;
        }

        private bool IsAggregationDimension(string val, string aggSymbol)
        {
            return val.Equals(aggSymbol);
        }

        private int GetPointSize(BestDimension dim, string key)
        {
            int pointSize = 0;
            if (dim.PointDis.ContainsKey(key))
            {
                pointSize = dim.PointDis[key];
            }
            return pointSize;
        }

        private int GetAnomalyPointSize(BestDimension dim, string key)
        {
            int anomalyPointSize = 0;
            if (dim.AnomalyDis.ContainsKey(key))
            {
                anomalyPointSize = dim.AnomalyDis[key];
            }
            return anomalyPointSize;
        }
    }

    public class DimensionInfo
    {
        public List<string> DetailDim { get; set; }
        public List<string> AggDim { get; set; }

        public static DimensionInfo CreateDefaultInstance()
        {
            DimensionInfo instance = new DimensionInfo();
            instance.DetailDim = new List<string>();
            instance.AggDim = new List<string>();
            return instance;
        }
    }

    public class PointTree
    {
        public Point ParentNode;
        public Dictionary<string, List<Point>> ChildrenNodes;
        public List<Point> Leaves;

        public static PointTree CreateDefaultInstance()
        {
            PointTree instance = new PointTree();
            instance.Leaves = new List<Point>();
            instance.ChildrenNodes = new Dictionary<string, List<Point>>();
            return instance;
        }
    }

    public sealed class Point : IEquatable<Point>
    {
        public double Value { get; set; }
        public double ExpectedValue { get; set; }
        public bool IsAnomaly { get; set; }
        public Dictionary<string, string> Dimensions { get; set; }

        public double Delta { get; set; }

        public Point(double value, double expectedValue, bool isAnomaly, Dictionary<string, string> dimensions)
        {
            Value = value;
            ExpectedValue = expectedValue;
            IsAnomaly = isAnomaly;
            Dimensions = dimensions;
            Delta = (value - expectedValue) / expectedValue;
            if (expectedValue == 0)
            {
                Delta = 0;
            }
        }

        public bool Equals(Point other)
        {
            foreach (KeyValuePair<string, string> item in Dimensions)
            {
                if (!other.Dimensions[item.Key].Equals(item.Value))
                {
                    return false;
                }
            }
            return true;
        }

        public override int GetHashCode()
        {
            return Dimensions.GetHashCode();
        }
    }

    public sealed class MetricSlice
    {
        public DateTime TimeStamp { get; set; }
        public List<Point> Points { get; set; }

        public MetricSlice(DateTime timeStamp, List<Point> points)
        {
            TimeStamp = timeStamp;
            Points = points;
        }
    }

    public sealed class BestDimension
    {
        public string DimensionKey;
        public double Entropy;
        public Dictionary<string, int> AnomalyDis;
        public Dictionary<string, int> PointDis;

        public BestDimension() { }
        public static BestDimension CreateDefaultInstance()
        {
            BestDimension instance = new BestDimension();
            instance.AnomalyDis = new Dictionary<string, int>();
            instance.PointDis = new Dictionary<string, int>();
            return instance;
        }
    }

    public sealed class AnomalyCause
    {
        public string DimensionKey;
        public List<Point> Anomalies;

        public AnomalyCause() { }
    }

    public sealed class RootCauseItem : IEquatable<RootCauseItem>
    {
        public double Score;
        public string Path;
        public Dictionary<string, string> Dimension;
        public AnomalyDirection Direction;

        public RootCauseItem(Dictionary<string, string> rootCause)
        {
            Dimension = rootCause;
        }

        public RootCauseItem(Dictionary<string, string> rootCause, string path)
        {
            Dimension = rootCause;
            Path = path;
        }
        public bool Equals(RootCauseItem other)
        {
            if (Dimension.Count == other.Dimension.Count)
            {
                foreach (KeyValuePair<string, string> item in Dimension)
                {
                    if (!other.Dimension[item.Key].Equals(item.Value))
                    {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }
    }

    public enum AnomalyDirection
    {
        /// <summary>
        /// the value is larger than expected value.
        /// </summary>
        Up = 0,
        /// <summary>
        /// the value is lower than expected value.
        ///  </summary>
        Down = 1
    }

    public class RootCauseScore
    {
        public double Surprise;
        public double ExplainaryScore;

        public RootCauseScore(double surprise, double explainaryScore)
        {
            Surprise = surprise;
            ExplainaryScore = explainaryScore;
        }
    }

    public enum AggregateType
    {
        /// <summary>
        /// Make the aggregate type as sum.
        /// </summary>
        Sum = 0,
        /// <summary>
        /// Make the aggregate type as average.
        ///  </summary>
        Avg = 1,
        /// <summary>
        /// Make the aggregate type as min.
        /// </summary>
        Min = 2,
        /// <summary>
        /// Make the aggregate type as max.
        /// </summary>
        Max = 3
    }
}
