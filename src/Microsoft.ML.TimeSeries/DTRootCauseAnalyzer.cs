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
        private static double _anomalyRatioThreshold = 0.5;

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
            //no aggregation dimension
            if (dimensionInfo.AggDims.Count == 0)
            {
                return dst;
            }
            Dictionary<string, string> subDim = GetSubDim(_src.AnomalyDimensions, dimensionInfo.DetailDim);
            List<Point> totalPoints = GetTotalPointsForAnomalyTimestamp(_src, subDim);
            GetRootCauseList(_src, ref dst, dimensionInfo, totalPoints, subDim, dimensionInfo.AggDims);
            UpdateRootCauseDirection(totalPoints, ref dst);
            GetRootCauseScore(totalPoints, _src.AnomalyDimensions, ref dst, _beta);

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

            return points;
        }

        public void GetRootCauseList(RootCauseLocalizationInput src, ref RootCause dst, DimensionInfo dimensionInfo, List<Point> totalPoints, Dictionary<string, string> subDim, List<string> aggDims)
        {
            Tuple<PointTree, PointTree, List<Point>> pointInfo = BuildPointInfo(totalPoints, dimensionInfo.AggDims, subDim, src.AggSymbol, src.AggType);
            PointTree pointTree = pointInfo.Item1;
            PointTree anomalyTree = pointInfo.Item2;
            List<Point> uniquePoints = pointInfo.Item3;

            //which means there is no anomaly point with the anomaly dimension
            if (anomalyTree.ParentNode == null)
            {
                return;
            }

            List<RootCauseItem> rootCauses = new List<RootCauseItem>();
            // no point under anomaly dimension
            if (uniquePoints.Count == 0)
            {
                if (anomalyTree.Leaves.Count != 0)
                {
                    throw new Exception("point leaves not match with anomaly leaves");
                }

                return;
            }
            else
            {
                double totalEntropy = 1;
                if (anomalyTree.Leaves.Count > 0)
                {
                    totalEntropy = GetEntropy(pointTree.Leaves.Count, anomalyTree.Leaves.Count);
                }

                rootCauses.AddRange(LocalizeRootCauseByDimension(anomalyTree, pointTree, totalEntropy, src.AnomalyDimensions, aggDims));
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
                    info.AggDims.Add(key);
                }
                else
                {
                    info.DetailDim.Add(key);
                }
            }

            return info;
        }

        protected Tuple<PointTree, PointTree, List<Point>> BuildPointInfo(List<Point> pointList, List<string> aggDims, Dictionary<string, string> subDim, string aggSymbol, AggregateType aggType)
        {

            List<Point> uniquePointList = new List<Point>();

            PointTree pointTree = PointTree.CreateDefaultInstance();
            PointTree anomalyTree = PointTree.CreateDefaultInstance();

            foreach (Point point in pointList)
            {
                if (ContainsAll(point.Dimension, subDim))
                {
                    //remove duplicated points
                    if (!uniquePointList.Contains(point))
                    {
                        uniquePointList.Add(point);
                        bool isValidPoint = point.IsAnomaly == true;
                        if (ContainsAll(point.Dimension, subDim))
                        {
                            BuildTree(pointTree, aggDims, point, aggSymbol);

                            if (isValidPoint)
                            {
                                BuildTree(anomalyTree, aggDims, point, aggSymbol);
                            }
                        }
                    }
                }
            }

            return new Tuple<PointTree, PointTree, List<Point>>(pointTree, anomalyTree, uniquePointList);
        }

        private void BuildTree(PointTree tree, List<string> aggDims, Point point, string aggSymbol)
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
                    if (IsAggregationDimension(point.Dimension[dim], aggSymbol))
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

        public Dictionary<string, string> GetSubDim(Dictionary<string, string> dimension, List<string> keyList)
        {
            Dictionary<string, string> subDim = new Dictionary<string, string>();

            foreach (string dim in keyList)
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
                if (ContainsAll(point.Dimension, subDim))
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

        protected List<RootCauseItem> LocalizeRootCauseByDimension(PointTree anomalyTree, PointTree pointTree, double totoalEntropy, Dictionary<string, string> anomalyDimension, List<string> aggDims)
        {
            BestDimension best = null;
            if (anomalyTree.ChildrenNodes.Count == 0)
            {
                best = SelectBestDimension(pointTree.Leaves, anomalyTree.Leaves, aggDims, totoalEntropy);
            }
            else
            {
                //has no leaves information, should calculate the entropy information according to the children nodes
                best = SelectBestDimension(pointTree.ChildrenNodes, anomalyTree.ChildrenNodes, aggDims, totoalEntropy);
            }

            if (best == null)
            {
                return new List<RootCauseItem>() { new RootCauseItem(anomalyDimension) };
            }

            List<Point> children = null;
            if (anomalyTree.ChildrenNodes.ContainsKey(best.DimensionKey))
            {
                children = GetTopAnomaly(anomalyTree.ChildrenNodes[best.DimensionKey], anomalyTree.ParentNode, pointTree.ChildrenNodes[best.DimensionKey].Count > 0 ? pointTree.ChildrenNodes[best.DimensionKey] : pointTree.Leaves, best.DimensionKey);
            }
            else
            {
                if (best.AnomalyDis.Count > 0)
                {
                    children = new List<Point>();
                    foreach (string dimValue in best.AnomalyDis.Keys.ToArray())
                    {
                        Point p = new Point(UpdateDimensionValue(anomalyDimension, best.DimensionKey, dimValue));
                        children.Add(p);
                    }
                }
            }

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
                    causes.Add(new RootCauseItem(UpdateDimensionValue(anomalyDimension, best.DimensionKey, anomaly.Dimension[best.DimensionKey]), best.DimensionKey));
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

                double gainRatio = gain / GetDimensionInstrinsicValue(dimension.PointDis, dimension.AnomalyDis);
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

                if (pointChildren.ContainsKey(dimKey))
                {
                    UpdateDistribution(dimension.PointDis, pointChildren[dimKey], dimKey);
                }
                if (anomalyChildren.ContainsKey(dimKey))
                {
                    UpdateDistribution(dimension.AnomalyDis, anomalyChildren[dimKey], dimKey);
                }

                double gain = GetDimensionEntropyGain(dimension.PointDis, dimension.AnomalyDis, totalEntropy, true);
                dimension.Entropy = totalEntropy - gain;
                entroyGainMap.Add(dimension, gain);

                double gainRatio = gain / GetDimensionInstrinsicValue(dimension.PointDis, dimension.AnomalyDis);
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
                foreach (KeyValuePair<string, string> item in p.Dimension)
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

            return size <= totalSize * _anomalyRatioThreshold;
        }

        private double GetDimensionEntropyGain(Dictionary<string, int> pointDis, Dictionary<string, int> anomalyDis, double totalEntropy, bool isChildren = false)
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

        private double GetDimensionInstrinsicValue(Dictionary<string, int> pointDis, Dictionary<string, int> anomalyDis)
        {
            int total = GetTotalNumber(pointDis);
            double instrinsicValue = 0;

            foreach (string key in pointDis.Keys)
            {
                instrinsicValue -= Log2((double)pointDis[key] / total) * (double)pointDis[key] / total;
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
                string dimVal = point.Dimension[dimKey];
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

        public static bool ContainsAll(Dictionary<string, string> bigDic, Dictionary<string, string> smallDic)
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
    }

    public class DimensionInfo
    {
        public List<string> DetailDim { get; set; }
        public List<string> AggDims { get; set; }

        public static DimensionInfo CreateDefaultInstance()
        {
            DimensionInfo instance = new DimensionInfo();
            instance.DetailDim = new List<string>();
            instance.AggDims = new List<string>();
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
}
