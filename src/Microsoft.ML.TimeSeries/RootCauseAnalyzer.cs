// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.TimeSeries
{
    internal class RootCauseAnalyzer
    {
        private static readonly double _anomalyRatioThreshold = 0.5;
        private static readonly double _anomalyPreDeltaThreshold = 2;

        private readonly RootCauseLocalizationInput _src;
        private readonly double _beta;
        private readonly double _rootCauseThreshold;
        private readonly List<RootCause> _preparedCauses;

        public RootCauseAnalyzer(RootCauseLocalizationInput src, double beta, double rootCauseThreshold)
        {
            _src = src;
            _beta = beta;
            _rootCauseThreshold = rootCauseThreshold;
            _preparedCauses = new List<RootCause>();
        }

        public RootCause Analyze()
        {
            return AnalyzeOneLayer(_src).FirstOrDefault();
        }

        public List<RootCause> AnalyzePossibleCauses()
        {
            return AnalyzeOneLayer(_src);
        }

        /// <summary>
        ///  This is a function for analyzing one layer for root cause. We rank dimensions according to their likelihood of containing the root case.
        ///  For each dimension, we select one dimension with values who contributes the most to the anomaly.
        /// </summary>
        private List<RootCause> AnalyzeOneLayer(RootCauseLocalizationInput src)
        {
            DimensionInfo dimensionInfo = SeparateDimension(src.AnomalyDimension, src.AggregateSymbol);
            Tuple<PointTree, PointTree, Dictionary<Dictionary<string, object>, TimeSeriesPoint>> pointInfo = GetPointsInfo(src, dimensionInfo);
            PointTree pointTree = pointInfo.Item1;
            PointTree anomalyTree = pointInfo.Item2;
            Dictionary<Dictionary<string, Object>, TimeSeriesPoint> dimPointMapping = pointInfo.Item3;

            //which means there is no anomaly point with the anomaly dimension or no point under anomaly dimension
            if (anomalyTree.ParentNode == null || dimPointMapping.Count == 0)
            {
                _preparedCauses.Add(new RootCause() { Items = new List<RootCauseItem>() });
                return _preparedCauses;
            }

            LocalizeRootCausesByDimension(anomalyTree, pointTree, src.AnomalyDimension, dimensionInfo.AggDims);
            foreach (var dst in _preparedCauses)
            {
                GetRootCauseDirectionAndScore(dimPointMapping, src.AnomalyDimension, dst, _beta, pointTree, src.AggregateType, src.AggregateSymbol);
            }

            return _preparedCauses;
        }

        protected List<TimeSeriesPoint> GetTotalPointsForAnomalyTimestamp(RootCauseLocalizationInput src)
        {
            MetricSlice slice = src.Slices.Single(slice => slice.TimeStamp.Equals(src.AnomalyTimestamp));
            return slice.Points;
        }

        private DimensionInfo SeparateDimension(Dictionary<string, Object> dimensions, Object aggSymbol)
        {
            DimensionInfo info = new DimensionInfo();
            foreach (KeyValuePair<string, Object> entry in dimensions)
            {
                string key = entry.Key;
                if (object.Equals(aggSymbol, entry.Value))
                {
                    info.AggDims.Add(key);
                }
                else
                {
                    info.DetailDims.Add(key);
                }
            }

            return info;
        }

        private Tuple<PointTree, PointTree, Dictionary<Dictionary<string, object>, TimeSeriesPoint>> GetPointsInfo(RootCauseLocalizationInput src, DimensionInfo dimensionInfo)
        {
            PointTree pointTree = new PointTree();
            PointTree anomalyTree = new PointTree();
            DimensionComparer dc = new DimensionComparer();
            Dictionary<Dictionary<string, object>, TimeSeriesPoint> dimPointMapping = new Dictionary<Dictionary<string, object>, TimeSeriesPoint>(dc);

            List<TimeSeriesPoint> totalPoints = GetTotalPointsForAnomalyTimestamp(src);
            Dictionary<string, Object> subDim = GetSubDim(src.AnomalyDimension, dimensionInfo.DetailDims);

            foreach (TimeSeriesPoint point in totalPoints)
            {
                if (ContainsAll(point.Dimension, subDim))
                {
                    if (!dimPointMapping.ContainsKey(point.Dimension))
                    {
                        dimPointMapping.Add(point.Dimension, point);
                        bool isValidPoint = point.IsAnomaly == true;
                        if (ContainsAll(point.Dimension, subDim))
                        {
                            BuildTree(pointTree, dimensionInfo.AggDims, point, src.AggregateSymbol);

                            if (isValidPoint)
                            {
                                BuildTree(anomalyTree, dimensionInfo.AggDims, point, src.AggregateSymbol);
                            }
                        }
                    }
                }
            }

            return new Tuple<PointTree, PointTree, Dictionary<Dictionary<string, Object>, TimeSeriesPoint>>(pointTree, anomalyTree, dimPointMapping);
        }

        protected Dictionary<string, Object> GetSubDim(Dictionary<string, Object> dimension, List<string> keyList)
        {
            return new Dictionary<string, object>(keyList.Select(dim => new KeyValuePair<string, object>(dim, dimension[dim])).ToDictionary(kvp => kvp.Key, kvp => kvp.Value));
        }

        private void LocalizeRootCausesByDimension(PointTree anomalyTree, PointTree pointTree, Dictionary<string, Object> anomalyDimension, List<string> aggDims)
        {
            IEnumerable<BestDimension> best;
            if (anomalyTree.ChildrenNodes.Count == 0)
            {
                //has no children node information, should use the leaves node (whose point has no aggregated dimensions) information
                best = SelectOrderedDimension(pointTree.Leaves, anomalyTree.Leaves, aggDims);
            }
            else
            {
                //has no leaves information, should calculate the entropy information according to the children nodes
                best = SelectOrderedDimensions(pointTree.ChildrenNodes, anomalyTree.ChildrenNodes, aggDims);
            }

            if (best == null)
            {
                _preparedCauses.Add(new RootCause { Items = new List<RootCauseItem>() { new RootCauseItem(anomalyDimension) } });
            }

            bool rootAsAnomaly = false;
            foreach (var dimension in best)
            {
                RootCause rootCause = new RootCause { Items = new List<RootCauseItem>() };

                rootCause.GainRatio = dimension.GainRatio;
                List<TimeSeriesPoint> children = null;
                if (anomalyTree.ChildrenNodes.ContainsKey(dimension.DimensionKey))
                {
                    //Use children node information to get top anomalies
                    children = GetTopAnomaly(anomalyTree.ChildrenNodes[dimension.DimensionKey], anomalyTree.ParentNode, pointTree.ChildrenNodes[dimension.DimensionKey].Count > 0 ? pointTree.ChildrenNodes[dimension.DimensionKey] : pointTree.Leaves, dimension.DimensionKey, !(pointTree.ChildrenNodes[dimension.DimensionKey].Count > 0));
                }
                else
                {
                    //Use leaves node information to get top anomalies
                    children = GetTopAnomaly(anomalyTree.Leaves, anomalyTree.ParentNode, pointTree.Leaves, dimension.DimensionKey, true);
                }

                if (children == null)
                {
                    //As the cause couldn't be found, the root cause should be itself
                    if (!rootAsAnomaly)
                    {
                        rootAsAnomaly = true;
                        rootCause.Items.Add(new RootCauseItem(anomalyDimension));
                    }
                }
                else
                {
                    rootCause.Items.AddRange(children.Select(anomaly =>
                        new RootCauseItem(UpdateDimensionValue(anomalyDimension, dimension.DimensionKey, anomaly.Dimension[dimension.DimensionKey]), new List<string>() { dimension.DimensionKey })));
                }

                _preparedCauses.Add(rootCause);
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

        protected List<TimeSeriesPoint> GetTopAnomaly(List<TimeSeriesPoint> anomalyPoints, TimeSeriesPoint root, List<TimeSeriesPoint> totalPoints, string dimKey, bool isLeaveslevel = false)
        {
            Dictionary<string, int> pointDistribution = new Dictionary<string, int>();
            UpdateDistribution(pointDistribution, totalPoints, dimKey);

            anomalyPoints = anomalyPoints.OrderBy(x => x.Delta).ToList();

            if (root.Delta > 0)
            {
                anomalyPoints.Reverse();
            }
            else
            {
                anomalyPoints = anomalyPoints.FindAll(x => x.Delta < 0);
            }
            if (anomalyPoints.Count == 1)
            {
                return anomalyPoints;
            }

            double delta = 0;
            double preDelta = 0;
            List<TimeSeriesPoint> causeList = new List<TimeSeriesPoint>();
            foreach (TimeSeriesPoint anomaly in anomalyPoints)
            {
                if (StopAnomalyComparison(delta, root.Delta, anomaly.Delta, preDelta))
                {
                    break;
                }

                delta += anomaly.Delta;
                causeList.Add(anomaly);
                preDelta = anomaly.Delta;
            }

            int pointSize = isLeaveslevel ? pointDistribution.Count : GetTotalNumber(pointDistribution);
            if (ShouldSeparateAnomaly(delta, root.Delta, pointSize, causeList.Count))
            {
                return causeList;
            }

            return null;
        }

        /// <summary>
        ///  Use leaves point information to select ordered dimensions
        /// </summary>
        protected IEnumerable<BestDimension> SelectOrderedDimension(List<TimeSeriesPoint> totalPoints, List<TimeSeriesPoint> anomalyPoints, List<string> aggDim)
        {
            double totalEntropy = GetEntropy(totalPoints.Count, anomalyPoints.Count);
            SortedDictionary<BestDimension, double> entropyGainMap = new SortedDictionary<BestDimension, double>();
            Dictionary<BestDimension, double> entroyGainRatioMap = new Dictionary<BestDimension, double>();
            double sumGain = 0;

            foreach (string dimKey in aggDim)
            {
                BestDimension dimension = new BestDimension();
                dimension.DimensionKey = dimKey;

                UpdateDistribution(dimension.PointDis, totalPoints, dimKey);
                UpdateDistribution(dimension.AnomalyDis, anomalyPoints, dimKey);

                double relativeEntropy = GetDimensionEntropy(dimension.PointDis, dimension.AnomalyDis);
                double gain = totalEntropy - relativeEntropy;
                if (Double.IsNaN(gain))
                {
                    gain = 0;
                }
                entropyGainMap.Add(dimension, gain);
                dimension.Gain = gain;

                double gainRatio = gain / GetDimensionIntrinsicValue(dimension.PointDis);
                if (Double.IsInfinity(gainRatio))
                {
                    gainRatio = 0;
                }
                entroyGainRatioMap.Add(dimension, gainRatio);
                dimension.GainRatio = gainRatio;

                sumGain += gain;
            }

            double meanGain = sumGain / aggDim.Count();

            return OrderDimensions(entropyGainMap, entroyGainRatioMap, meanGain);
        }

        /// <summary>
        ///  Use children point information to select ordered dimensions
        /// </summary>
        private IEnumerable<BestDimension> SelectOrderedDimensions(Dictionary<string, List<TimeSeriesPoint>> pointChildren, Dictionary<string, List<TimeSeriesPoint>> anomalyChildren, List<string> aggDim)
        {
            SortedDictionary<BestDimension, double> entropyMap = new SortedDictionary<BestDimension, double>();
            Dictionary<BestDimension, double> entropyRatioMap = new Dictionary<BestDimension, double>();
            double sumGain = 0;

            foreach (string dimKey in aggDim)
            {
                BestDimension dimension = new BestDimension();
                dimension.DimensionKey = dimKey;

                if (pointChildren.ContainsKey(dimKey))
                {
                    UpdateDistribution(dimension.PointDis, pointChildren[dimKey], dimKey);
                }
                if (anomalyChildren.ContainsKey(dimKey))
                {
                    UpdateDistribution(dimension.AnomalyDis, anomalyChildren[dimKey], dimKey);
                }

                double entropy = GetEntropy(dimension.PointDis.Count, dimension.AnomalyDis.Count);
                if (Double.IsNaN(entropy))
                {
                    entropy = Double.MaxValue;
                }
                entropyMap.Add(dimension, entropy);

                double gainRatio = entropy / GetDimensionIntrinsicValue(dimension.PointDis);

                if (Double.IsInfinity(gainRatio))
                {
                    gainRatio = 0;
                }
                entropyRatioMap.Add(dimension, gainRatio);
                dimension.GainRatio = gainRatio;

                sumGain += entropy;
            }

            double meanGain = sumGain / aggDim.Count;

            return OrderDimensions(entropyMap, entropyRatioMap, meanGain, false);
        }

        private AnomalyDirection GetRootCauseDirection(TimeSeriesPoint rootCausePoint)
        {
            if (rootCausePoint.ExpectedValue < rootCausePoint.Value)
            {
                return AnomalyDirection.Up;
            }
            else if (rootCausePoint.ExpectedValue > rootCausePoint.Value)
            {
                return AnomalyDirection.Down;
            }
            else
            {
                return AnomalyDirection.Same;
            }
        }

        private void GetRootCauseDirectionAndScore(Dictionary<Dictionary<string, Object>, TimeSeriesPoint> dimPointMapping, Dictionary<string, Object> anomalyRoot, RootCause dst, double beta, PointTree pointTree, AggregateType aggType, Object aggSymbol)
        {
            TimeSeriesPoint anomalyPoint = GetPointByDimension(dimPointMapping, anomalyRoot, pointTree, aggType, aggSymbol);
            if (dst.Items.Count > 1)
            {
                //get surprise value and explanatory power value
                List<RootCauseScore> scoreList = new List<RootCauseScore>();

                foreach (RootCauseItem item in dst.Items)
                {
                    TimeSeriesPoint rootCausePoint = GetPointByDimension(dimPointMapping, item.Dimension, pointTree, aggType, aggSymbol);
                    if (anomalyPoint != null && rootCausePoint != null)
                    {
                        Tuple<double, double> scores = GetSurpriseAndExplanatoryScore(rootCausePoint, anomalyPoint);
                        scoreList.Add(new RootCauseScore(scores.Item1, scores.Item2));
                        item.Direction = GetRootCauseDirection(rootCausePoint);
                    }
                }

                //get final score
                for (int i = 0; i < scoreList.Count; i++)
                {
                    if (aggType.Equals(AggregateType.Max) || aggType.Equals(AggregateType.Min))
                    {
                        dst.Items[i].Score = 1;
                    }
                    else
                    {
                        dst.Items[i].Score = GetFinalScore(scoreList[i].Surprise, Math.Abs(scoreList[i].ExplanatoryScore), beta);
                    }
                }
            }
            else if (dst.Items.Count == 1)
            {
                TimeSeriesPoint rootCausePoint = GetPointByDimension(dimPointMapping, dst.Items[0].Dimension, pointTree, aggType, aggSymbol);
                if (anomalyPoint != null && rootCausePoint != null)
                {
                    Tuple<double, double> scores = GetSurpriseAndExplanatoryScore(rootCausePoint, anomalyPoint);
                    if (aggType.Equals(AggregateType.Max) || aggType.Equals(AggregateType.Min))
                    {
                        dst.Items[0].Score = 1;
                    }
                    else
                    {
                        dst.Items[0].Score = GetFinalScore(scores.Item1, scores.Item2, beta);
                    }
                    dst.Items[0].Direction = GetRootCauseDirection(rootCausePoint);
                }
            }
        }

        private TimeSeriesPoint GetPointByDimension(Dictionary<Dictionary<string, Object>, TimeSeriesPoint> dimPointMapping, Dictionary<string, Object> dimension, PointTree pointTree, AggregateType aggType, Object aggSymbol)
        {
            if (dimPointMapping.ContainsKey(dimension))
            {
                return dimPointMapping[dimension];
            }

            int count = 0;
            TimeSeriesPoint p = new TimeSeriesPoint(dimension);
            DimensionInfo dimensionInfo = SeparateDimension(dimension, aggSymbol);
            Dictionary<string, Object> subDim = GetSubDim(dimension, dimensionInfo.DetailDims);

            foreach (TimeSeriesPoint leave in pointTree.Leaves)
            {
                if (ContainsAll(leave.Dimension, subDim))
                {
                    count++;

                    p.Value = +leave.Value;
                    p.ExpectedValue = +leave.ExpectedValue;
                    p.Delta = +leave.Delta;
                }

            }
            if (aggType.Equals(AggregateType.Avg))
            {
                p.Value = p.Value / count;
                p.ExpectedValue = p.ExpectedValue / count;
                p.Delta = p.Delta / count;
            }

            if (count > 0)
            {
                return p;
            }
            else
            {
                return null;
            }
        }

        private void BuildTree(PointTree tree, List<string> aggDims, TimeSeriesPoint point, Object aggSymbol)
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
                    tree.ChildrenNodes.Add(nextDim, new List<TimeSeriesPoint>());
                }
                tree.ChildrenNodes[nextDim].Add(point);
            }

            if (aggNum == 0)
            {
                tree.Leaves.Add(point);
            }
        }

        private IEnumerable<BestDimension> OrderDimensions(SortedDictionary<BestDimension, double> valueMap, Dictionary<BestDimension, double> valueRatioMap, double meanGain, bool isLeavesLevel = true)
        {
            List<KeyValuePair<BestDimension, double>> valueMapAsList = valueMap.ToList();
            List<BestDimension> ordered = new List<BestDimension>();

            BestDimension best;
            do
            {
                best = null;

                foreach (KeyValuePair<BestDimension, double> dimension in valueMapAsList)
                {
                    if (dimension.Key.AnomalyDis.Count == 1 || (isLeavesLevel ? dimension.Value >= meanGain : dimension.Value <= meanGain))
                    {
                        if (best == null)
                        {
                            best = dimension.Key;
                        }
                        else
                        {
                            bool isRatioNan = Double.IsNaN(valueRatioMap[best]);
                            if (dimension.Key.AnomalyDis.Count > 1)
                            {
                                if (best.AnomalyDis.Count != 1 && !isRatioNan && (isLeavesLevel ? valueRatioMap[best].CompareTo(dimension.Value) <= 0 : valueRatioMap[best].CompareTo(dimension.Value) >= 0))
                                {
                                    best = GetBestDimension(best, dimension, valueRatioMap);
                                }
                            }
                            else if (dimension.Key.AnomalyDis.Count == 1)
                            {

                                if (best.AnomalyDis.Count > 1)
                                {
                                    best = dimension.Key;
                                }
                                else if (best.AnomalyDis.Count == 1)
                                {
                                    if (!isRatioNan && (isLeavesLevel ? valueRatioMap[best].CompareTo(dimension.Value) <= 0 : valueRatioMap[best].CompareTo(dimension.Value) >= 0))
                                    {
                                        best = GetBestDimension(best, dimension, valueRatioMap);
                                    }
                                }
                            }
                        }
                    }
                }

                if (best != null)
                {
                    valueMapAsList.RemoveAll(kv => kv.Key == best);
                    ordered.Add(best);
                }
            } while (best != null);

            return ordered;
        }

        private BestDimension GetBestDimension(BestDimension best, KeyValuePair<BestDimension, double> dimension, Dictionary<BestDimension, Double> valueRatioMap)
        {
            if (valueRatioMap[best].CompareTo(dimension.Value) == 0)
            {
                if (dimension.Key.AnomalyDis.Count != dimension.Key.PointDis.Count)
                {
                    best = dimension.Key;
                }
            }
            else
            {
                best = dimension.Key;
            }
            return best;
        }

        /// <summary>
        /// Calculate the surprise score according to root cause point and anomaly point
        /// </summary>
        /// <param name="rootCausePoint">A point which has been detected as root cause</param>
        /// <param name="anomalyPoint">The anomaly point</param>
        /// <remarks>
        /// <format type="text/markdown">
        ///  [!include[io](~/../docs/samples/docs/api-reference/time-series-root-cause-surprise-score.md)]
        /// </format>
        /// </remarks>
        /// <returns>Surprise score</returns>
        private double GetSurpriseScore(TimeSeriesPoint rootCausePoint, TimeSeriesPoint anomalyPoint)
        {
            double p;
            double q;

            if (anomalyPoint.ExpectedValue == 0)
            {
                p = 0;
            }
            else
            {
                p = rootCausePoint.ExpectedValue / anomalyPoint.ExpectedValue;
            }

            if (anomalyPoint.Value == 0)
            {
                q = 0;
            }
            else
            {
                q = rootCausePoint.Value / anomalyPoint.Value;
            }

            double surprise = 0;

            if (p == 0)
            {
                surprise = 0.5 * (q * Log2(2 * q / (p + q)));
            }
            else if (q == 0)
            {
                surprise = 0.5 * (p * Log2(2 * p / (p + q)));
            }
            else
            {
                surprise = 0.5 * (p * Log2(2 * p / (p + q)) + q * Log2(2 * q / (p + q)));
            }

            return surprise;
        }

        private double GetFinalScore(double surprise, double ep, double beta)
        {
            double a = 0;
            double b = 0;
            if (surprise == 0)
            {
                a = 0;
            }
            if (ep == 0)
            {
                b = 0;
            }
            else
            {
                a = (1 - Math.Pow(2, -surprise));
                if (Double.IsNaN(a))
                {
                    a = 1;
                }
                b = (1 - Math.Pow(2, -ep));
            }

            return beta * a + (1 - beta) * b;
        }

        private Tuple<double, double> GetSurpriseAndExplanatoryScore(TimeSeriesPoint rootCausePoint, TimeSeriesPoint anomalyPoint)
        {
            double surprise = GetSurpriseScore(rootCausePoint, anomalyPoint);

            double ep = anomalyPoint.Value - anomalyPoint.ExpectedValue == 0 ? 0 : Math.Abs((rootCausePoint.Value - rootCausePoint.ExpectedValue) / (anomalyPoint.Value - anomalyPoint.ExpectedValue));

            return new Tuple<double, double>(surprise, ep);
        }

        private static Dictionary<string, Object> UpdateDimensionValue(Dictionary<string, Object> dimension, string key, Object value)
        {
            Dictionary<string, Object> newDim = new Dictionary<string, Object>(dimension);
            newDim[key] = value;
            return newDim;
        }

        private bool StopAnomalyComparison(double preTotal, double parent, double current, double pre)
        {
            if (Math.Abs(preTotal) < Math.Abs(parent) * _rootCauseThreshold)
            {
                return false;
            }

            return Math.Abs(pre) / Math.Abs(current) > _anomalyPreDeltaThreshold;
        }

        private bool ShouldSeparateAnomaly(double total, double parent, int totalSize, int size)
        {
            if (Math.Abs(total) < Math.Abs(parent) * _rootCauseThreshold)
            {
                return false;
            }

            if (size == totalSize && size == 1)
            {
                return true;
            }

            return size <= totalSize * _anomalyRatioThreshold;
        }

        private double GetDimensionEntropy(Dictionary<string, int> pointDis, Dictionary<string, int> anomalyDis)
        {
            int total = GetTotalNumber(pointDis);
            double entropy = 0;

            foreach (string key in anomalyDis.Keys)
            {
                double dimEntropy = GetEntropy(pointDis[key], anomalyDis[key]);
                entropy += dimEntropy * pointDis[key] / total;
            }

            return entropy;
        }

        private double GetDimensionIntrinsicValue(Dictionary<string, int> pointDis)
        {
            int total = GetTotalNumber(pointDis);
            double intrinsicValue = 0;

            foreach (string key in pointDis.Keys)
            {
                intrinsicValue -= Log2((double)pointDis[key] / total) * (double)pointDis[key] / total;
            }

            return intrinsicValue;
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

        private void UpdateDistribution(Dictionary<string, int> distribution, List<TimeSeriesPoint> points, string dimKey)
        {
            foreach (TimeSeriesPoint point in points)
            {
                string dimVal = Convert.ToString(point.Dimension[dimKey]);
                if (!distribution.ContainsKey(dimVal))
                {
                    distribution.Add(dimVal, 0);
                }
                distribution[dimVal] = distribution[dimVal] + 1;
            }
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        private double Log2(double val) => Double.IsNaN(val) ? 0 : Math.Log(val) / Math.Log(2);

        private static bool ContainsAll(Dictionary<string, Object> bigDictionary, Dictionary<string, Object> smallDictionary)
        {
            foreach (var item in smallDictionary)
            {
                if (!bigDictionary.ContainsKey(item.Key) || !object.Equals(bigDictionary[item.Key], smallDictionary[item.Key]))
                {
                    return false;
                }
            }

            return true;
        }

        private bool IsAggregationDimension(Object val, Object aggSymbol)
        {
            return object.Equals(val, aggSymbol);
        }
    }

    internal class DimensionInfo
    {
        internal List<string> DetailDims { get; set; }
        internal List<string> AggDims { get; set; }

        public DimensionInfo()
        {
            DetailDims = new List<string>();
            AggDims = new List<string>();
        }
    }

    internal class PointTree
    {
        internal TimeSeriesPoint ParentNode;
        internal Dictionary<string, List<TimeSeriesPoint>> ChildrenNodes;
        internal List<TimeSeriesPoint> Leaves;

        public PointTree()
        {
            Leaves = new List<TimeSeriesPoint>();
            ChildrenNodes = new Dictionary<string, List<TimeSeriesPoint>>();
        }
    }

    internal class BestDimension : IComparable
    {
        internal string DimensionKey;
        internal Dictionary<string, int> AnomalyDis;
        internal Dictionary<string, int> PointDis;
        internal double Gain;
        internal double GainRatio;

        public BestDimension()
        {
            AnomalyDis = new Dictionary<string, int>();
            PointDis = new Dictionary<string, int>();
        }

        public int CompareTo(object obj)
        {
            if (obj == null) return 1;

            BestDimension other = obj as BestDimension;
            if (other != null)
                return DimensionKey.CompareTo(other.DimensionKey);
            else
                throw new ArgumentException("Object is not a BestDimension");
        }
    }

    internal class RootCauseScore
    {
        internal double Surprise;
        internal double ExplanatoryScore;

        public RootCauseScore(double surprise, double explanatoryScore)
        {
            Surprise = surprise;
            ExplanatoryScore = explanatoryScore;
        }
    }

    internal class DimensionComparer : EqualityComparer<Dictionary<string, object>>
    {
        public override bool Equals(Dictionary<string, object> x, Dictionary<string, object> y)
        {
            if (x == null && y == null)
            {
                return true;
            }
            if ((x == null && y != null) || (x != null && y == null))
            {
                return false;
            }
            if (x.Count != y.Count)
            {
                return false;
            }
            if (x.Keys.Except(y.Keys).Any())
            {
                return false;
            }
            if (y.Keys.Except(x.Keys).Any())
            {
                return false;
            }
            foreach (var pair in x)
            {
                if (!object.Equals(pair.Value, y[pair.Key]))
                {
                    return false;
                }
            }
            return true;
        }

        public override int GetHashCode(Dictionary<string, object> obj)
        {
            int code = 0;
            foreach (KeyValuePair<string, object> pair in obj)
                code = code ^ pair.GetHashCode();
            return code;
        }
    }
}
