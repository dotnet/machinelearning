// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.TimeSeries
{
    public sealed class RootCause
    {
        /// <summary>
        /// A List for root cause item. Instance of the item should be <see cref="RootCauseItem"/>.
        /// </summary>
        public List<RootCauseItem> Items { get; set; }

        /// <summary>
        /// The gain for the potential root cause
        /// </summary>
        public double Gain { get; set; }

        /// <summary>
        /// The gain ratio for the potential root cause
        /// </summary>
        public double GainRatio { get; set; }

        public RootCause()
        {
            Items = new List<RootCauseItem>();
        }
    }

    public sealed class RootCauseLocalizationInput
    {
        /// <summary>
        /// When the anomaly incident occurs.
        /// </summary>
        public DateTime AnomalyTimestamp { get; set; }

        /// <summary>
        /// Point with the anomaly dimension must exist in the slice list at the anomaly timestamp, or the library will not calculate the root cause.
        /// </summary>
        public Dictionary<string, Object> AnomalyDimension { get; set; }

        /// <summary>
        /// A list of points at different timestamp. If the slices don't contain point data corresponding to the anomaly timestamp, the root cause localization alogorithm will not calculate the root cause as no information at the anomaly timestamp is provided.
        /// </summary>
        public List<MetricSlice> Slices { get; set; }

        /// <summary>
        /// The aggregated type, the type should be  <see cref="TimeSeries.AggregateType"/>.
        /// </summary>
        public AggregateType AggregateType { get; set; }

        /// <summary>
        /// The string you defined as a aggregated symbol in the AnomalyDimension and point dimension.
        /// </summary>
        public Object AggregateSymbol { get; set; }

        public RootCauseLocalizationInput(DateTime anomalyTimestamp, Dictionary<string, Object> anomalyDimension, List<MetricSlice> slices, AggregateType aggregateType, Object aggregateSymbol)
        {
            AnomalyTimestamp = anomalyTimestamp;
            AnomalyDimension = anomalyDimension;
            Slices = slices;
            AggregateType = aggregateType;
            AggregateSymbol = aggregateSymbol;
        }

        public RootCauseLocalizationInput(DateTime anomalyTimestamp, Dictionary<string, Object> anomalyDimension, List<MetricSlice> slices, Object aggregateSymbol)
        {
            AnomalyTimestamp = anomalyTimestamp;
            AnomalyDimension = anomalyDimension;
            Slices = slices;
            AggregateType = AggregateType.Unknown;
            AggregateSymbol = aggregateSymbol;
        }

        public RootCauseLocalizationInput() { }
    }

    public enum AggregateType
    {
        /// <summary>
        /// Make the aggregate type as unknown type.
        /// </summary>
        Unknown = 0,
        /// <summary>
        /// Make the aggregate type as summation.
        /// </summary>
        Sum = 1,
        /// <summary>
        /// Make the aggregate type as average.
        ///  </summary>
        Avg = 2,
        /// <summary>
        /// Make the aggregate type as min.
        /// </summary>
        Min = 3,
        /// <summary>
        /// Make the aggregate type as max.
        /// </summary>
        Max = 4
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
        Down = 1,
        /// <summary>
        /// the value is the same as expected value.
        ///  </summary>
        Same = 2
    }

    public sealed class RootCauseItem : IEquatable<RootCauseItem>
    {
        /// <summary>
        ///The score is a value to evaluate the contribution to the anomaly incident. The range is between [0,1]. The larger the score, the root cause contributes the most to the anomaly. The parameter beta has an influence on this score. For how the score is calculated, you can refer to the source code.
        ///</summary>
        public double Score;
        /// <summary>
        /// Path is a list of the dimension key that the library selected for you. In this root cause localization library, for one time call for the library, the path will be obtained and the length of path list will always be 1. Different RootCauseItem obtained from one library call will have the same path as it is the best dimension selected for the input.
        /// </summary>
        public List<string> Path;
        /// <summary>
        /// The dimension for the detected root cause point.
        /// </summary>
        public Dictionary<string, Object> Dimension;
        /// <summary>
        /// The direction for the detected root cause point, should be <see cref="AnomalyDirection"/>.
        /// </summary>
        public AnomalyDirection Direction;

        public RootCauseItem(Dictionary<string, Object> rootCause)
        {
            Dimension = rootCause;
            Path = new List<string>();
        }

        public RootCauseItem(Dictionary<string, Object> rootCause, List<string> path)
        {
            Dimension = rootCause;
            Path = path;
        }
        public bool Equals(RootCauseItem other)
        {
            if (Dimension.Count == other.Dimension.Count)
            {
                foreach (KeyValuePair<string, Object> item in Dimension)
                {
                    if (!object.Equals(other.Dimension[item.Key], item.Value))
                    {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }
    }

    public sealed class MetricSlice
    {
        /// <summary>
        /// Timestamp for the point list.
        /// </summary>
        public DateTime TimeStamp { get; set; }
        /// <summary>
        /// A list of points
        /// </summary>
        public List<TimeSeriesPoint> Points { get; set; }

        public MetricSlice(DateTime timeStamp, List<TimeSeriesPoint> points)
        {
            TimeStamp = timeStamp;
            Points = points;
        }

        public MetricSlice() { }
    }

    public sealed class TimeSeriesPoint : IEquatable<TimeSeriesPoint>
    {
        /// <summary>
        /// Value of a time series point.
        /// </summary>
        public double Value { get; set; }
        /// <summary>
        /// Forecasted value for the time series point.
        /// </summary>
        public double ExpectedValue { get; set; }
        /// <summary>
        /// Whether the point is an anomaly point.
        /// </summary>
        public bool IsAnomaly { get; set; }
        /// <summary>
        /// Dimension information for the point. For example, City = New York City, Dataceter = DC1. The value for this dictionary is an object, when the Dimension is used, the equals function for the Object will be used. If you have a customized class, you need to define the Equals function.
        /// </summary>
        public Dictionary<string, Object> Dimension { get; set; }
        /// <summary>
        /// Difference between value and expected value.
        /// </summary>
        public double Delta { get; set; }

        public TimeSeriesPoint(Dictionary<string, Object> dimension)
        {
            Dimension = dimension;
        }
        public TimeSeriesPoint() { }

        public TimeSeriesPoint(double value, double expectedValue, bool isAnomaly, Dictionary<string, Object> dimension)
        {
            Value = value;
            ExpectedValue = expectedValue;
            IsAnomaly = isAnomaly;
            Dimension = dimension;
            Delta = value - expectedValue;
        }

        public bool Equals(TimeSeriesPoint other)
        {
            foreach (KeyValuePair<string, Object> item in Dimension)
            {
                if (!object.Equals(other.Dimension[item.Key], item.Value))
                {
                    return false;
                }
            }
            return true;
        }

        public override int GetHashCode()
        {
            return Dimension.GetHashCode();
        }
    }
}
