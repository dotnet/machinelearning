// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// Allows a member to be marked as a <see cref="RootCauseLocalizationInputDataViewType"/>, primarily allowing one to set
    /// root cause localization input.
    /// </summary>
    public sealed class RootCauseLocalizationInputTypeAttribute : DataViewTypeAttribute
    {
        /// <summary>
        /// Create a root cause localizagin input type.
        /// </summary>
        public RootCauseLocalizationInputTypeAttribute()
        {
        }

        /// <summary>
        /// Equal function.
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (!(other is RootCauseLocalizationInputTypeAttribute otherAttribute))
                return false;
            return true;
        }

        /// <summary>
        /// Produce the same hash code for all RootCauseLocalizationInputTypeAttribute.
        /// </summary>
        public override int GetHashCode()
        {
            return 0;
        }

        public override void Register()
        {
            DataViewTypeManager.Register(new RootCauseLocalizationInputDataViewType(), typeof(RootCauseLocalizationInput), this);
        }
    }

    /// <summary>
    /// Allows a member to be marked as a <see cref="RootCauseDataViewType"/>, primarily allowing one to set
    /// root cause result.
    /// </summary>
    public sealed class RootCauseTypeAttribute : DataViewTypeAttribute
    {
        /// <summary>
        /// Create an root cause type.
        /// </summary>
        public RootCauseTypeAttribute()
        {
        }

        /// <summary>
        /// RootCauseTypeAttribute with the same type should equal.
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (other is RootCauseTypeAttribute otherAttribute)
                return true;
            return false;
        }

        /// <summary>
        /// Produce the same hash code for all RootCauseTypeAttribute.
        /// </summary>
        public override int GetHashCode()
        {
            return 0;
        }

        public override void Register()
        {
            DataViewTypeManager.Register(new RootCauseDataViewType(), typeof(RootCause), this);
        }
    }

    public sealed class RootCause
    {
        public List<RootCauseItem> Items { get; set; }
        public RootCause()
        {
            Items = new List<RootCauseItem>();
        }
    }

    public sealed class RootCauseLocalizationInput
    {
        public DateTime AnomalyTimestamp { get; set; }

        public Dictionary<string, string> AnomalyDimensions { get; set; }

        public List<MetricSlice> Slices { get; set; }

        public AggregateType AggType { get; set; }

        public string AggSymbol { get; set; }

        public RootCauseLocalizationInput(DateTime anomalyTimestamp, Dictionary<string, string> anomalyDimensions, List<MetricSlice> slices, AggregateType aggregateType, string aggregateSymbol)
        {
            AnomalyTimestamp = anomalyTimestamp;
            AnomalyDimensions = anomalyDimensions;
            Slices = slices;
            AggType = aggregateType;
            AggSymbol = aggregateSymbol;
        }
        public void Dispose()
        {
            AnomalyDimensions = null;
            Slices = null;
        }
    }

    public sealed class RootCauseDataViewType : StructuredDataViewType
    {
        public RootCauseDataViewType()
           : base(typeof(RootCause))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            if (!(other is RootCauseDataViewType tmp))
                return false;
            return true;
        }

        public override int GetHashCode()
        {
            return 0;
        }

        public override string ToString()
        {
            return typeof(RootCauseDataViewType).Name;
        }
    }

    public sealed class RootCauseLocalizationInputDataViewType : StructuredDataViewType
    {
        public RootCauseLocalizationInputDataViewType()
           : base(typeof(RootCauseLocalizationInput))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (!(other is RootCauseLocalizationInputDataViewType tmp))
                return false;
            return true;
        }

        public override int GetHashCode()
        {
            return 0;
        }

        public override string ToString()
        {
            return typeof(RootCauseLocalizationInputDataViewType).Name;
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

    public sealed class Point : IEquatable<Point>
    {
        public double Value { get; set; }
        public double ExpectedValue { get; set; }
        public bool IsAnomaly { get; set; }
        public Dictionary<string, string> Dimension { get; set; }

        public double Delta { get; set; }

        public Point( Dictionary<string, string> dimensions)
        {
            Dimension = dimensions;
        }
        public Point(double value, double expectedValue, bool isAnomaly, Dictionary<string, string> dimensions)
        {
            Value = value;
            ExpectedValue = expectedValue;
            IsAnomaly = isAnomaly;
            Dimension = dimensions;
            Delta = value - expectedValue;
        }

        public bool Equals(Point other)
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

        public override int GetHashCode()
        {
            return Dimension.GetHashCode();
        }
    }

}
