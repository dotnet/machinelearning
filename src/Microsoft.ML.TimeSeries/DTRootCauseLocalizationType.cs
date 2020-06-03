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
}
