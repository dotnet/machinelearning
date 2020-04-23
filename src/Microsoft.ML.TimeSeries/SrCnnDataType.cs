// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// The detect modes of SrCnn models.
    /// </summary>
    public enum SrCnnDetectMode
    {
        /// <summary>
        /// In this mode, output (IsAnomaly, RawScore, Mag).
        /// </summary>
        AnomalyOnly = 0,

        /// <summary>
        /// In this mode, output (IsAnomaly, AnomalyScore, Mag, ExpectedValue, BoundaryUnit, UpperBoundary, LowerBoundary).
        /// </summary>
        AnomalyAndMargin = 1,

        /// <summary>
        /// In this mode, output (IsAnomaly, RawScore, Mag, ExpectedValue).
        /// </summary>
        AnomalyAndExpectedValue = 2
    }

    /// <summary>
    /// Timeseries point definition for SrCnn models.
    /// </summary>
    public sealed class SrCnnTsPoint
    {
        public DateTime Timestamp { get; set; }
        public Double Value { get; set; }

        public SrCnnTsPoint(DateTime timestamp, Double value)
        {
            Timestamp = timestamp;
            Value = value;
        }
    }

    /// <summary>
    /// Dataview type of <see cref="SrCnnTsPoint"/>.
    /// </summary>
    public sealed class SrCnnTsPointDataViewType : StructuredDataViewType
    {
        public SrCnnTsPointDataViewType()
            : base(typeof(SrCnnTsPoint))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (!(other is SrCnnTsPointDataViewType otherDataViewType))
                return false;
            return true;
        }

        public override int GetHashCode()
        {
            return 0;
        }

        public override string ToString()
        {
            return typeof(SrCnnTsPoint).Name;
        }
    }

    /// <summary>
    /// Allows a member to be marked as a <see cref="SrCnnTsPoint"/>.
    /// </summary>
    public sealed class SrCnnTsPointTypeAttribute : DataViewTypeAttribute
    {
        public SrCnnTsPointTypeAttribute()
        {
        }

        public override bool Equals(DataViewTypeAttribute other)
        {
            if (!(other is SrCnnTsPointTypeAttribute otherAttribute))
                return false;
            return true;
        }

        public override int GetHashCode()
        {
            return 0;
        }

        public override void Register()
        {
            DataViewTypeManager.Register(new SrCnnTsPointDataViewType(), typeof(SrCnnTsPoint), this);
        }
    }
}
