using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.Transforms.TimeSeries
{
    public enum SrCnnDetectMode
    {
        AnomalyOnly = 0,
        AnomalyAndMargin = 1
    }

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

    public sealed class SrCnnTsPointTypeAttrbute : DataViewTypeAttribute
    {
        public SrCnnTsPointTypeAttrbute()
        {
        }

        public override bool Equals(DataViewTypeAttribute other)
        {
            if (!(other is SrCnnTsPointTypeAttrbute otherAttribute))
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
