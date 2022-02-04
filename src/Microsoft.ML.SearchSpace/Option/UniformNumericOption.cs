// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.SearchSpace.Option
{
    public abstract class UniformNumericOption : OptionBase
    {
        public UniformNumericOption(double min, double max, bool logBase = false)
        {
            Contracts.Check(max > min, "max must be larger than min.");
            Contracts.Check(min > 0 || logBase == false, "min must be larger than 0 if logBase is true.");
            this.Min = min;
            this.Max = max;
            this.LogBase = logBase;
            this.Default = Enumerable.Repeat(0.0, this.FeatureSpaceDim).ToArray();
        }

        public double Min { get; }

        public double Max { get; }

        public bool LogBase { get; }

        public override int FeatureSpaceDim => 1;

        public override int?[] Step => new int?[] { null };

        public override double[] MappingToFeatureSpace(IParameter param)
        {
            var x = param.AsType<double>();
            Contracts.Check(x <= this.Max && x >= this.Min, $"{x} is not within [{this.Min}, {this.Max}]");
            if (this.LogBase)
            {
                var logMax = Math.Log(this.Max);
                var logMin = Math.Log(this.Min);
                var logX = Math.Log(x);

                return new[] { logX / (logMax - logMin) - logMin / (logMax - logMin) };
            }
            else
            {
                return new[] { x / (this.Max - this.Min) - this.Min / (this.Max - this.Min) };
            }
        }

        public override IParameter SampleFromFeatureSpace(double[] values)
        {
            Contracts.Check(values.Length == 1, "values length must be 1");
            var value = values[0];
            Contracts.Check(value <= 1 && value >= 0, $"{value} must be between [0,1]");

            if (this.LogBase)
            {
                var order = Math.Pow(this.Max / this.Min, value);
                var res = this.Min * order;
                return Parameter.FromDouble(res);
            }
            else
            {
                return Parameter.FromDouble((this.Min + (this.Max - this.Min) * value));
            }
        }
    }

    public class UniformDoubleOption : UniformNumericOption
    {
        public UniformDoubleOption(double min, double max, bool logBase = false, double? defaultValue = null)
            : base(min, max, logBase)
        {
            if (defaultValue != null)
            {
                this.Default = this.MappingToFeatureSpace(Parameter.FromDouble(defaultValue.Value));
            }
        }
    }

    public class UniformSingleOption : UniformNumericOption
    {
        public UniformSingleOption(float min, float max, bool logBase = false, float? defaultValue = null)
            : base(min, max, logBase)
        {
            if (defaultValue != null)
            {
                this.Default = this.MappingToFeatureSpace(Parameter.FromFloat(defaultValue.Value));
            }
        }

        public override double[] MappingToFeatureSpace(IParameter param)
        {
            var singleValue = param.AsType<float>();
            var doubleValue = Convert.ToDouble(singleValue);
            return base.MappingToFeatureSpace(Parameter.FromDouble(doubleValue));
        }

        public override IParameter SampleFromFeatureSpace(double[] values)
        {
            var doubleValue = base.SampleFromFeatureSpace(values).AsType<double>();
            var singleValue = Convert.ToSingle(doubleValue);
            return Parameter.FromFloat(singleValue);
        }
    }

    public class UniformIntOption : UniformNumericOption
    {
        public UniformIntOption(int min, int max, bool logBase = false, int? defaultValue = null)
            : base(min, max, logBase)
        {
            if (defaultValue != null)
            {
                this.Default = this.MappingToFeatureSpace(Parameter.FromInt(defaultValue.Value));
            }
        }

        public override IParameter SampleFromFeatureSpace(double[] values)
        {
            var param = base.SampleFromFeatureSpace(values);
            var intValue = Convert.ToInt32(Math.Floor(param.AsType<double>()));

            return Parameter.FromInt(intValue);
        }

        public override double[] MappingToFeatureSpace(IParameter param)
        {
            var value = param.AsType<int>();
            var valueAsDouble = Convert.ToDouble(value);
            return base.MappingToFeatureSpace(Parameter.FromDouble(valueAsDouble));
        }
    }
}
