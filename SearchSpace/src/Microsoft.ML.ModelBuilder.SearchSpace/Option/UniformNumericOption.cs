// <copyright file="UniformNumericOption.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Diagnostics.Contracts;
using System.Globalization;
using System.Linq;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Option
{
    public abstract class UniformNumericOption : OptionBase
    {
        public UniformNumericOption(double min, double max, bool logBase = false)
        {
            Contract.Requires(max > min, "max must be larger than min.");
            Contract.Requires(min > 0 || logBase == false, "min must be larger than 0 if logBase is true.");
            this.Min = min;
            this.Max = max;
            this.LogBase = logBase;
            this.Default = Enumerable.Repeat(0.0, this.FeatureSpaceDim).ToArray();
        }

        public double Min { get; }

        public double Max { get; }

        public bool LogBase { get; }

        public override int FeatureSpaceDim => 1;

        public override double[] MappingToFeatureSpace(Parameter param)
        {
            var x = param.AsType<double>();
            Contract.Requires(x < this.Max && x >= this.Min, $"{x} is not within [{this.Min}, {this.Max})");
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

        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            Contract.Requires(values.Length == 1, "values length must be 1");
            var value = values[0];
            Contract.Requires(value < 1 && value >= 0, $"{value} must be between [0,1)");

            if (this.LogBase)
            {
                var order = Math.Log(this.Min) + Math.Log(this.Max / this.Min) * value;
                return new Parameter(Math.Exp(order));
            }
            else
            {
                return new Parameter((this.Min + (this.Max - this.Min) * value));
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
                this.Default = this.MappingToFeatureSpace(new Parameter(defaultValue));
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
                this.Default = this.MappingToFeatureSpace(new Parameter(defaultValue));
            }
        }

        public override double[] MappingToFeatureSpace(Parameter param)
        {
            var singleValue = param.AsType<float>();
            var doubleValue = Convert.ToDouble(singleValue);
            return base.MappingToFeatureSpace(new Parameter(doubleValue));
        }

        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            var doubleValue = base.SampleFromFeatureSpace(values).AsType<double>();
            var singleValue = Convert.ToSingle(doubleValue);
            return new Parameter(singleValue);
        }
    }

    public class UniformIntOption : UniformNumericOption
    {
        public UniformIntOption(int min, int max, bool logBase = false, int? defaultValue = null)
            : base(min, max, logBase)
        {
            if (defaultValue != null)
            {
                this.Default = this.MappingToFeatureSpace(new Parameter(defaultValue));
            }
        }

        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            var param = base.SampleFromFeatureSpace(values);
            var intValue = Convert.ToInt32(Math.Floor(param.AsType<double>()));

            return new Parameter(intValue);
        }

        public override double[] MappingToFeatureSpace(Parameter param)
        {
            var value = param.AsType<int>();
            var valueAsDouble = Convert.ToDouble(value);
            return base.MappingToFeatureSpace(new Parameter(valueAsDouble));
        }
    }
}
