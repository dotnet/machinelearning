// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.SearchSpace.Option
{
    /// <summary>
    /// abstract class for numeric option.
    /// </summary>
    public abstract class UniformNumericOption : OptionBase
    {
        /// <summary>
        /// Create a <see cref="UniformNumericOption"/> using <paramref name="min"/> and <paramref name="max"/>.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="logBase">Indicate whether it should be log base or not.</param>
        public UniformNumericOption(double min, double max, bool logBase = false)
        {
            Contracts.Check(max > min, "max must be larger than min.");
            Contracts.Check(min > 0 || logBase == false, "min must be larger than 0 if logBase is true.");
            Min = min;
            Max = max;
            LogBase = logBase;
            Default = Enumerable.Repeat(0.0, FeatureSpaceDim).ToArray();
        }

        /// <summary>
        /// Gets minimum value of this option.
        /// </summary>
        public double Min { get; }

        /// <summary>
        /// Gets maximum value of this option.
        /// </summary>
        public double Max { get; }

        /// <summary>
        /// Gets if this option is log base or not.
        /// </summary>
        public bool LogBase { get; }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override int FeatureSpaceDim => 1;

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override int?[] Step => new int?[] { null };

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override double[] MappingToFeatureSpace(Parameter param)
        {
            var x = param.AsType<double>();
            Contracts.Check(x <= Max && x >= Min, $"{x} is not within [{Min}, {Max}]");
            if (LogBase)
            {
                var logMax = Math.Log(Max);
                var logMin = Math.Log(Min);
                var logX = Math.Log(x);

                return new[] { logX / (logMax - logMin) - logMin / (logMax - logMin) };
            }
            else
            {
                return new[] { x / (Max - Min) - Min / (Max - Min) };
            }
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            Contracts.Check(values.Length == 1, "values length must be 1");
            var value = values[0];
            Contracts.Check(value <= 1 && value >= 0, $"{value} must be between [0,1]");

            if (LogBase)
            {
                var order = Math.Pow(Max / Min, value);
                var res = Min * order;
                return Parameter.FromDouble(res);
            }
            else
            {
                return Parameter.FromDouble((Min + (Max - Min) * value));
            }
        }
    }

    /// <summary>
    /// class for uniform double option.
    /// </summary>
    public sealed class UniformDoubleOption : UniformNumericOption
    {
        /// <summary>
        /// Create a <see cref="UniformDoubleOption"/> using <paramref name="min"/>, <paramref name="max"/> and <paramref name="defaultValue"/>.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="logBase"></param>
        /// <param name="defaultValue"></param>
        public UniformDoubleOption(double min, double max, bool logBase = false, double? defaultValue = null)
            : base(min, max, logBase)
        {
            if (defaultValue != null)
            {
                Default = MappingToFeatureSpace(Parameter.FromDouble(defaultValue.Value));
            }
        }
    }

    /// <summary>
    /// class for uniform single option.
    /// </summary>
    public sealed class UniformSingleOption : UniformNumericOption
    {
        /// <summary>
        /// Create a <see cref="UniformSingleOption"/> using <paramref name="min"/>, <paramref name="max"/> and <paramref name="defaultValue"/>.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="logBase"></param>
        /// <param name="defaultValue"></param>
        public UniformSingleOption(float min, float max, bool logBase = false, float? defaultValue = null)
            : base(min, max, logBase)
        {
            if (defaultValue != null)
            {
                Default = MappingToFeatureSpace(Parameter.FromFloat(defaultValue.Value));
            }
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override double[] MappingToFeatureSpace(Parameter param)
        {
            var singleValue = param.AsType<float>();
            var doubleValue = Convert.ToDouble(singleValue);
            return base.MappingToFeatureSpace(Parameter.FromDouble(doubleValue));
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            var doubleValue = base.SampleFromFeatureSpace(values).AsType<double>();
            var singleValue = Convert.ToSingle(doubleValue);
            return Parameter.FromFloat(singleValue);
        }
    }

    /// <summary>
    /// class for uniform int option.
    /// </summary>
    public sealed class UniformIntOption : UniformNumericOption
    {
        /// <summary>
        /// Create a <see cref="UniformIntOption"/> using <paramref name="min"/>, <paramref name="max"/> and <paramref name="defaultValue"/>.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="logBase"></param>
        /// <param name="defaultValue"></param>
        public UniformIntOption(int min, int max, bool logBase = false, int? defaultValue = null)
            : base(min, max, logBase)
        {
            if (defaultValue != null)
            {
                Default = MappingToFeatureSpace(Parameter.FromInt(defaultValue.Value));
            }
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            var param = base.SampleFromFeatureSpace(values);
            var intValue = Convert.ToInt32(Math.Floor(param.AsType<double>()));

            return Parameter.FromInt(intValue);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override double[] MappingToFeatureSpace(Parameter param)
        {
            var value = param.AsType<int>();
            var valueAsDouble = Convert.ToDouble(value);
            return base.MappingToFeatureSpace(Parameter.FromDouble(valueAsDouble));
        }
    }
}
