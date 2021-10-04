// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.AutoML
{
    internal abstract class BaseParamArguments
    {
        // Parameter name
        public string Name;
    }

    internal abstract class NumericParamArguments : BaseParamArguments
    {
        // Number of steps for grid run-through.
        public int NumSteps;

        // Amount of increment between steps (multiplicative if log).
        public Double? StepSize;

        // Log scale.
        public bool LogBase;

        public NumericParamArguments()
        {
            NumSteps = 100;
            StepSize = null;
            LogBase = false;
        }
    }

    internal class FloatParamArguments : NumericParamArguments
    {
        // Minimum value
        public float Min;

        // Maximum value
        public float Max;
    }

    internal class LongParamArguments : NumericParamArguments
    {
        // Minimum value
        public long Min;

        // Maximum value
        public long Max;
    }

    internal class DiscreteParamArguments : BaseParamArguments
    {
        // Values
        public string[] Values;
    }

    internal sealed class LongParameterValue : IParameterValue<long>
    {
        private readonly string _name;
        private readonly string _valueText;
        private readonly long _value;

        public string Name
        {
            get { return _name; }
        }

        public string ValueText
        {
            get { return _valueText; }
        }

        public long Value
        {
            get { return _value; }
        }

        public LongParameterValue(string name, long value)
        {
            _name = name;
            _value = value;
            _valueText = _value.ToString("D", CultureInfo.InvariantCulture);
        }

        public bool Equals(IParameterValue other)
        {
            return Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var lpv = obj as LongParameterValue;
            return lpv != null && Name == lpv.Name && _value == lpv._value;
        }

        public override int GetHashCode()
        {
            return Hashing.CombinedHash(0, typeof(LongParameterValue), _name, _value);
        }
    }

    internal sealed class FloatParameterValue : IParameterValue<float>
    {
        private readonly string _name;
        private readonly string _valueText;
        private readonly float _value;

        public string Name
        {
            get { return _name; }
        }

        public string ValueText
        {
            get { return _valueText; }
        }

        public float Value
        {
            get { return _value; }
        }

        public FloatParameterValue(string name, float value)
        {
            Runtime.Contracts.Assert(!float.IsNaN(value));
            _name = name;
            _value = value;
            _valueText = _value.ToString("R", CultureInfo.InvariantCulture);
        }

        public bool Equals(IParameterValue other)
        {
            return Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var fpv = obj as FloatParameterValue;
            return fpv != null && Name == fpv.Name && _value == fpv._value;
        }

        public override int GetHashCode()
        {
            return Hashing.CombinedHash(0, typeof(FloatParameterValue), _name, _value);
        }
    }

    internal sealed class StringParameterValue : IParameterValue<string>
    {
        private readonly string _name;
        private readonly string _value;

        public string Name
        {
            get { return _name; }
        }

        public string ValueText
        {
            get { return _value; }
        }

        public string Value
        {
            get { return _value; }
        }

        public StringParameterValue(string name, string value)
        {
            _name = name;
            _value = value;
        }

        public bool Equals(IParameterValue other)
        {
            return Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var spv = obj as StringParameterValue;
            return spv != null && Name == spv.Name && ValueText == spv.ValueText;
        }

        public override int GetHashCode()
        {
            return Hashing.CombinedHash(0, typeof(StringParameterValue), _name, _value);
        }
    }

    internal interface INumericValueGenerator : IValueGenerator
    {
        float NormalizeValue(IParameterValue value);
        bool InRange(IParameterValue value);
    }

    /// <summary>
    /// The integer type parameter sweep.
    /// </summary>
    internal class LongValueGenerator : INumericValueGenerator
    {
        private readonly LongParamArguments _args;
        private IParameterValue[] _gridValues;

        public string Name { get { return _args.Name; } }

        public LongValueGenerator(LongParamArguments args)
        {
            Runtime.Contracts.Assert(args.Min < args.Max, "min must be less than max");
            // REVIEW: this condition can be relaxed if we change the math below to deal with it
            Runtime.Contracts.Assert(!args.LogBase || args.Min > 0, "min must be positive if log scale is used");
            Runtime.Contracts.Assert(!args.LogBase || args.StepSize == null || args.StepSize > 1, "StepSize must be greater than 1 if log scale is used");
            Runtime.Contracts.Assert(args.LogBase || args.StepSize == null || args.StepSize > 0, "StepSize must be greater than 0 if linear scale is used");
            _args = args;
        }

        // REVIEW: Is Float accurate enough?
        public IParameterValue CreateFromNormalized(Double normalizedValue)
        {
            long val;
            if (_args.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = !_args.StepSize.HasValue
                    ? Math.Pow(1.0 * _args.Max / _args.Min, 1.0 / (_args.NumSteps - 1))
                    : _args.StepSize.Value;
                var logMax = Math.Log(_args.Max, logBase);
                var logMin = Math.Log(_args.Min, logBase);
                val = (long)(_args.Min * Math.Pow(logBase, normalizedValue * (logMax - logMin)));
            }
            else
                val = (long)(_args.Min + normalizedValue * (_args.Max - _args.Min));

            return new LongParameterValue(_args.Name, val);
        }

        private void EnsureParameterValues()
        {
            if (_gridValues != null)
                return;

            var result = new List<IParameterValue>();
            if ((_args.StepSize == null && _args.NumSteps > (_args.Max - _args.Min)) ||
                (_args.StepSize != null && _args.StepSize <= 1))
            {
                for (long i = _args.Min; i <= _args.Max; i++)
                    result.Add(new LongParameterValue(_args.Name, i));
            }
            else
            {
                if (_args.LogBase)
                {
                    // REVIEW: review the math below, it only works for positive Min and Max
                    var logBase = _args.StepSize ?? Math.Pow(1.0 * _args.Max / _args.Min, 1.0 / (_args.NumSteps - 1));

                    long prevValue = long.MinValue;
                    var maxPlusEpsilon = _args.Max * Math.Sqrt(logBase);
                    for (Double value = _args.Min; value <= maxPlusEpsilon; value *= logBase)
                    {
                        var longValue = (long)value;
                        if (longValue > prevValue)
                            result.Add(new LongParameterValue(_args.Name, longValue));
                        prevValue = longValue;
                    }
                }
                else
                {
                    var stepSize = _args.StepSize ?? (Double)(_args.Max - _args.Min) / (_args.NumSteps - 1);
                    long prevValue = long.MinValue;
                    var maxPlusEpsilon = _args.Max + stepSize / 2;
                    for (Double value = _args.Min; value <= maxPlusEpsilon; value += stepSize)
                    {
                        var longValue = (long)value;
                        if (longValue > prevValue)
                            result.Add(new LongParameterValue(_args.Name, longValue));
                        prevValue = longValue;
                    }
                }
            }
            _gridValues = result.ToArray();
        }

        public IParameterValue this[int i]
        {
            get
            {
                EnsureParameterValues();
                return _gridValues[i];
            }
        }

        public int Count
        {
            get
            {
                EnsureParameterValues();
                return _gridValues.Length;
            }
        }

        public float NormalizeValue(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            Runtime.Contracts.Assert(valueTyped != null, "LongValueGenerator could not normalized parameter because it is not of the correct type");
            Runtime.Contracts.Assert(_args.Min <= valueTyped.Value && valueTyped.Value <= _args.Max, "Value not in correct range");

            if (_args.LogBase)
            {
                float logBase = (float)(_args.StepSize ?? Math.Pow(1.0 * _args.Max / _args.Min, 1.0 / (_args.NumSteps - 1)));
                return (float)((Math.Log(valueTyped.Value, logBase) - Math.Log(_args.Min, logBase)) / (Math.Log(_args.Max, logBase) - Math.Log(_args.Min, logBase)));
            }
            else
                return (float)(valueTyped.Value - _args.Min) / (_args.Max - _args.Min);
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            return (_args.Min <= valueTyped.Value && valueTyped.Value <= _args.Max);
        }
    }

    /// <summary>
    /// The floating point type parameter sweep.
    /// </summary>
    internal class FloatValueGenerator : INumericValueGenerator
    {
        private readonly FloatParamArguments _args;
        private IParameterValue[] _gridValues;

        public string Name { get { return _args.Name; } }

        public FloatValueGenerator(FloatParamArguments args)
        {
            Runtime.Contracts.Assert(args.Min < args.Max, "min must be less than max");
            // REVIEW: this condition can be relaxed if we change the math below to deal with it
            Runtime.Contracts.Assert(!args.LogBase || args.Min > 0, "min must be positive if log scale is used");
            Runtime.Contracts.Assert(!args.LogBase || args.StepSize == null || args.StepSize > 1, "StepSize must be greater than 1 if log scale is used");
            Runtime.Contracts.Assert(args.LogBase || args.StepSize == null || args.StepSize > 0, "StepSize must be greater than 0 if linear scale is used");
            _args = args;
        }

        // REVIEW: Is Float accurate enough?
        public IParameterValue CreateFromNormalized(Double normalizedValue)
        {
            float val;
            if (_args.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = !_args.StepSize.HasValue
                    ? Math.Pow(1.0 * _args.Max / _args.Min, 1.0 / (_args.NumSteps - 1))
                    : _args.StepSize.Value;
                var logMax = Math.Log(_args.Max, logBase);
                var logMin = Math.Log(_args.Min, logBase);
                val = (float)(_args.Min * Math.Pow(logBase, normalizedValue * (logMax - logMin)));
            }
            else
                val = (float)(_args.Min + normalizedValue * (_args.Max - _args.Min));

            return new FloatParameterValue(_args.Name, val);
        }

        private void EnsureParameterValues()
        {
            if (_gridValues != null)
                return;

            var result = new List<IParameterValue>();
            if (_args.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = _args.StepSize ?? Math.Pow(1.0 * _args.Max / _args.Min, 1.0 / (_args.NumSteps - 1));

                float prevValue = float.NegativeInfinity;
                var maxPlusEpsilon = _args.Max * Math.Sqrt(logBase);
                for (Double value = _args.Min; value <= maxPlusEpsilon; value *= logBase)
                {
                    var floatValue = (float)value;
                    if (floatValue > prevValue)
                        result.Add(new FloatParameterValue(_args.Name, floatValue));
                    prevValue = floatValue;
                }
            }
            else
            {
                var stepSize = _args.StepSize ?? (Double)(_args.Max - _args.Min) / (_args.NumSteps - 1);
                float prevValue = float.NegativeInfinity;
                var maxPlusEpsilon = _args.Max + stepSize / 2;
                for (Double value = _args.Min; value <= maxPlusEpsilon; value += stepSize)
                {
                    var floatValue = (float)value;
                    if (floatValue > prevValue)
                        result.Add(new FloatParameterValue(_args.Name, floatValue));
                    prevValue = floatValue;
                }
            }

            _gridValues = result.ToArray();
        }

        public IParameterValue this[int i]
        {
            get
            {
                EnsureParameterValues();
                return _gridValues[i];
            }
        }

        public int Count
        {
            get
            {
                EnsureParameterValues();
                return _gridValues.Length;
            }
        }

        public float NormalizeValue(IParameterValue value)
        {
            var valueTyped = value as FloatParameterValue;
            Runtime.Contracts.Assert(valueTyped != null, "FloatValueGenerator could not normalized parameter because it is not of the correct type");
            Runtime.Contracts.Assert(_args.Min <= valueTyped.Value && valueTyped.Value <= _args.Max, "Value not in correct range");

            if (_args.LogBase)
            {
                float logBase = (float)(_args.StepSize ?? Math.Pow(1.0 * _args.Max / _args.Min, 1.0 / (_args.NumSteps - 1)));
                return (float)((Math.Log(valueTyped.Value, logBase) - Math.Log(_args.Min, logBase)) / (Math.Log(_args.Max, logBase) - Math.Log(_args.Min, logBase)));
            }
            else
                return (valueTyped.Value - _args.Min) / (_args.Max - _args.Min);
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as FloatParameterValue;
            Runtime.Contracts.Assert(valueTyped != null, "Parameter should be of type FloatParameterValue");
            return (_args.Min <= valueTyped.Value && valueTyped.Value <= _args.Max);
        }
    }

    /// <summary>
    /// The discrete parameter sweep.
    /// </summary>
    internal class DiscreteValueGenerator : IValueGenerator
    {
        private readonly DiscreteParamArguments _args;

        public string Name { get { return _args.Name; } }

        public DiscreteValueGenerator(DiscreteParamArguments args)
        {
            _args = args;
        }

        // REVIEW: Is Float accurate enough?
        public IParameterValue CreateFromNormalized(Double normalizedValue)
        {
            return new StringParameterValue(_args.Name, _args.Values[(int)(_args.Values.Length * normalizedValue)]);
        }

        public IParameterValue this[int i]
        {
            get
            {
                return new StringParameterValue(_args.Name, _args.Values[i]);
            }
        }

        public int Count
        {
            get
            {
                return _args.Values.Length;
            }
        }
    }
}
