// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Sweeper;

[assembly: LoadableClass(typeof(LongValueGenerator), typeof(LongParamArguments), typeof(SignatureSweeperParameter),
    "Long parameter", "lp")]
[assembly: LoadableClass(typeof(FloatValueGenerator), typeof(FloatParamArguments), typeof(SignatureSweeperParameter),
    "Float parameter", "fp")]
[assembly: LoadableClass(typeof(DiscreteValueGenerator), typeof(DiscreteParamArguments), typeof(SignatureSweeperParameter),
    "Discrete parameter", "dp")]

namespace Microsoft.ML.Runtime.Sweeper
{
    public delegate void SignatureSweeperParameter();

    public abstract class BaseParamArguments
    {
        [Argument(ArgumentType.Required, HelpText = "Parameter name", ShortName = "n")]
        public string Name;
    }

    public abstract class NumericParamArguments : BaseParamArguments
    {
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of steps for grid runthrough.", ShortName = "steps")]
        public int NumSteps = 100;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Amount of increment between steps (multiplicative if log).", ShortName = "inc")]
        public Double? StepSize = null;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Log scale.", ShortName = "log")]
        public bool LogBase = false;
    }

    public class FloatParamArguments : NumericParamArguments
    {
        [Argument(ArgumentType.Required, HelpText = "Minimum value")]
        public Float Min;

        [Argument(ArgumentType.Required, HelpText = "Maximum value")]
        public Float Max;
    }

    public class LongParamArguments : NumericParamArguments
    {
        [Argument(ArgumentType.Required, HelpText = "Minimum value")]
        public long Min;

        [Argument(ArgumentType.Required, HelpText = "Maximum value")]
        public long Max;
    }

    public class DiscreteParamArguments : BaseParamArguments
    {
        [Argument(ArgumentType.Multiple, HelpText = "Values", ShortName = "v")]
        public string[] Values = null;
    }

    public sealed class LongParameterValue : IParameterValue<long>
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
            _valueText = _value.ToString("D");
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

    public sealed class FloatParameterValue : IParameterValue<Float>
    {
        private readonly string _name;
        private readonly string _valueText;
        private readonly Float _value;

        public string Name
        {
            get { return _name; }
        }

        public string ValueText
        {
            get { return _valueText; }
        }

        public Float Value
        {
            get { return _value; }
        }

        public FloatParameterValue(string name, Float value)
        {
            Contracts.Check(!Float.IsNaN(value));
            _name = name;
            _value = value;
            _valueText = _value.ToString("R");
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

    public sealed class StringParameterValue : IParameterValue<string>
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

    public interface INumericValueGenerator : IValueGenerator
    {
        Float NormalizeValue(IParameterValue value);
        bool InRange(IParameterValue value);
    }

    /// <summary>
    /// The integer type parameter sweep.
    /// </summary>
    public class LongValueGenerator : INumericValueGenerator
    {
        private readonly LongParamArguments _args;
        private IParameterValue[] _gridValues;

        public string Name { get { return _args.Name; } }

        public LongValueGenerator(LongParamArguments args)
        {
            Contracts.Check(args.Min < args.Max, "min must be less than max");
            // REVIEW: this condition can be relaxed if we change the math below to deal with it
            Contracts.Check(!args.LogBase || args.Min > 0, "min must be positive if log scale is used");
            Contracts.Check(!args.LogBase || args.StepSize == null || args.StepSize > 1, "StepSize must be greater than 1 if log scale is used");
            Contracts.Check(args.LogBase || args.StepSize == null || args.StepSize > 0, "StepSize must be greater than 0 if linear scale is used");
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

        public Float NormalizeValue(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            Contracts.Check(valueTyped != null, "LongValueGenerator could not normalized parameter because it is not of the correct type");
            Contracts.Check(_args.Min <= valueTyped.Value && valueTyped.Value <= _args.Max, "Value not in correct range");

            if (_args.LogBase)
            {
                Float logBase = (Float)(_args.StepSize ?? Math.Pow(1.0 * _args.Max / _args.Min, 1.0 / (_args.NumSteps - 1)));
                return (Float)((Math.Log(valueTyped.Value, logBase) - Math.Log(_args.Min, logBase)) / (Math.Log(_args.Max, logBase) - Math.Log(_args.Min, logBase)));
            }
            else
                return (Float)(valueTyped.Value - _args.Min) / (_args.Max - _args.Min);
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            Contracts.Check(valueTyped != null, "Parameter should be of type LongParameterValue");
            return (_args.Min <= valueTyped.Value && valueTyped.Value <= _args.Max);
        }

        public string ToStringParameter(IHostEnvironment env)
        {
            return $" p=lp{{{CmdParser.GetSettings(env, _args, new LongParamArguments())}}}";
        }
    }

    /// <summary>
    /// The floating point type parameter sweep.
    /// </summary>
    public class FloatValueGenerator : INumericValueGenerator
    {
        private readonly FloatParamArguments _args;
        private IParameterValue[] _gridValues;

        public string Name { get { return _args.Name; } }

        public FloatValueGenerator(FloatParamArguments args)
        {
            Contracts.Check(args.Min < args.Max, "min must be less than max");
            // REVIEW: this condition can be relaxed if we change the math below to deal with it
            Contracts.Check(!args.LogBase || args.Min > 0, "min must be positive if log scale is used");
            Contracts.Check(!args.LogBase || args.StepSize == null || args.StepSize > 1, "StepSize must be greater than 1 if log scale is used");
            Contracts.Check(args.LogBase || args.StepSize == null || args.StepSize > 0, "StepSize must be greater than 0 if linear scale is used");
            _args = args;
        }

        // REVIEW: Is Float accurate enough?
        public IParameterValue CreateFromNormalized(Double normalizedValue)
        {
            Float val;
            if (_args.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = !_args.StepSize.HasValue
                    ? Math.Pow(1.0 * _args.Max / _args.Min, 1.0 / (_args.NumSteps - 1))
                    : _args.StepSize.Value;
                var logMax = Math.Log(_args.Max, logBase);
                var logMin = Math.Log(_args.Min, logBase);
                val = (Float)(_args.Min * Math.Pow(logBase, normalizedValue * (logMax - logMin)));
            }
            else
                val = (Float)(_args.Min + normalizedValue * (_args.Max - _args.Min));

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

                Float prevValue = Float.NegativeInfinity;
                var maxPlusEpsilon = _args.Max * Math.Sqrt(logBase);
                for (Double value = _args.Min; value <= maxPlusEpsilon; value *= logBase)
                {
                    var floatValue = (Float)value;
                    if (floatValue > prevValue)
                        result.Add(new FloatParameterValue(_args.Name, floatValue));
                    prevValue = floatValue;
                }
            }
            else
            {
                var stepSize = _args.StepSize ?? (Double)(_args.Max - _args.Min) / (_args.NumSteps - 1);
                Float prevValue = Float.NegativeInfinity;
                var maxPlusEpsilon = _args.Max + stepSize / 2;
                for (Double value = _args.Min; value <= maxPlusEpsilon; value += stepSize)
                {
                    var floatValue = (Float)value;
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

        public Float NormalizeValue(IParameterValue value)
        {
            var valueTyped = value as FloatParameterValue;
            Contracts.Check(valueTyped != null, "FloatValueGenerator could not normalized parameter because it is not of the correct type");
            Contracts.Check(_args.Min <= valueTyped.Value && valueTyped.Value <= _args.Max, "Value not in correct range");

            if (_args.LogBase)
            {
                Float logBase = (Float)(_args.StepSize ?? Math.Pow(1.0 * _args.Max / _args.Min, 1.0 / (_args.NumSteps - 1)));
                return (Float)((Math.Log(valueTyped.Value, logBase) - Math.Log(_args.Min, logBase)) / (Math.Log(_args.Max, logBase) - Math.Log(_args.Min, logBase)));
            }
            else
                return (valueTyped.Value - _args.Min) / (_args.Max - _args.Min);
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as FloatParameterValue;
            Contracts.Check(valueTyped != null, "Parameter should be of type FloatParameterValue");
            return (_args.Min <= valueTyped.Value && valueTyped.Value <= _args.Max);
        }

        public string ToStringParameter(IHostEnvironment env)
        {
            return $" p=fp{{{CmdParser.GetSettings(env, _args, new FloatParamArguments())}}}";
        }
    }

    /// <summary>
    /// The discrete parameter sweep.
    /// </summary>
    public class DiscreteValueGenerator : IValueGenerator
    {
        private readonly DiscreteParamArguments _args;

        public string Name { get { return _args.Name; } }

        public DiscreteValueGenerator(DiscreteParamArguments args)
        {
            Contracts.Check(args.Values.Length > 0);
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

        public string ToStringParameter(IHostEnvironment env)
        {
            return $" p=dp{{{CmdParser.GetSettings(env, _args, new DiscreteParamArguments())}}}";
        }
    }

    public sealed class SuggestedSweepsParser
    {
        /// <summary>
        /// Generic parameter parser. Currently hand-hacked to auto-detect type.
        ///
        /// Generic form:   Name:Values
        /// for example,    lr:0.05-0.4
        ///          lambda:0.1-1000@log10
        ///          nl:2-64@log2
        ///          norm:-,+
        /// </summary>
        /// REVIEW: allow overriding auto-detection to specify type
        /// and delegate to parameter type for actual parsing
        /// REVIEW: specifying ordinal discrete parameters
        public bool TryParseParameter(string paramValue, Type paramType, string paramName, out IValueGenerator sweepValues, out string error)
        {
            sweepValues = null;
            error = null;

            if (paramValue.Contains(','))
            {
                var generatorArgs = new DiscreteParamArguments();
                generatorArgs.Name = paramName;
                generatorArgs.Values = paramValue.Split(',');
                sweepValues = new DiscreteValueGenerator(generatorArgs);
                return true;
            }

            // numeric parameter
            if (!CmdParser.IsNumericType(paramType))
                return false;

            // REVIEW:  deal with negative bounds
            string scaleStr = null;
            int atIdx = paramValue.IndexOf('@');
            if (atIdx < 0)
                atIdx = paramValue.IndexOf(';');
            if (atIdx >= 0)
            {
                scaleStr = paramValue.Substring(atIdx + 1);
                paramValue = paramValue.Substring(0, atIdx);
                if (scaleStr.Length < 1)
                {
                    error = $"Could not parse sweep range for parameter: {paramName}";
                    return false;
                }
            }

            // Extract the minimum, and the maximum value of the list of suggested sweeps.
            // Positive lookahead splitting at the '-' character.
            // It is used for the Float and Long param types.
            // Example format: "0.02-0.1;steps:5".
            string[] minMaxRegex = Regex.Split(paramValue, "(?<=[^eE])-");
            if (minMaxRegex.Length != 2)
            {
                if (minMaxRegex.Length > 2)
                    error = $"Could not parse sweep range for parameter: {paramName}";

                return false;
            }
            string minStr = minMaxRegex[0];
            string maxStr = minMaxRegex[1];

            int numSteps = 100;
            Double stepSize = -1;
            bool logBase = false;
            if (scaleStr != null)
            {
                try
                {
                    string[] options = scaleStr.Split(';');
                    bool[] optionsSpecified = new bool[3];
                    foreach (string option in options)
                    {
                        if (option.StartsWith("log") && !option.StartsWith("log-") && !option.StartsWith("log:-"))
                        {
                            logBase = true;
                            optionsSpecified[0] = true;
                        }
                        if (option.StartsWith("steps"))
                        {
                            numSteps = int.Parse(option.Substring(option.IndexOf(':') + 1));
                            optionsSpecified[1] = true;
                        }
                        if (option.StartsWith("inc"))
                        {
                            stepSize = Double.Parse(option.Substring(option.IndexOf(':') + 1), CultureInfo.InvariantCulture);
                            optionsSpecified[2] = true;
                        }
                    }
                    if (options.Length != optionsSpecified.Count(b => b))
                    {
                        error = $"Could not parse sweep range for parameter: {paramName}";
                        return false;
                    }
                }
                catch (Exception e)
                {
                    error = $"Error creating sweep generator for parameter '{paramName}': {e.Message}";
                    return false;
                }
            }

            if (paramType == typeof(UInt16)
                || paramType == typeof(UInt32)
                || paramType == typeof(UInt64)
                || paramType == typeof(short)
                || paramType == typeof(int)
                || paramType == typeof(long))
            {
                long min;
                long max;
                if (!long.TryParse(minStr, out min) || !long.TryParse(maxStr, out max))
                    return false;
                var generatorArgs = new Microsoft.ML.Runtime.Sweeper.LongParamArguments();
                generatorArgs.Name = paramName;
                generatorArgs.Min = min;
                generatorArgs.Max = max;
                generatorArgs.NumSteps = numSteps;
                generatorArgs.StepSize = (stepSize > 0 ? stepSize : new Nullable<Double>());
                generatorArgs.LogBase = logBase;

                try
                {
                    sweepValues = new LongValueGenerator(generatorArgs);
                }
                catch (Exception e)
                {
                    error = $"Error creating sweep generator for parameter '{paramName}': {e.Message}";
                    return false;
                }
            }
            else
            {
                Float minF;
                Float maxF;
                if (!Float.TryParse(minStr, out minF) || !Float.TryParse(maxStr, out maxF))
                    return false;
                var floatArgs = new FloatParamArguments();
                floatArgs.Name = paramName;
                floatArgs.Min = minF;
                floatArgs.Max = maxF;
                floatArgs.NumSteps = numSteps;
                floatArgs.StepSize = (stepSize > 0 ? stepSize : new Nullable<Double>());
                floatArgs.LogBase = logBase;

                try
                {
                    sweepValues = new FloatValueGenerator(floatArgs);
                }
                catch (Exception e)
                {
                    error = $"Error creating sweep generator for parameter '{paramName}': {e.Message}";
                    return false;
                }
            }
            return true;
        }
    }
}
