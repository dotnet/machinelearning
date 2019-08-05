// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Linq;
using System.Text;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Used to indicate suggested sweep ranges for parameter sweeping.
    /// </summary>
    internal abstract class SweepableParam
    {
        public string Name { get; set; }
        private IComparable _rawValue;
        public virtual IComparable RawValue
        {
            get => _rawValue;
            set
            {
                if (!Frozen)
                    _rawValue = value;
            }
        }

        // The raw value will store an index for discrete parameters,
        // but sometimes we want the text or numeric value itself,
        // not the hot index. The processed value does that for discrete
        // params. For other params, it just returns the raw value itself.
        public virtual IComparable ProcessedValue() => _rawValue;

        // Allows for hyperparameter value freezing, so that sweeps
        // will not alter the current value when true.
        public bool Frozen { get; set; }

        // Allows the sweepable param to be set directly using the
        // available ValueText attribute on IParameterValues (from
        // the ParameterSets used in the old hyperparameter sweepers).
        public abstract void SetUsingValueText(string valueText);

        public abstract SweepableParam Clone();
    }

    internal sealed class SweepableDiscreteParam : SweepableParam
    {
        public object[] Options { get; }

        public SweepableDiscreteParam(string name, object[] values, bool isBool = false) : this(values, isBool)
        {
            Name = name;
        }

        public SweepableDiscreteParam(object[] values, bool isBool = false)
        {
            Options = isBool ? new object[] { false, true } : values;
        }

        public override IComparable RawValue
        {
            get => base.RawValue;
            set
            {
                var val = Convert.ToInt32(value);
                if (!Frozen && 0 <= val && val < Options.Length)
                    base.RawValue = val;
            }
        }

        public override void SetUsingValueText(string valueText)
        {
            for (int i = 0; i < Options.Length; i++)
                if (valueText == Options[i].ToString())
                    RawValue = i;
        }

        private static string TranslateOption(object o)
        {
            switch (o)
            {
                case float _:
                case double _:
                    return $"{o}f";
                case long _:
                case int _:
                case byte _:
                case short _:
                    return o.ToString();
                case bool _:
                    return o.ToString().ToLower();
                case Enum _:
                    var type = o.GetType();
                    var defaultName = $"Enums.{type.Name}.{o.ToString()}";
                    var name = type.FullName?.Replace("+", ".");
                    if (name == null)
                        return defaultName;
                    var index1 = name.LastIndexOf(".", StringComparison.Ordinal);
                    var index2 = name.Substring(0, index1).LastIndexOf(".", StringComparison.Ordinal) + 1;
                    if (index2 >= 0)
                        return $"{name.Substring(index2)}.{o.ToString()}";
                    return defaultName;
                default:
                    return $"\"{o}\"";
            }
        }

        public override SweepableParam Clone() =>
            new SweepableDiscreteParam(Name, Options) { RawValue = RawValue, Frozen = Frozen };

        public override string ToString()
        {
            var name = string.IsNullOrEmpty(Name) ? "" : $"\"{Name}\", ";
            return $"[{GetType().Name}({name}new object[]{{{string.Join(", ", Options.Select(TranslateOption))}}})]";
        }

        public override IComparable ProcessedValue() => (IComparable)Options[(int)RawValue];
    }

    internal sealed class SweepableFloatParam : SweepableParam
    {
        public float Min { get; }
        public float Max { get; }
        public float? StepSize { get; }
        public int? NumSteps { get; }
        public bool IsLogScale { get; }

        public SweepableFloatParam(string name, float min, float max, float stepSize = -1, int numSteps = -1,
            bool isLogScale = false) : this(min, max, stepSize, numSteps, isLogScale)
        {
            Name = name;
        }

        public SweepableFloatParam(float min, float max, float stepSize = -1, int numSteps = -1, bool isLogScale = false)
        {
            Min = min;
            Max = max;
            if (!stepSize.Equals(-1))
                StepSize = stepSize;
            if (numSteps != -1)
                NumSteps = numSteps;
            IsLogScale = isLogScale;
        }

        public override void SetUsingValueText(string valueText)
        {
            RawValue = float.Parse(valueText, CultureInfo.InvariantCulture);
        }

        public override SweepableParam Clone() =>
            new SweepableFloatParam(Name, Min, Max, StepSize ?? -1, NumSteps ?? -1, IsLogScale) { RawValue = RawValue, Frozen = Frozen };

        public override string ToString()
        {
            var optional = new StringBuilder();
            if (StepSize != null)
                optional.Append($", stepSize:{StepSize}");
            if (NumSteps != null)
                optional.Append($", numSteps:{NumSteps}");
            if (IsLogScale)
                optional.Append($", isLogScale:true");
            var name = string.IsNullOrEmpty(Name) ? "" : $"\"{Name}\", ";
            return $"[{GetType().Name}({name}{Min}f, {Max}f{optional})]";
        }
    }

    internal sealed class SweepableLongParam : SweepableParam
    {
        public long Min { get; }
        public long Max { get; }
        public float? StepSize { get; }
        public int? NumSteps { get; }
        public bool IsLogScale { get; }

        public SweepableLongParam(string name, long min, long max, float stepSize = -1, int numSteps = -1,
            bool isLogScale = false) : this(min, max, stepSize, numSteps, isLogScale)
        {
            Name = name;
        }

        public SweepableLongParam(long min, long max, float stepSize = -1, int numSteps = -1, bool isLogScale = false)
        {
            Min = min;
            Max = max;
            if (!stepSize.Equals(-1))
                StepSize = stepSize;
            if (numSteps != -1)
                NumSteps = numSteps;
            IsLogScale = isLogScale;
        }

        public override void SetUsingValueText(string valueText)
        {
            RawValue = long.Parse(valueText);
        }

        public override SweepableParam Clone() =>
            new SweepableLongParam(Name, Min, Max, StepSize ?? -1, NumSteps ?? -1, IsLogScale) { RawValue = RawValue, Frozen = Frozen };

        public override string ToString()
        {
            var optional = new StringBuilder();
            if (StepSize != null)
                optional.Append($", stepSize:{StepSize}");
            if (NumSteps != null)
                optional.Append($", numSteps:{NumSteps}");
            if (IsLogScale)
                optional.Append($", isLogScale:true");
            var name = string.IsNullOrEmpty(Name) ? "" : $"\"{Name}\", ";
            return $"[{GetType().Name}({name}{Min}, {Max}{optional})]";
        }
    }
}
