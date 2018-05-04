// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// This class defines attributes to annotate module inputs, outputs, entry points etc. when defining 
    /// the module interface.
    /// </summary>
    public static class TlcModule
    {
        /// <summary>
        /// An attribute used to annotate the component.
        /// </summary>
        [AttributeUsage(AttributeTargets.Class)]
        public sealed class ComponentAttribute : Attribute
        {
            /// <summary>
            /// The load name of the component. Must be unique within its kind.
            /// </summary>
            public string Name { get; set; }

            /// <summary>
            /// UI friendly name. Can contain spaces and other forbidden for Name symbols.
            /// </summary>
            public string FriendlyName { get; set; }

            /// <summary>
            /// Alternative names of the component. Each alias must also be unique in the component's kind.
            /// </summary>
            public string[] Aliases { get; set; }

            /// <summary>
            /// Comma-separated <see cref="Aliases"/>.
            /// </summary>
            public string Alias
            {
                get
                {
                    if (Aliases == null)
                        return null;
                    return string.Join(",", Aliases);
                }
                set
                {
                    if (string.IsNullOrWhiteSpace(value))
                        Aliases = null;
                    else
                    {
                        var parts = value.Split(',');
                        Aliases = parts.Select(x => x.Trim()).ToArray();
                    }
                }
            }

            /// <summary>
            /// Description of the component.
            /// </summary>
            public string Desc { get; set; }

            /// <summary>
            /// This should indicate a name of an embedded resource that contains detailed documents
            /// for the component, e.g., markdown document with the .md extension. The embedded resource
            /// is assumed to be in the same assembly as the class on which this attribute is ascribed.
            /// </summary>
            public string DocName { get; set; }
        }

        /// <summary>
        /// An attribute used to annotate the signature interface.
        /// Effectively, this is a way to associate the signature interface with a user-friendly name.
        /// </summary>
        [AttributeUsage(AttributeTargets.Interface)]
        public sealed class ComponentKindAttribute : Attribute
        {
            public readonly string Kind;

            public ComponentKindAttribute(string kind)
            {
                Kind = kind;
            }
        }

        /// <summary>
        /// An attribute used to annotate the kind of entry points.
        /// Typically it is used on the input classes.
        /// </summary>
        [AttributeUsage(AttributeTargets.Class)]
        public sealed class EntryPointKindAttribute : Attribute
        {
            public readonly Type[] Kinds;

            public EntryPointKindAttribute(params Type[] kinds)
            {
                Kinds = kinds;
            }
        }

        /// <summary>
        /// An attribute used to annotate the outputs of the module.
        /// </summary>
        [AttributeUsage(AttributeTargets.Field)]
        public sealed class OutputAttribute : Attribute
        {
            /// <summary>
            /// Official name of the output. If it is not specified, the field name is used.
            /// </summary>
            public string Name { get; set; }

            /// <summary>
            /// The description of the output.
            /// </summary>
            public string Desc { get; set; }

            /// <summary>
            /// The rank order of the output. Because .NET reflection returns members in an unspecfied order, this 
            /// is the only way to ensure consistency.
            /// </summary>
            public Double SortOrder { get; set; }
        }

        /// <summary>
        /// An attribute to indicate that a field is optional in an EntryPoint module.
        /// A node can be run without optional input fields.
        /// </summary>
        [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
        public sealed class OptionalInputAttribute : Attribute { }

        /// <summary>
        /// An attribute used to annotate the valid range of a numeric input.
        /// </summary>
        [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
        public sealed class RangeAttribute : Attribute
        {
            private object _min;
            private object _max;
            private object _inf;
            private object _sup;
            private Type _type;

            /// <summary>
            /// The target type of this range attribute, as determined by the type of
            /// the set range bound values.
            /// </summary>
            public Type Type => _type;

            /// <summary>
            /// An inclusive lower bound of the value.
            /// </summary>
            public object Min
            {
                get { return _min; }
                set
                {
                    CheckType(value);
                    Contracts.Check(_inf == null,
                        "The minimum and infimum cannot be both set in a range attribute");
                    Contracts.Check(_max == null || ((IComparable)_max).CompareTo(value) != -1,
                        "The minimum must be less than or equal to the maximum");
                    Contracts.Check(_sup == null || ((IComparable)_sup).CompareTo(value) == 1,
                        "The minimum must be less than the supremum");
                    _min = value;
                }
            }

            /// <summary>
            /// An inclusive upper bound of the value.
            /// </summary>
            public object Max
            {
                get { return _max; }
                set
                {
                    CheckType(value);
                    Contracts.Check(_sup == null,
                        "The maximum and supremum cannot be both set in a range attribute");
                    Contracts.Check(_min == null || ((IComparable)_min).CompareTo(value) != 1,
                        "The maximum must be greater than or equal to the minimum");
                    Contracts.Check(_inf == null || ((IComparable)_inf).CompareTo(value) == -1,
                        "The maximum must be greater than the infimum");
                    _max = value;
                }
            }

            /// <summary>
            /// An exclusive lower bound of the value.
            /// </summary>
            public object Inf
            {
                get { return _inf; }
                set
                {
                    CheckType(value);
                    Contracts.Check(_min == null,
                        "The infimum and minimum cannot be both set in a range attribute");
                    Contracts.Check(_max == null || ((IComparable)_max).CompareTo(value) == 1,
                        "The infimum must be less than the maximum");
                    Contracts.Check(_sup == null || ((IComparable)_sup).CompareTo(value) == 1,
                        "The infimum must be less than the supremum");
                    _inf = value;
                }
            }

            /// <summary>
            /// An exclusive upper bound of the value.
            /// </summary>
            public object Sup
            {
                get { return _sup; }
                set
                {
                    CheckType(value);
                    Contracts.Check(_max == null,
                        "The supremum and maximum cannot be both set in a range attribute");
                    Contracts.Check(_min == null || ((IComparable)_min).CompareTo(value) == -1,
                        "The supremum must be greater than the minimum");
                    Contracts.Check(_inf == null || ((IComparable)_inf).CompareTo(value) == -1,
                        "The supremum must be greater than the infimum");
                    _sup = value;
                }
            }

            private void CheckType(object val)
            {
                Contracts.CheckValue(val, nameof(val));
                if (_type == null)
                {
                    Contracts.Check(val is IComparable, "Type for range attribute must support IComparable");
                    _type = val.GetType();
                }
                else
                    Contracts.Check(_type == val.GetType(), "All Range attribute values must be of the same type");
            }

            public void CastToDouble()
            {
                _type = typeof(double);
                if (_inf != null)
                    _inf = Convert.ToDouble(_inf);
                if (_min != null)
                    _min = Convert.ToDouble(_min);
                if (_max != null)
                    _max = Convert.ToDouble(_max);
                if (_sup != null)
                    _sup = Convert.ToDouble(_sup);
            }

            public override string ToString()
            {
                string optionalTypeSpecifier = "";
                if (_type == typeof(double))
                    optionalTypeSpecifier = "d";
                else if (_type == typeof(float))
                    optionalTypeSpecifier = "f";

                var pieces = new List<string>();
                if (_inf != null)
                    pieces.Add($"Inf = {_inf}{optionalTypeSpecifier}");
                if (_min != null)
                    pieces.Add($"Min = {_min}{optionalTypeSpecifier}");
                if (_max != null)
                    pieces.Add($"Max = {_max}{optionalTypeSpecifier}");
                if (_sup != null)
                    pieces.Add($"Sup = {_sup}{optionalTypeSpecifier}");
                return $"[TlcModule.Range({string.Join(", ", pieces)})]";
            }
        }

        /// <summary>
        /// An attribute used to indicate suggested sweep ranges for parameter sweeping.
        /// </summary>
        public abstract class SweepableParamAttribute : Attribute
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

            public abstract SweepableParamAttribute Clone();
        }

        /// <summary>
        /// An attribute used to indicate suggested sweep ranges for discrete parameter sweeping.
        /// The value is the index of the option chosen. Use Options[Value] to get the corresponding
        /// option using the index.
        /// </summary>
        [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
        public sealed class SweepableDiscreteParamAttribute : SweepableParamAttribute
        {
            public object[] Options { get; }

            public SweepableDiscreteParamAttribute(string name, object[] values, bool isBool = false) : this(values, isBool)
            {
                Name = name;
            }

            public SweepableDiscreteParamAttribute(object[] values, bool isBool = false)
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

            public int IndexOf(object option)
            {
                for (int i = 0; i < Options.Length; i++)
                    if (option == Options[i])
                        return i;
                return -1;
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

            public override SweepableParamAttribute Clone() =>
                new SweepableDiscreteParamAttribute(Name, Options) { RawValue = RawValue, Frozen = Frozen };

            public override string ToString()
            {
                var name = string.IsNullOrEmpty(Name) ? "" : $"\"{Name}\", ";
                return $"[TlcModule.{GetType().Name}({name}new object[]{{{string.Join(", ", Options.Select(TranslateOption))}}})]";
            }

            public override IComparable ProcessedValue() => (IComparable)Options[(int)RawValue];
        }

        /// <summary>
        /// An attribute used to indicate suggested sweep ranges for float parameter sweeping.
        /// </summary>
        [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
        public sealed class SweepableFloatParamAttribute : SweepableParamAttribute
        {
            public float Min { get; }
            public float Max { get; }
            public float? StepSize { get; }
            public int? NumSteps { get; }
            public bool IsLogScale { get; }

            public SweepableFloatParamAttribute(string name, float min, float max, float stepSize = -1, int numSteps = -1,
                bool isLogScale = false) : this(min, max, stepSize, numSteps, isLogScale)
            {
                Name = name;
            }

            public SweepableFloatParamAttribute(float min, float max, float stepSize = -1, int numSteps = -1, bool isLogScale = false)
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
                RawValue = float.Parse(valueText);
            }

            public override SweepableParamAttribute Clone() =>
                new SweepableFloatParamAttribute(Name, Min, Max, StepSize ?? -1, NumSteps ?? -1, IsLogScale) { RawValue = RawValue, Frozen = Frozen };

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
                return $"[TlcModule.{GetType().Name}({name}{Min}f, {Max}f{optional})]";
            }
        }

        /// <summary>
        /// An attribute used to indicate suggested sweep ranges for long parameter sweeping.
        /// </summary>
        [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
        public sealed class SweepableLongParamAttribute : SweepableParamAttribute
        {
            public long Min { get; }
            public long Max { get; }
            public float? StepSize { get; }
            public int? NumSteps { get; }
            public bool IsLogScale { get; }

            public SweepableLongParamAttribute(string name, long min, long max, float stepSize = -1, int numSteps = -1,
                bool isLogScale = false) : this(min, max, stepSize, numSteps, isLogScale)
            {
                Name = name;
            }

            public SweepableLongParamAttribute(long min, long max, float stepSize = -1, int numSteps = -1, bool isLogScale = false)
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

            public override SweepableParamAttribute Clone() =>
                new SweepableLongParamAttribute(Name, Min, Max, StepSize ?? -1, NumSteps ?? -1, IsLogScale) { RawValue = RawValue, Frozen = Frozen };

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
                return $"[TlcModule.{GetType().Name}({name}{Min}, {Max}{optional})]";
            }
        }

        /// <summary>
        /// An attribute to mark an entry point of a module.
        /// </summary>
        [AttributeUsage(AttributeTargets.Method)]
        public sealed class EntryPointAttribute : Attribute
        {
            /// <summary>
            /// The entry point name.
            /// </summary>
            public string Name { get; set; }

            /// <summary>
            /// The entry point description.
            /// </summary>
            public string Desc { get; set; }

            /// <summary>
            /// UI friendly name. Can contain spaces and other forbidden for Name symbols.
            /// </summary>
            public string UserName { get; set; }

            /// <summary>
            /// Short name of the Entry Point
            /// </summary>
            public string ShortName { get; set; }
        }

        /// <summary>
        /// The list of data types that are supported as inputs or outputs.
        /// </summary>
        public enum DataKind
        {
            /// <summary>
            /// Not used.
            /// </summary>
            Unknown = 0,
            /// <summary>
            /// Integer, including long. 
            /// </summary>
            Int,
            /// <summary>
            /// Unsigned integer, including ulong. 
            /// </summary>
            UInt,
            /// <summary>
            /// Floating point, including double.
            /// </summary>
            Float,
            /// <summary>
            /// A char.
            /// </summary>
            Char,
            /// <summary>
            /// A string.
            /// </summary>
            String,
            /// <summary>
            /// A boolean value.
            /// </summary>
            Bool,
            /// <summary>
            /// A dataset, represented by an <see cref="IDataView"/>.
            /// </summary>
            DataView,
            /// <summary>
            /// A file handle, represented by an <see cref="IFileHandle"/>.
            /// </summary>
            FileHandle,
            /// <summary>
            /// A transform model, represented by an <see cref="ITransformModel"/>.
            /// </summary>
            TransformModel,
            /// <summary>
            /// A predictor model, represented by an <see cref="IPredictorModel"/>.
            /// </summary>
            PredictorModel,
            /// <summary>
            /// An enum: one value of a specified list.
            /// </summary>
            Enum,
            /// <summary>
            /// An array (0 or more values of the same type, accessible by index). 
            /// </summary>
            Array,
            /// <summary>
            /// A dictionary (0 or more values of the same type, identified by a unique string key). 
            /// The underlying C# representation is <see cref="System.Collections.Generic.Dictionary{TKey, TValue}"/>
            /// </summary>
            Dictionary,
            /// <summary>
            /// A component of a specified kind. The component is identified by the "load name" (unique per kind) and,
            /// optionally, a set of parameters, unique to each component. Example: "BinaryClassifierEvaluator{threshold=0.5}".
            /// The C# representation is <see cref="IComponentFactory"/>.
            /// </summary>
            Component,
            /// <summary>
            /// An C# object that represents state, such as <see cref="IMlState"/>. 
            /// </summary>
            State
        }

        public static DataKind GetDataType(Type type)
        {
            Contracts.AssertValue(type);

            // If this is a Optional-wrapped type, unwrap it and examine
            // the inner type.
            if (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(Optional<>) || type.GetGenericTypeDefinition() == typeof(Nullable<>)))
                type = type.GetGenericArguments()[0];

            if (type == typeof(char))
                return DataKind.Char;
            if (type == typeof(string))
                return DataKind.String;
            if (type == typeof(bool))
                return DataKind.Bool;
            if (type == typeof(int) || type == typeof(long))
                return DataKind.Int;
            if (type == typeof(uint) || type == typeof(ulong))
                return DataKind.UInt;
            if (type == typeof(Single) || type == typeof(Double))
                return DataKind.Float;
            if (typeof(IDataView).IsAssignableFrom(type))
                return DataKind.DataView;
            if (typeof(ITransformModel).IsAssignableFrom(type))
                return DataKind.TransformModel;
            if (typeof(IPredictorModel).IsAssignableFrom(type))
                return DataKind.PredictorModel;
            if (typeof(IFileHandle).IsAssignableFrom(type))
                return DataKind.FileHandle;
            if (type.IsEnum)
                return DataKind.Enum;
            if (type.IsArray)
                return DataKind.Array;
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Dictionary<,>)
                && type.GetGenericArguments()[0] == typeof(string))
            {
                return DataKind.Dictionary;
            }
            if (typeof(IComponentFactory).IsAssignableFrom(type))
                return DataKind.Component;
            if (typeof(IMlState).IsAssignableFrom(type))
                return DataKind.State;

            return DataKind.Unknown;
        }

        public static bool IsNumericKind(DataKind kind)
        {
            return kind == DataKind.Int || kind == DataKind.UInt || kind == DataKind.Float;
        }
    }

    /// <summary>
    /// The untyped base class for 'maybe'.
    /// </summary>
    public abstract class Optional
    {
        /// <summary>
        /// Whether the value was set 'explicitly', or 'implicitly'.
        /// </summary>
        public readonly bool IsExplicit;

        public abstract object GetValue();

        protected Optional(bool isExplicit)
        {
            IsExplicit = isExplicit;
        }
    }

    /// <summary>
    /// This is a 'maybe' class that is able to differentiate the cases when the value is set 'explicitly', or 'implicitly'.
    /// The idea is that if the default value is specified by the user, in some cases it needs to be treated differently
    /// than if it's auto-filled.
    /// 
    /// An example is the weight column: the default behavior is to use 'Weight' column if it's present. But if the user explicitly sets 
    /// the weight column to be 'Weight', we need to actually enforce the presence of the column.
    /// </summary>
    /// <typeparam name="T">The type of the value</typeparam>
    public sealed class Optional<T> : Optional
    {
        public readonly T Value;

        private Optional(bool isExplicit, T value)
            : base(isExplicit)
        {
            Value = value;
        }

        /// <summary>
        /// Create the 'implicit' value.
        /// </summary>
        public static Optional<T> Implicit(T value)
        {
            return new Optional<T>(false, value);
        }

        public static Optional<T> Explicit(T value)
        {
            return new Optional<T>(true, value);
        }

        /// <summary>
        /// The implicit conversion into <typeparamref name="T"/>.
        /// </summary>
        public static implicit operator T(Optional<T> optional)
        {
            return optional.Value;
        }

        /// <summary>
        /// The implicit conversion from <typeparamref name="T"/>. 
        /// This will assume that the parameter is set 'explicitly'.
        /// </summary>
        public static implicit operator Optional<T>(T value)
        {
            return new Optional<T>(true, value);
        }

        public override object GetValue()
        {
            return Value;
        }

        public override string ToString()
        {
            if (Value == null)
                return "";
            return Value.ToString();
        }
    }
}
