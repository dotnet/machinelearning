// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.SearchSpace
{
    /// <summary>
    /// This class is used to represent a set of <see cref="OptionBase"/>, which can be either one of <see cref="ChoiceOption"/>, <see cref="UniformNumericOption"/> or <see cref="NestOption"/>.
    /// </summary>
    public class SearchSpace : OptionBase, IDictionary<string, OptionBase>
    {
        private readonly Dictionary<string, OptionBase> _options;
        private readonly Parameter _defaultOption;

        /// <summary>
        /// Create <see cref="SearchSpace"/> using a group of <see cref="OptionBase"/>.
        /// </summary>
        /// <param name="options"></param>
        internal SearchSpace(params KeyValuePair<string, OptionBase>[] options)
            : this()
        {
            _options = options.ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        internal SearchSpace(IEnumerable<KeyValuePair<string, OptionBase>> options)
            : this()
        {
            _options = options.ToDictionary(kv => kv.Key, kv => kv.Value);
        }
        /// <inheritdoc/>

        public SearchSpace()
        {
            _options = new Dictionary<string, OptionBase>();
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override double[] Default
        {
            get
            {
                return this.OrderBy(kv => kv.Key)
                           .SelectMany(kv => kv.Value.Default)
                           .ToArray();
            }
        }
        /// <inheritdoc/>

        protected SearchSpace(Type typeInfo, Parameter defaultOption = null)
            : this()
        {
            _options = GetOptionsFromType(typeInfo);
            _defaultOption = defaultOption;
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override int FeatureSpaceDim
        {
            get
            {
                return _options.Values.Select(x => x.FeatureSpaceDim).Sum();
            }
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public ICollection<string> Keys => ((IDictionary<string, OptionBase>)_options).Keys;

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public ICollection<OptionBase> Values => ((IDictionary<string, OptionBase>)_options).Values;

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public int Count => ((ICollection<KeyValuePair<string, OptionBase>>)_options).Count;

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool IsReadOnly => ((ICollection<KeyValuePair<string, OptionBase>>)_options).IsReadOnly;

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override int?[] Step
        {
            get
            {
                return this.OrderBy(kv => kv.Key)
                           .SelectMany(kv => kv.Value.Step)
                           .ToArray();
            }
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public OptionBase this[string key] { get => ((IDictionary<string, OptionBase>)_options)[key]; set => ((IDictionary<string, OptionBase>)_options)[key] = value; }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override Parameter SampleFromFeatureSpace(double[] feature)
        {
            Contracts.Check(feature.Length == FeatureSpaceDim, "input feature doesn't match");
            var param = Parameter.CreateNestedParameter();
            var cur = 0;

            foreach (var key in _options.Keys.OrderBy(k => k))
            {
                var option = _options[key];
                var input = feature.Skip(cur).Take(option.FeatureSpaceDim).ToArray();
                var value = option.SampleFromFeatureSpace(input);
                param[key] = value;
                cur += option.FeatureSpaceDim;
            }

            if (_defaultOption != null)
            {
                return Update(_defaultOption, param);
            }

            return param;
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override double[] MappingToFeatureSpace(Parameter parameter)
        {
            var res = new List<double>();
            foreach (var key in _options.Keys.OrderBy(k => k))
            {
                var option = _options[key];
                var input = parameter[key];
                var value = option.MappingToFeatureSpace(input);
                res.AddRange(value);
            }

            return res.ToArray();
        }

        private Dictionary<string, OptionBase> GetOptionsFromType(Type typeInfo)
        {
            var fieldOptions = GetOptionsFromField(typeInfo);
            var propertyOptions = GetOptionsFromProperty(typeInfo);
            return fieldOptions.Concat(propertyOptions).ToDictionary(kv => kv.Key, kv => kv.Value);
        }


        private NestOption GetNestOptionFromType(Type typeInfo)
        {
            var propertyOptions = GetOptionsFromProperty(typeInfo);
            var fieldOptions = GetOptionsFromField(typeInfo);
            var nestOption = new NestOption();
            foreach (var kv in propertyOptions.Concat(fieldOptions))
            {
                nestOption[kv.Key] = kv.Value;
            }

            return nestOption;
        }

        private Dictionary<string, OptionBase> GetOptionsFromField(Type typeInfo)
        {
            var fieldInfos = typeInfo.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            var res = new Dictionary<string, OptionBase>();

            foreach (var field in fieldInfos)
            {
                var choiceAttributes = field.GetCustomAttributes(typeof(ChoiceAttribute), false);
                var rangeAttributes = field.GetCustomAttributes(typeof(RangeAttribute), false);
                var booleanChoiceAttributes = field.GetCustomAttributes(typeof(BooleanChoiceAttribute), false);
                var nestOptionAttributes = field.GetCustomAttributes(typeof(NestOptionAttribute), false);

                var attributes = choiceAttributes.Concat(rangeAttributes).Concat(booleanChoiceAttributes).Concat(nestOptionAttributes);
                Contracts.Check(attributes.Count() <= 1, $"{field.Name} can only define one of the choice|range|option attribute");
                if (attributes.Count() == 0)
                {
                    continue;
                }
                else
                {
                    CheckOptionType(attributes.First(), field.Name, field.FieldType);

                    OptionBase option = attributes.First() switch
                    {
                        ChoiceAttribute choice => choice.Option,
                        RangeAttribute range => range.Option,
                        BooleanChoiceAttribute booleanChoice => booleanChoice.Option,
                        NestOptionAttribute nest => GetNestOptionFromType(field.FieldType),
                        _ => throw new NotImplementedException(),
                    };

                    res.Add(field.Name, option);
                }
            }

            return res;
        }

        private Dictionary<string, OptionBase> GetOptionsFromProperty(Type typeInfo)
        {
            var propertyInfo = typeInfo.GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            var res = new Dictionary<string, OptionBase>();

            foreach (var property in propertyInfo)
            {
                var choiceAttributes = property.GetCustomAttributes(typeof(ChoiceAttribute), false);
                var rangeAttributes = property.GetCustomAttributes(typeof(RangeAttribute), false);
                var booleanChoiceAttributes = property.GetCustomAttributes(typeof(BooleanChoiceAttribute), false);
                var nestOptionAttributes = property.GetCustomAttributes(typeof(NestOptionAttribute), false);

                var attributes = choiceAttributes.Concat(rangeAttributes).Concat(booleanChoiceAttributes).Concat(nestOptionAttributes);
                Contracts.Check(attributes.Count() <= 1, $"{property.Name} can only define one of the choice|range|option attribute");
                if (attributes.Count() == 0)
                {
                    continue;
                }
                else
                {
                    CheckOptionType(attributes.First(), property.Name, property.PropertyType);

                    OptionBase option = attributes.First() switch
                    {
                        ChoiceAttribute choice => choice.Option,
                        RangeAttribute range => range.Option,
                        BooleanChoiceAttribute booleanChoice => booleanChoice.Option,
                        NestOptionAttribute nest => GetNestOptionFromType(property.PropertyType),
                        _ => throw new NotImplementedException(),
                    };

                    res.Add(property.Name, option);
                }
            }

            return res;
        }

        private void CheckOptionType(object attribute, string optionName, Type type)
        {
            if (attribute is BooleanChoiceAttribute)
            {
                Contracts.Assert(type == typeof(bool), $"[Option:{optionName}] BooleanChoice can only apply to property or field which type is bool");
                return;
            }

            if (attribute is RangeAttribute range && (range.Option is UniformDoubleOption || range.Option is UniformSingleOption))
            {
                Contracts.Assert(type != typeof(int) && type != typeof(short) && type != typeof(long), $"[Option:{optionName}] UniformDoubleOption or UniformSingleOption can't apply to property or field which type is int or short or long");
                return;
            }

            if (attribute is ChoiceAttribute)
            {
                var supportTypes = new Type[] { typeof(string), typeof(int), typeof(short), typeof(long), typeof(float), typeof(double), typeof(char) };
                Contracts.Assert(supportTypes.Contains(type) || type.IsEnum, $"[Option:{optionName}] ChoiceAttribute can only apply to enum or the following types {string.Join(",", supportTypes.Select(x => x.Name))}");
                return;
            }
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>

        public void Add(string key, OptionBase value)
        {
            ((IDictionary<string, OptionBase>)_options).Add(key, value);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool ContainsKey(string key)
        {
            return ((IDictionary<string, OptionBase>)_options).ContainsKey(key);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool Remove(string key)
        {
            return ((IDictionary<string, OptionBase>)_options).Remove(key);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool TryGetValue(string key, out OptionBase value)
        {
            return ((IDictionary<string, OptionBase>)_options).TryGetValue(key, out value);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public void Add(KeyValuePair<string, OptionBase> item)
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)_options).Add(item);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public void Clear()
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)_options).Clear();
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool Contains(KeyValuePair<string, OptionBase> item)
        {
            return ((ICollection<KeyValuePair<string, OptionBase>>)_options).Contains(item);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public void CopyTo(KeyValuePair<string, OptionBase>[] array, int arrayIndex)
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)_options).CopyTo(array, arrayIndex);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public bool Remove(KeyValuePair<string, OptionBase> item)
        {
            return ((ICollection<KeyValuePair<string, OptionBase>>)_options).Remove(item);
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public IEnumerator<KeyValuePair<string, OptionBase>> GetEnumerator()
        {
            return ((IEnumerable<KeyValuePair<string, OptionBase>>)_options).GetEnumerator();
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return ((System.Collections.IEnumerable)_options).GetEnumerator();
        }

        private Parameter Update(Parameter left, Parameter right)
        {
            var res = (left?.ParameterType, right?.ParameterType) switch
            {
                (ParameterType.Object, ParameterType.Object) => null,
                (_, null) => left,
                _ => right,
            };

            if (res != null)
            {
                return res;
            }

            res = Parameter.CreateNestedParameter();
            foreach (var kv in left.Concat(right))
            {
                res[kv.Key] = Update(left.ContainsKey(kv.Key) ? left[kv.Key] : null, right.ContainsKey(kv.Key) ? right[kv.Key] : null);
            }

            return res;
        }
    }

    /// <summary>
    /// <inheritdoc/>
    /// </summary>
    public sealed class SearchSpace<T> : SearchSpace
        where T : class, new()
    {
        private readonly T _defaultOption = null;

        /// <summary>
        /// Create <see cref="SearchSpace{T}"/> from <typeparamref name="T"/>. This initializer search for the <see cref="NestOptionAttribute"/> in <typeparamref name="T"/> and create searching space accordingly.
        /// </summary>
        public SearchSpace()
            : base(typeof(T))
        {
        }

        /// <summary>
        /// Create <see cref="SearchSpace{T}"/> from <typeparamref name="T"/> and <paramref name="defaultOption"/>. This initializer search for the <see cref="NestOptionAttribute"/> in <typeparamref name="T"/> and create searching space accordingly.
        /// </summary>
        public SearchSpace(T defaultOption)
            : base(typeof(T), Parameter.FromObject(defaultOption))
        {
            _defaultOption = defaultOption;
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public new T SampleFromFeatureSpace(double[] feature)
        {
            var param = base.SampleFromFeatureSpace(feature);
            var option = param.AsType<T>();

            return option;
        }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public double[] MappingToFeatureSpace(T input)
        {
            var param = Parameter.FromObject(input);
            return MappingToFeatureSpace(param);
        }
    }
}
