﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.SearchSpace.Converter;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.SearchSpace
{
    /// <summary>
    /// This class is used to represent a set of <see cref="OptionBase"/>, which can be either one of <see cref="ChoiceOption"/>, <see cref="UniformNumericOption"/> or another nested search space.
    /// </summary>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    /// [!code-csharp[AutoMLExperiment](~/../docs/samples/docs/samples/Microsoft.ML.AutoML.Samples/Sweepable/SearchSpaceExample.cs)]
    /// ]]>
    /// </format>
    /// </example>
    [JsonConverter(typeof(SearchSpaceConverter))]
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

        /// <inheritdoc/>
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

        /// <inheritdoc/>
        public override int FeatureSpaceDim
        {
            get
            {
                return _options.Values.Select(x => x.FeatureSpaceDim).Sum();
            }
        }

        /// <inheritdoc/>
        public ICollection<string> Keys => ((IDictionary<string, OptionBase>)_options).Keys;

        /// <inheritdoc/>
        public ICollection<OptionBase> Values => ((IDictionary<string, OptionBase>)_options).Values;

        /// <inheritdoc/>
        public int Count => ((ICollection<KeyValuePair<string, OptionBase>>)_options).Count;

        /// <inheritdoc/>
        public bool IsReadOnly => ((ICollection<KeyValuePair<string, OptionBase>>)_options).IsReadOnly;

        /// <inheritdoc/>
        public override int?[] Step
        {
            get
            {
                return this.OrderBy(kv => kv.Key)
                           .SelectMany(kv => kv.Value.Step)
                           .ToArray();
            }
        }

        /// <inheritdoc/>
        public OptionBase this[string key] { get => ((IDictionary<string, OptionBase>)_options)[key]; set => ((IDictionary<string, OptionBase>)_options)[key] = value; }

        /// <inheritdoc/>
        public override Parameter SampleFromFeatureSpace(double[] feature)
        {
            Contract.Assert(feature.Length == FeatureSpaceDim, "input feature doesn't match");
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

        /// <inheritdoc/>
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

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            // hash code is calculated in the following process
            // 1. sample parameter from search space with all feature value equals to 0.5
            // 2. serialize sampled parameter to json string
            // 3. return json string hash code

            var featureSpace = Enumerable.Repeat(0.5, FeatureSpaceDim).ToArray();
            var parameter = SampleFromFeatureSpace(featureSpace);
            var json = JsonSerializer.Serialize(parameter);

            // we need to make sure the hash code is the same not only during the same training session, but also
            // on different platform/CLR, so we can't use string.GetHashCode() here.
            uint hash = 31;
            foreach (var c in json)
            {
                hash = ((hash << 5) + hash) ^ c;
            }

            // make sure hash code is greater than 0

            return (int)(hash >> 1);
        }

        private Dictionary<string, OptionBase> GetOptionsFromType(Type typeInfo)
        {
            var fieldOptions = GetOptionsFromField(typeInfo);
            var propertyOptions = GetOptionsFromProperty(typeInfo);
            return fieldOptions.Concat(propertyOptions).ToDictionary(kv => kv.Key, kv => kv.Value);
        }


        private SearchSpace GetSearchSpaceOptionFromType(Type typeInfo)
        {
            var propertyOptions = GetOptionsFromProperty(typeInfo);
            var fieldOptions = GetOptionsFromField(typeInfo);
            var nestOption = new SearchSpace();
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
                Contract.Assert(attributes.Count() <= 1, $"{field.Name} can only define one of the choice|range|option attribute");
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
                        NestOptionAttribute nest => GetSearchSpaceOptionFromType(field.FieldType),
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
                Contract.Assert(attributes.Count() <= 1, $"{property.Name} can only define one of the choice|range|option attribute");
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
                        NestOptionAttribute nest => GetSearchSpaceOptionFromType(property.PropertyType),
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
                Contract.Assert(type == typeof(bool), $"[Option:{optionName}] BooleanChoice can only apply to property or field which type is bool");
                return;
            }

            if (attribute is RangeAttribute range && (range.Option is UniformDoubleOption || range.Option is UniformSingleOption))
            {
                Contract.Assert(type != typeof(int) && type != typeof(short) && type != typeof(long), $"[Option:{optionName}] UniformDoubleOption or UniformSingleOption can't apply to property or field which type is int or short or long");
                return;
            }

            if (attribute is ChoiceAttribute)
            {
                var supportTypes = new Type[] { typeof(string), typeof(int), typeof(short), typeof(long), typeof(float), typeof(double), typeof(char) };
                Contract.Assert(supportTypes.Contains(type) || type.IsEnum, $"[Option:{optionName}] ChoiceAttribute can only apply to enum or the following types {string.Join(",", supportTypes.Select(x => x.Name))}");
                return;
            }
        }

        /// <inheritdoc/>

        public void Add(string key, OptionBase value)
        {
            ((IDictionary<string, OptionBase>)_options).Add(key, value);
        }

        /// <inheritdoc/>
        public bool ContainsKey(string key)
        {
            return ((IDictionary<string, OptionBase>)_options).ContainsKey(key);
        }

        /// <inheritdoc/>
        public bool Remove(string key)
        {
            return ((IDictionary<string, OptionBase>)_options).Remove(key);
        }

        /// <inheritdoc/>
        public bool TryGetValue(string key, out OptionBase value)
        {
            return ((IDictionary<string, OptionBase>)_options).TryGetValue(key, out value);
        }

        /// <inheritdoc/>
        public void Add(KeyValuePair<string, OptionBase> item)
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)_options).Add(item);
        }

        /// <inheritdoc/>
        public void Clear()
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)_options).Clear();
        }

        /// <inheritdoc/>
        public bool Contains(KeyValuePair<string, OptionBase> item)
        {
            return ((ICollection<KeyValuePair<string, OptionBase>>)_options).Contains(item);
        }

        /// <inheritdoc/>
        public void CopyTo(KeyValuePair<string, OptionBase>[] array, int arrayIndex)
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)_options).CopyTo(array, arrayIndex);
        }

        /// <inheritdoc/>
        public bool Remove(KeyValuePair<string, OptionBase> item)
        {
            return ((ICollection<KeyValuePair<string, OptionBase>>)_options).Remove(item);
        }

        /// <inheritdoc/>
        public IEnumerator<KeyValuePair<string, OptionBase>> GetEnumerator()
        {
            return ((IEnumerable<KeyValuePair<string, OptionBase>>)_options).GetEnumerator();
        }

        /// <inheritdoc/>
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

    /// <inheritdoc/>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    /// [!code-csharp[AutoMLExperiment](~/../docs/samples/docs/samples/Microsoft.ML.AutoML.Samples/Sweepable/SearchSpaceExample.cs)]
    /// ]]>
    /// </format>
    /// </example>
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

        /// <inheritdoc/>
        public new T SampleFromFeatureSpace(double[] feature)
        {
            var param = base.SampleFromFeatureSpace(feature);
            var option = param.AsType<T>();

            return option;
        }

        /// <inheritdoc/>
        public double[] MappingToFeatureSpace(T input)
        {
            var param = Parameter.FromObject(input);
            return MappingToFeatureSpace(param);
        }
    }
}
