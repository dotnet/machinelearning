// <copyright file="SearchSpace.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.ModelBuilder.SearchSpace.Option;
using Newtonsoft.Json;

namespace Microsoft.ML.ModelBuilder.SearchSpace
{
    public class SearchSpace: OptionBase, IDictionary<string, OptionBase>
    {
        private Dictionary<string, OptionBase> options;
        private Parameter defaultOption;

        public SearchSpace(params KeyValuePair<string, OptionBase>[] options)
            : this()
        {
            this.options = options.ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        public SearchSpace(IEnumerable<KeyValuePair<string, OptionBase>> options)
            : this()
        {
            this.options = options.ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        public SearchSpace()
        {
            this.options = new Dictionary<string, OptionBase>();
        }

        public override double[] Default
        {
            get
            {
                return this.OrderBy(kv => kv.Key)
                           .SelectMany(kv => kv.Value.Default)
                           .ToArray();
            }
        }

        protected SearchSpace(Type typeInfo, Parameter defaultOption = null)
            : this()
        {
            this.options = this.GetOptionsFromType(typeInfo);
            var nestedSS = this.GetNestedSearchSpaceFromType(typeInfo);
            foreach (var ss in nestedSS)
            {
                this.options.Add(ss.Key, ss.Value);
            }

            this.defaultOption = defaultOption;
        }

        public override int FeatureSpaceDim
        {
            get
            {
                return this.options.Values.Select(x => x.FeatureSpaceDim).Sum();
            }
        }

        public ICollection<string> Keys => ((IDictionary<string, OptionBase>)options).Keys;

        public ICollection<OptionBase> Values => ((IDictionary<string, OptionBase>)options).Values;

        public int Count => ((ICollection<KeyValuePair<string, OptionBase>>)options).Count;

        public bool IsReadOnly => ((ICollection<KeyValuePair<string, OptionBase>>)options).IsReadOnly;

        public override int?[] Step
        {
            get
            {
                return this.OrderBy(kv => kv.Key)
                           .SelectMany(kv => kv.Value.Step)
                           .ToArray();
            }
        }

        public OptionBase this[string key] { get => ((IDictionary<string, OptionBase>)options)[key]; set => ((IDictionary<string, OptionBase>)options)[key] = value; }

        public override IParameter SampleFromFeatureSpace(double[] feature)
        {
            Contract.Requires(feature.Length == this.FeatureSpaceDim, "input feature doesn't match");
            var param = Parameter.CreateNestedParameter();
            var cur = 0;

            foreach (var key in this.options.Keys.OrderBy(k => k))
            {
                var option = this.options[key];
                var input = feature.Skip(cur).Take(option.FeatureSpaceDim).ToArray();
                var value = option.SampleFromFeatureSpace(input);
                param[key] = value;
                cur += option.FeatureSpaceDim;
            }

            if (this.defaultOption != null)
            {
                return this.Update(this.defaultOption, param);
            }

            return param;
        }

        public override double[] MappingToFeatureSpace(IParameter parameter)
        {
            var res = new List<double>();
            foreach (var key in this.options.Keys.OrderBy(k => k))
            {
                var option = this.options[key];
                var input = parameter[key];
                var value = option.MappingToFeatureSpace(input);
                res.AddRange(value);
            }

            return res.ToArray();
        }

        private Dictionary<string, OptionBase> GetOptionsFromType(Type typeInfo)
        {
            var fieldOptions = this.GetOptionsFromField(typeInfo);
            var propertyOptions = this.GetOptionsFromProperty(typeInfo);
            return fieldOptions.Concat(propertyOptions).ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        private Dictionary<string, SearchSpace> GetNestedSearchSpaceFromType(Type typeInfo)
        {
            var fieldSS = this.GetSearchSpacesFromField(typeInfo);
            var propertySS = this.GetSearchSpacesFromProperty(typeInfo);
            return fieldSS.Concat(propertySS).ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        private Dictionary<string, SearchSpace> GetSearchSpacesFromField(Type typeInfo)
        {
            var fieldInfos = typeInfo.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            var res = new Dictionary<string, SearchSpace>();

            foreach (var field in fieldInfos)
            {
                var optionAttribute = field.GetCustomAttributes(typeof(OptionAttribute), false);
                if (optionAttribute.Count() == 0)
                {
                    continue;
                }
                else
                {
                    var ss = new SearchSpace(field.FieldType);
                    res.Add(field.Name, ss);
                }
            }

            return res;
        }

        private Dictionary<string, SearchSpace> GetSearchSpacesFromProperty(Type typeInfo)
        {
            var propertyInfos = typeInfo.GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            var res = new Dictionary<string, SearchSpace>();

            foreach (var property in propertyInfos)
            {
                var optionAttribute = property.GetCustomAttributes(typeof(OptionAttribute), false);
                if (optionAttribute.Count() == 0)
                {
                    continue;
                }
                else
                {
                    var ss = new SearchSpace(property.PropertyType);
                    res.Add(property.Name, ss);
                }
            }

            return res;
        }

        private Dictionary<string, OptionBase> GetOptionsFromField(Type typeInfo)
        {
            var fieldInfos = typeInfo.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            var res = new Dictionary<string, OptionBase>();

            foreach (var field in fieldInfos)
            {
                var choiceAttributes = field.GetCustomAttributes(typeof(ChoiceAttribute), false);
                var rangeAttributes = field.GetCustomAttributes(typeof(RangeAttribute), false);

                var attributes = choiceAttributes.Concat(rangeAttributes);
                Contract.Requires(attributes.Count() <= 1, $"{field.Name} can only define one of the choice|range|option attribute");
                if (attributes.Count() == 0)
                {
                    continue;
                }
                else
                {
                    OptionBase option = attributes.First() switch
                    {
                        ChoiceAttribute choice => choice.Option,
                        RangeAttribute range => range.Option,
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

                var attributes = choiceAttributes.Concat(rangeAttributes);
                Contract.Requires(attributes.Count() <= 1, $"{property.Name} can only define one of the choice|range|option attribute");
                if (attributes.Count() == 0)
                {
                    continue;
                }
                else
                {
                    OptionBase option = attributes.First() switch
                    {
                        ChoiceAttribute choice => choice.Option,
                        RangeAttribute range => range.Option,
                        _ => throw new NotImplementedException(),
                    };

                    res.Add(property.Name, option);
                }
            }

            return res;
        }

        public void Add(string key, OptionBase value)
        {
            ((IDictionary<string, OptionBase>)options).Add(key, value);
        }

        public bool ContainsKey(string key)
        {
            return ((IDictionary<string, OptionBase>)options).ContainsKey(key);
        }

        public bool Remove(string key)
        {
            return ((IDictionary<string, OptionBase>)options).Remove(key);
        }

        public bool TryGetValue(string key, out OptionBase value)
        {
            return ((IDictionary<string, OptionBase>)options).TryGetValue(key, out value);
        }

        public void Add(KeyValuePair<string, OptionBase> item)
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)options).Add(item);
        }

        public void Clear()
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)options).Clear();
        }

        public bool Contains(KeyValuePair<string, OptionBase> item)
        {
            return ((ICollection<KeyValuePair<string, OptionBase>>)options).Contains(item);
        }

        public void CopyTo(KeyValuePair<string, OptionBase>[] array, int arrayIndex)
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)options).CopyTo(array, arrayIndex);
        }

        public bool Remove(KeyValuePair<string, OptionBase> item)
        {
            return ((ICollection<KeyValuePair<string, OptionBase>>)options).Remove(item);
        }

        public IEnumerator<KeyValuePair<string, OptionBase>> GetEnumerator()
        {
            return ((IEnumerable<KeyValuePair<string, OptionBase>>)options).GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return ((System.Collections.IEnumerable)options).GetEnumerator();
        }

        private IParameter Update(IParameter left, IParameter right)
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
                res[kv.Key] = this.Update(left.ContainsKey(kv.Key) ? left[kv.Key] : null, right.ContainsKey(kv.Key) ? right[kv.Key] : null);
            }

            return res;
        }
    }
}
