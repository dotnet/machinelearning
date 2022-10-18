// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.SearchSpace.Option
{
    /// <summary>
    /// This class represent nest option, which is an option that contains other options, like <see cref="ChoiceOption"/>, <see cref="UniformNumericOption"/> or even <see cref="NestOption"/> itself.
    /// </summary>
    public sealed class NestOption : OptionBase, IDictionary<string, OptionBase>
    {
        private readonly Dictionary<string, OptionBase> _options = new Dictionary<string, OptionBase>();

        /// <inheritdoc/>
        public OptionBase this[string key] { get => ((IDictionary<string, OptionBase>)_options)[key]; set => ((IDictionary<string, OptionBase>)_options)[key] = value; }

        /// <inheritdoc/>
        public override int FeatureSpaceDim => _options.Values.Select(x => x.FeatureSpaceDim).Sum();

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
        public override Parameter SampleFromFeatureSpace(double[] values)
        {
            var param = Parameter.CreateNestedParameter();
            var startIndex = 0;
            foreach (var kv in this.OrderBy(kv => kv.Key))
            {
                var dim = kv.Value.FeatureSpaceDim;
                var feature = values.Skip(startIndex).Take(dim).ToArray();
                param[kv.Key] = kv.Value.SampleFromFeatureSpace(feature);
                startIndex += dim;
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
        public ICollection<string> Keys => ((IDictionary<string, OptionBase>)_options).Keys;

        /// <inheritdoc/>
        public ICollection<OptionBase> Values => ((IDictionary<string, OptionBase>)_options).Values;

        /// <inheritdoc/>
        public int Count => ((ICollection<KeyValuePair<string, OptionBase>>)_options).Count;

        /// <inheritdoc/>
        public bool IsReadOnly => ((ICollection<KeyValuePair<string, OptionBase>>)_options).IsReadOnly;

        /// <inheritdoc/>
        public void Add(string key, OptionBase value)
        {
            ((IDictionary<string, OptionBase>)_options).Add(key, value);
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
        public bool ContainsKey(string key)
        {
            return ((IDictionary<string, OptionBase>)_options).ContainsKey(key);
        }

        /// <inheritdoc/>
        public void CopyTo(KeyValuePair<string, OptionBase>[] array, int arrayIndex)
        {
            ((ICollection<KeyValuePair<string, OptionBase>>)_options).CopyTo(array, arrayIndex);
        }

        /// <inheritdoc/>
        public IEnumerator<KeyValuePair<string, OptionBase>> GetEnumerator()
        {
            return ((IEnumerable<KeyValuePair<string, OptionBase>>)_options).GetEnumerator();
        }

        /// <inheritdoc/>
        public bool Remove(string key)
        {
            return ((IDictionary<string, OptionBase>)_options).Remove(key);
        }

        /// <inheritdoc/>
        public bool Remove(KeyValuePair<string, OptionBase> item)
        {
            return ((ICollection<KeyValuePair<string, OptionBase>>)_options).Remove(item);
        }

        /// <inheritdoc/>
        public bool TryGetValue(string key, out OptionBase value)
        {
            return ((IDictionary<string, OptionBase>)_options).TryGetValue(key, out value);
        }

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)_options).GetEnumerator();
        }
    }
}
