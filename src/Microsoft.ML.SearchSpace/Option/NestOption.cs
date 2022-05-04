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

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public OptionBase this[string key] { get => ((IDictionary<string, OptionBase>)_options)[key]; set => ((IDictionary<string, OptionBase>)_options)[key] = value; }

        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        public override int FeatureSpaceDim => _options.Values.Select(x => x.FeatureSpaceDim).Sum();

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
        public void Add(string key, OptionBase value)
        {
            ((IDictionary<string, OptionBase>)_options).Add(key, value);
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
        public bool ContainsKey(string key)
        {
            return ((IDictionary<string, OptionBase>)_options).ContainsKey(key);
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
        public IEnumerator<KeyValuePair<string, OptionBase>> GetEnumerator()
        {
            return ((IEnumerable<KeyValuePair<string, OptionBase>>)_options).GetEnumerator();
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
        public bool Remove(KeyValuePair<string, OptionBase> item)
        {
            return ((ICollection<KeyValuePair<string, OptionBase>>)_options).Remove(item);
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
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)_options).GetEnumerator();
        }
    }
}
